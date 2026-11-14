"""Bounded multi-step identification from frozen phase2 trajectories, not PPO."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sttw_control.playground_arx import Reference,states,predict
from sttw_control.playground_prediction import NominalRollout
from sttw_control.playground_history_model import HistoryDynamics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--nominal',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--updates',type=int,default=150)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--resume',type=Path)
p.add_argument('--initial-update',type=int,default=0)
p.add_argument('--balanced-mixture',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
started=time.perf_counter();device='cuda'
writer=SummaryWriter(str(a.output/'tensorboard'),flush_secs=10)
cfg=json.loads((a.evaluation/'resolved_config.json').read_text())
status=json.loads((a.evaluation/'status.json').read_text())
schema=json.loads((a.evaluation/'model_state_schema.json').read_text())
bank=np.load(cfg['ground_handoff']['reference_path'])
hcfg=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text())
follower=dict(hcfg['follower'],lookahead_scale=1.3)
nominal=dict(np.load(a.nominal))
datasets=[];offset=0;window_rows={h:[] for h in [5,15,50,150]}
for row in status['results']:
    with np.load(a.evaluation/row['case_dir']/'trajectory.npz') as data:
        pose,xi=states(data['qpos'][::4],data['qvel'][::4],schema)
        controller=data['controller_state'][::4];phase=data['trace'][::4,-1]
        ref=Reference(bank,row['route'],follower);eta=ref.features(pose,xi)
        inputs=np.column_stack((xi,eta,pose[:,10:12]))
        datasets.append(dict(row=row,pose=pose,xi=xi,controller=controller,inputs=inputs,phase=phase,offset=offset))
        for h in window_rows:
            indices=np.arange(1,len(pose)-h)
            bad=np.r_[0,np.cumsum(phase!=2)]
            indices=indices[(bad[indices+h+1]-bad[indices-1])==0]
            if row['variant']['data_split']!='TRAIN':indices=indices[::10]
            window_rows[h].extend(dict(index=int(i+offset),route=row['route'],case=row['case_dir'],split=row['variant']['data_split'],anchor_s=float(i*.02)) for i in indices)
        offset+=len(pose)
train=[d for d in datasets if d['row']['variant']['data_split']=='TRAIN']
controller_train=np.concatenate([d['controller'][d['phase']==2] for d in train])
inputs_train=np.concatenate([d['inputs'][d['phase']==2] for d in train])
controller_mean=controller_train.mean(0);controller_scale=controller_train.std(0);controller_scale[controller_scale<1e-4]=1.
input_mean=inputs_train.mean(0);input_scale=inputs_train.std(0);input_scale[input_scale<1e-3]=1.
corrections=[]
for d in train:
    good=(d['phase'][1:-1]==2)&(d['phase'][2:]==2)&(d['phase'][:-2]==2)
    x=d['xi'];eta=d['inputs'][:,9:24]
    corrections.append((x[2:]-predict(nominal,x[1:-1],x[:-2],eta[1:-1]))[good])
correction_scale=np.maximum(np.concatenate(corrections).std(0),[.05,.05,.1,.2,.2,.2,.3,1.,1.])
base=NominalRollout(nominal,follower,device)
model=HistoryDynamics(base,controller_mean,controller_scale,input_mean,input_scale,correction_scale)
if a.resume:model.load_state_dict(torch.load(a.resume,weights_only=True))
tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
poses=tensor(np.concatenate([d['pose'] for d in datasets]));velocities=tensor(np.concatenate([d['xi'] for d in datasets]));controllers=tensor(np.concatenate([d['controller'] for d in datasets]))
path_s=tensor(bank['s']);path_xy=tensor(bank['xy']);path_h=tensor(bank['heading']);path_k=tensor(bank['curvature'])
train_windows={h:torch.tensor([[r['index'],r['route']] for r in rows if r['split']=='TRAIN'],device=device) for h,rows in window_rows.items()}
from collections import Counter
window_weights={}
for h,rows in window_rows.items():
    train_rows=[r for r in rows if r['split']=='TRAIN'];counts=Counter(r['case'] for r in train_rows)
    window_weights[h]=tensor([1./counts[r['case']] for r in train_rows])
dev_rows={h:[r for r in window_rows[h] if r['split']!='TRAIN'] for h in [50,150]}
shutil.copyfile(a.nominal,a.output/'nominal.npz')
contract=dict(source=str(a.evaluation),nominal=str(a.nominal),updates=a.updates,seed=a.seed,batch=64,
              initialization='weights_only_fresh_Adam' if a.resume else 'zero_correction',parent_checkpoint=str(a.resume),initial_update=a.initial_update,balanced_mixture=a.balanced_mixture,
              checkpoints_every=25,selection='mean maximum XY over fixed available1s DEV windows',
              training='flat TRAIN only, includes pre-failure observations; no future real controller states in rollout',
              architecture='381D controller/history initializes64D hidden, GRU64,128x128 correction to nominal next dynamics',
              objective='multi-step position/quaternion/velocity error; frozen RB/H75/physics; no policy updates',
              horizons_steps=[5,15,50,150],windows={str(h):dict(TRAIN=len(train_windows[h]),DEV=len([r for r in rows if r['split']!='TRAIN'])) for h,rows in window_rows.items()},
              follower=follower,reference=str(cfg['ground_handoff']['reference_path']))
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
for source in [Path(__file__),Path('learning/src/sttw_control/playground_history_model.py'),Path('learning/src/sttw_control/playground_prediction.py')]:shutil.copyfile(source,a.output/source.name)
def rollout(indices,routes,h):
    return model.rollout(poses[indices],velocities[indices],velocities[indices-1],controllers[indices],path_s,path_xy[routes],path_h[routes],path_k[routes],h)

@torch.no_grad()
def evaluate(save=False):
    records=[];examples={}
    for h,rows in dev_rows.items():
        for start in range(0,len(rows),64):
            batch=rows[start:start+64];ids=torch.tensor([r['index'] for r in batch],device=device);routes=torch.tensor([r['route'] for r in batch],device=device)
            predicted,_=rollout(ids,routes,h);actual=poses[ids[:,None]+torch.arange(h+1,device=device)]
            errors=(predicted[:,:,:2]-actual[:,:,:2]).norm(dim=-1).amax(1).cpu().numpy()
            endpoints=(predicted[:,-1,:2]-actual[:,-1,:2]).norm(dim=-1).cpu().numpy()
            for j,row in enumerate(batch):
                records.append(dict(row,horizon_s=h*.02,max_xy_m=float(errors[j]),endpoint_xy_m=float(endpoints[j])))
                key=f"{row['case']}_h{h}"
                if save and key+'_actual' not in examples:
                    examples[key+'_actual']=actual[j].cpu().numpy();examples[key+'_prediction']=predicted[j].cpu().numpy()
    metrics={str(h*.02):float(np.mean([r['max_xy_m'] for r in records if r['horizon_s']==h*.02])) for h in dev_rows}
    if save:
        (a.output/'dev_windows.json').write_text(json.dumps(records,indent=2)+'\n');np.savez_compressed(a.output/'dev_examples.npz',**examples)
    return metrics

optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
position_scale=tensor([.2,.2,.1]);velocity_scale=tensor([.3,.2,.3,.5,.5,.5,1.,5.,5.])
best=float('inf');logs=[]
for update in range(a.updates+1):
    iteration=a.initial_update+update
    if update==0 or update%25==0 or update==a.updates:
        model.eval();metrics=evaluate();score=metrics['1.0']
        for horizon,value in metrics.items():writer.add_scalar('DEV/mean_max_xy_'+horizon+'s',value,iteration)
        writer.flush()
        torch.save(model.state_dict(),a.output/f'model_{iteration:04d}.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration),a.output/'training_state.pt')
        if np.isfinite(score) and score<best:
            best=score;torch.save(model.state_dict(),a.output/'best.pt');(a.output/'best.json').write_text(json.dumps(dict(update=iteration,score=score,criterion=contract['selection']),indent=2)+'\n')
        log=dict(update=iteration,kind='validation',dev_mean_max_xy_m=metrics,elapsed_s=time.perf_counter()-started);logs.append(log)
        with (a.output/'scalars.jsonl').open('a') as stream:stream.write(json.dumps(log)+'\n')
        (a.output/'status.json').write_text(json.dumps(dict(stage='training',update=iteration,total=a.initial_update+a.updates,last_validation=metrics,elapsed_s=time.perf_counter()-started),indent=2)+'\n')
        model.train()
    if update==a.updates:break
    h=[5,15,50,150][update%4] if a.balanced_mixture else 5 if iteration<25 else 15 if iteration<50 else 50 if iteration<100 else 150
    candidates=train_windows[h]
    selection=torch.multinomial(window_weights[h],64,replacement=True) if a.balanced_mixture else torch.randint(len(candidates),(64,),device=device)
    sample=candidates[selection];ids=sample[:,0];routes=sample[:,1]
    optimizer.zero_grad(set_to_none=True)
    predicted,speed=rollout(ids,routes,h);target_ids=ids[:,None]+torch.arange(h+1,device=device)
    actual=poses[target_ids];actual_speed=velocities[target_ids]
    position_loss=((predicted[:,1:,:3]-actual[:,1:,:3])/position_scale).square().mean()
    rotation_loss=(1-(predicted[:,1:,3:7]*actual[:,1:,3:7]).sum(-1).square()).mean()
    velocity_loss=((speed[:,1:]-actual_speed[:,1:])/velocity_scale).square().mean()
    loss=position_loss+10*rotation_loss+.25*velocity_loss
    loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    log=dict(update=iteration+1,kind='training',horizon_steps=h,loss=float(loss.detach()),position_loss=float(position_loss.detach()),velocity_loss=float(velocity_loss.detach()),gradient_norm=float(grad),elapsed_s=time.perf_counter()-started)
    with (a.output/'scalars.jsonl').open('a') as stream:stream.write(json.dumps(log)+'\n')
    for key in ['loss','position_loss','velocity_loss','gradient_norm']:writer.add_scalar('TRAIN/'+key,log[key],iteration+1)
model.load_state_dict(torch.load(a.output/'best.pt',weights_only=True));model.eval();metrics=evaluate(save=True)
(a.output/'status.json').write_text(json.dumps(dict(stage='complete',updates=a.updates,total_updates=a.initial_update+a.updates,best=json.loads((a.output/'best.json').read_text()),dev_metrics=metrics,elapsed_s=time.perf_counter()-started),indent=2)+'\n')
writer.close()
print((a.output/'status.json').read_text())
