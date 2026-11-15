"""Bounded multi-step identification from frozen ground-phase trajectories, not PPO."""
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
p.add_argument('--phase',type=int,choices=[0,2],default=2)
p.add_argument('--resume',type=Path)
p.add_argument('--resume-optimizer',action='store_true')
p.add_argument('--retain-best',type=Path,help='Parent directory best.pt/best.json under the same fixed evaluation contract')
p.add_argument('--initial-update',type=int,default=0)
p.add_argument('--balanced-mixture',action='store_true')
p.add_argument('--terrain-features',action='store_true')
p.add_argument('--alpha-conditioned',action='store_true')
p.add_argument('--additional-evaluations',type=Path,nargs='*',default=[])
p.add_argument('--model-kind',choices=['residual','direct'],default='residual')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
started=time.perf_counter();device='cuda'
writer=SummaryWriter(str(a.output/'tensorboard'),flush_secs=10)
evaluations=[a.evaluation]+a.additional_evaluations
configs=[json.loads((source/'resolved_config.json').read_text()) for source in evaluations]
cfg=configs[0]
status=json.loads((a.evaluation/'status.json').read_text())
schema=json.loads((a.evaluation/'model_state_schema.json').read_text())
bank=np.load(cfg['ground_handoff']['reference_path'])
hcfg=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text())
follower=dict(hcfg['follower'],lookahead_scale=1. if a.phase==0 else 1.3)
nominal=dict(np.load(a.nominal))
terrain=None;terrain_specs=[];source_map_ids=[]
if a.terrain_features:
    from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec
    for config in configs:
        spec=config['ditch_terrain']
        if spec not in terrain_specs:terrain_specs.append(spec)
        source_map_ids.append(terrain_specs.index(spec))
    rasters=[raster_from_spec(spec) for spec in terrain_specs]
    terrain_data=dict(rasters[0])
    if len(rasters)>1:
        for key in ['height','support']:terrain_data[key]=np.stack([r[key] for r in rasters])
    terrain_data['spacing']=float(terrain_data['spacing'])
    np.savez(a.output/'terrain.npz',**terrain_data)
    terrain=TerrainFeatures(**terrain_data,device='cpu')
datasets=[];offset=0;window_rows={h:[] for h in [5,15,50,150]}
for source_id,source in enumerate(evaluations):
    status=json.loads((source/'status.json').read_text())
    for row in status['results']:
        if a.phase==0 and row['route']%3!=0:continue
        with np.load(source/row['case_dir']/'trajectory.npz') as data:
            pose,xi=states(data['qpos'][::4],data['qvel'][::4],schema)
            controller=data['controller_state'][::4];phase=data['trace'][::4,-1];requests=data['trace'][::4,18]
            ref=Reference(bank,row['route'],follower);eta=ref.features(pose,xi);eta[:,13]=requests
            inputs=np.column_stack((xi,eta,pose[:,10:12]))
            if terrain is not None:
                with torch.no_grad():inputs=np.column_stack((inputs,terrain(torch.tensor(pose,dtype=torch.float32),torch.full((len(pose),),source_map_ids[source_id],dtype=torch.long)).numpy()))
            stage_alpha=(configs[source_id]['ground_handoff']['approach']['alpha'] if a.phase==0 else row.get('continuation_alpha',configs[source_id]['ground_handoff']['continuation']['alpha'])) if a.alpha_conditioned else 0.
            if a.alpha_conditioned:inputs=np.column_stack((inputs,np.full(len(pose),stage_alpha)))
            datasets.append(dict(alpha=stage_alpha,row=row,pose=pose,xi=xi,controller=controller,inputs=inputs,phase=phase,requests=requests,terrain_id=source_map_ids[source_id] if terrain is not None else 0,offset=offset))
            for h in window_rows:
                indices=np.arange(1,len(pose)-h)
                bad=np.r_[0,np.cumsum(phase!=a.phase)]
                indices=indices[(bad[indices+h+1]-bad[indices-1])==0]
                if row['variant']['data_split']!='TRAIN':indices=indices[::10]
                window_rows[h].extend(dict(index=int(i+offset),route=row['route'],case=(f'source{source_id}/' if a.additional_evaluations else '')+row['case_dir'],evaluation=str(source),source_case=row['case_dir'],split=row['variant']['data_split'],anchor_s=float(i*.02)) for i in indices)
            offset+=len(pose)
train=[d for d in datasets if d['row']['variant']['data_split']=='TRAIN']
controller_train=np.concatenate([d['controller'][d['phase']==a.phase] for d in train])
inputs_train=np.concatenate([d['inputs'][d['phase']==a.phase] for d in train])
controller_mean=controller_train.mean(0);controller_scale=controller_train.std(0);controller_scale[controller_scale<1e-4]=1.
input_mean=inputs_train.mean(0);input_scale=inputs_train.std(0);input_scale[input_scale<1e-3]=1.
corrections=[]
for d in train:
    good=(d['phase'][1:-1]==a.phase)&(d['phase'][2:]==a.phase)&(d['phase'][:-2]==a.phase)
    x=d['xi'];eta=d['inputs'][:,9:24]
    corrections.append((x[2:]-predict(nominal,x[1:-1],x[:-2],eta[1:-1]))[good])
correction_scale=np.maximum(np.concatenate(corrections).std(0),[.05,.05,.1,.2,.2,.2,.3,1.,1.])
base=NominalRollout(nominal,follower,device)
if terrain is not None:terrain.to(device)
model=HistoryDynamics(base,controller_mean,controller_scale,input_mean,input_scale,correction_scale,terrain=terrain,model_kind=a.model_kind,alpha_conditioned=a.alpha_conditioned)
if a.resume:
    resumed=torch.load(a.resume,weights_only=True)
    model.load_state_dict(resumed['model'] if a.resume_optimizer else resumed)
tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
poses=tensor(np.concatenate([d['pose'] for d in datasets]));velocities=tensor(np.concatenate([d['xi'] for d in datasets]));controllers=tensor(np.concatenate([d['controller'] for d in datasets]))
requests=tensor(np.concatenate([d['requests'] for d in datasets]))
alphas=tensor(np.concatenate([np.full(len(d['pose']),d['alpha']) for d in datasets]))
terrain_ids=torch.as_tensor(np.concatenate([np.full(len(d['pose']),d['terrain_id']) for d in datasets]),device=device,dtype=torch.long)
path_s=tensor(bank['s']);path_xy=tensor(bank['xy']);path_h=tensor(bank['heading']);path_k=tensor(bank['curvature'])
train_windows={h:torch.tensor([[r['index'],r['route']] for r in rows if r['split']=='TRAIN'],device=device) for h,rows in window_rows.items()}
from collections import Counter
window_weights={}
for h,rows in window_rows.items():
    train_rows=[r for r in rows if r['split']=='TRAIN'];counts=Counter(r['case'] for r in train_rows)
    window_weights[h]=tensor([1./counts[r['case']] for r in train_rows])
dev_rows={h:[r for r in window_rows[h] if r['split']!='TRAIN'] for h in [50,150]}
shutil.copyfile(a.nominal,a.output/'nominal.npz')
contract=dict(phase=a.phase,route_selection="canonical downstream0 subset for each approach; other physical executions are not exact duplicates" if a.phase==0 else "all routes",source=str(a.evaluation),additional_evaluations=[str(x) for x in a.additional_evaluations],nominal=str(a.nominal),updates=a.updates,seed=a.seed,batch=64,
              initialization='model_and_Adam_resume_new_sampling_seed' if a.resume_optimizer else 'weights_only_fresh_Adam' if a.resume else 'zero_correction',parent_checkpoint=str(a.resume),retained_best=str(a.retain_best),initial_update=a.initial_update,balanced_mixture=a.balanced_mixture,
              checkpoints_every=25,selection='mean maximum XY over fixed available1s DEV windows',
              training='Declared source TRAIN only, includes pre-failure observations; no future real controller states in rollout',
              alpha_conditioned=a.alpha_conditioned,alpha_train_values=sorted({d['alpha'] for d in train}) if a.alpha_conditioned else None,terrain_features=a.terrain_features,terrain_specs=terrain_specs,source_map_ids=source_map_ids,model_kind=a.model_kind,
              terrain_descriptor='15yaw-aligned locations: height, x/y slopes, local height variation, support mask; raster model input not collision geometry' if terrain is not None else None,
              architecture='381D history initializes GRU64;128x128 head; residual adds to nominal, direct predicts full velocity increment with shared pose integration',
              objective='multi-step position/quaternion/velocity error; frozen RB/H75/physics; no policy updates',
              horizons_steps=[5,15,50,150],windows={str(h):dict(TRAIN=len(train_windows[h]),DEV=len([r for r in rows if r['split']!='TRAIN'])) for h,rows in window_rows.items()},
              follower=follower,reference=str(cfg['ground_handoff']['reference_path']))
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
for source in [Path(__file__),Path('learning/src/sttw_control/playground_history_model.py'),Path('learning/src/sttw_control/playground_prediction.py'),Path('learning/src/sttw_control/playground_terrain_features.py')]:shutil.copyfile(source,a.output/source.name)
def rollout(indices,routes,h):
    return model.rollout(poses[indices],velocities[indices],velocities[indices-1],controllers[indices],path_s,path_xy[routes],path_h[routes],path_k[routes],h,speed_request=requests[indices],terrain_ids=terrain_ids[indices],alpha_request=alphas[indices] if a.alpha_conditioned else None)

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
if a.resume_optimizer:optimizer.load_state_dict(resumed['optimizer'])
position_scale=tensor([.2,.2,.1]);velocity_scale=tensor([.3,.2,.3,.5,.5,.5,1.,5.,5.])
best=float('inf');logs=[]
if a.retain_best:
    best=json.loads((a.retain_best/'best.json').read_text())['score']
    for name in ['best.pt','best.json']:shutil.copyfile(a.retain_best/name,a.output/name)
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
    h=[5,15,50,150][iteration%4] if a.balanced_mixture else 5 if iteration<25 else 15 if iteration<50 else 50 if iteration<100 else 150
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
