"""Bounded supervised full-joint RB model training; frozen control skills."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sttw_control.skill_world_model import SkillWorldModel,JOINT_POSITION_SCALES,JOINT_VELOCITY_SCALES

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=150)
p.add_argument('--resume',type=Path);p.add_argument('--horizons',type=int,nargs='+',default=[1,5,15,50]);p.add_argument('--balance-phases',action='store_true');p.add_argument('--full-entry',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.manual_seed(a.seed);device='cuda';started=time.perf_counter()
manifest=json.loads((a.data/'manifest.json').read_text());raw=np.load(a.data/'sequences.npz');normal=np.load(a.data/'normalization.npz')
tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
data={k:tensor(raw[k]) for k in ['qpos','qvel','history','contact','context']};phase=torch.as_tensor(raw['phase'],device=device,dtype=torch.long)
model=SkillWorldModel(537,5,*[normal[k] for k in ['input_mean','input_scale','velocity_scale','history_scale']]).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);writer=SummaryWriter(str(a.output/'tensorboard'))
initial_update=0
if a.resume:
 resumed=torch.load(a.resume,map_location=device,weights_only=True);model.load_state_dict(resumed['model']);optimizer.load_state_dict(resumed['optimizer']);initial_update=resumed['update']
horizons=a.horizons;train={};dev={}
for h in horizons:
 train[h]=torch.tensor([i for r in manifest['rows'] if r['split']=='TRAIN' for i in range(r['start'],r['start']+r['length']-h)],device=device)
 dev[h]=torch.tensor([i for r in manifest['rows'] if r['split']=='DEV' for i in range(r['start'],r['start']+r['length']-h,max(1,h//2))],device=device)
assert all(len(train[h]) and len(dev[h]) for h in horizons)
full_groups={}
for split in ['TRAIN','DEV']:
 full_groups[split]={}
 for row in manifest['rows']:
  if row['split']==split:full_groups[split].setdefault(row['length']-1,[]).append(row['start'])
 full_groups[split]={h:torch.tensor(ids,device=device) for h,ids in full_groups[split].items()}
full_lengths=sorted(full_groups['TRAIN'])
schema_path=Path(manifest['rows'][0]['evaluation'])/'model_state_schema.json';schema=json.loads(schema_path.read_text())
vscale_values=[.3,.3,.3,.5,.5,.5]+[JOINT_VELOCITY_SCALES[name] for name in schema['qvel_names'][6:]]
jscale_values=[JOINT_POSITION_SCALES[name] for name in schema['qpos_names'][7:]]
hscale=tensor(normal['history_state_scale']);vscale=tensor(vscale_values);jscale=tensor(jscale_values);pscale=tensor([.2,.2,.1])
train_indices=torch.tensor([i for r in manifest['rows'] if r['split']=='TRAIN' for i in range(r['start'],r['start']+r['length'])],device=device)
counts=torch.bincount(phase[train_indices],minlength=5).float();phase_weights=torch.where(counts>0,counts.sum()/counts.clamp_min(1),torch.zeros_like(counts));phase_weights=phase_weights/phase_weights[phase_weights>0].mean() if a.balance_phases else torch.ones_like(counts)
contract=dict(rotation_scale_rad=.2,handoff_loss='Mean normalized physical error at actual phase2 transition, weight1; labels only, not rollout inputs',state_schema=str(schema_path),velocity_loss_scale=vscale_values,joint_loss_scale=jscale_values,full_entry=a.full_entry,resume=str(a.resume),initial_update=initial_update,balance_phases=a.balance_phases,phase_weights=phase_weights.cpu().tolist(),seed=a.seed,updates=a.updates,data=str(a.data),batch=64,lr=3e-4,gradient_clip=1.,horizons_steps=horizons,checkpoint_every=25,selection='DEV mean normalized multistep physical/history/contact/phase loss, including transition and endpoint weighting; decision-quality evaluation still required',loss_weights=dict(position=1,rotation=1,velocity=1,joints=1,history=.1,contact=.2,phase=.2,endpoint_position=1),transition_weight=3,normalization='TRAIN only',initialization='model and Adam resume; new sampling RNG' if a.resume else 'random model; no control policy training',device='GPU model and resident tensors; scalar/trajectory transfers only at checkpoint evaluation',stop='finite updates or outer25min wall budget; no automatic extension',history='Predicted controller/RB estimates after initial state; no measured future injected')
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')

def predict(ids,h):
 state=model.initialize(data['qpos'][ids],data['qvel'][ids],data['history'][ids,:381],data['history'][ids,381:],data['contact'][ids],F.one_hot(phase[ids],5).float(),data['context'][ids])
 return model.rollout(state,data['context'][ids],h)

def losses(pred,ids,h):
 idx=ids[:,None]+torch.arange(1,h+1,device=device);q=data['qpos'][idx];v=data['qvel'][idx];pq=pred['qpos'][:,1:];pv=pred['qvel'][:,1:]
 transition=(phase[idx]!=phase[idx-1])|((data['contact'][idx]-data['contact'][idx-1]).abs().sum(-1)>0)
 weight=1+2*transition.float()
 xyz=(((pq[:,:,:3]-q[:,:,:3])/pscale).square().mean(-1)*weight).mean()
 rotation_error=(1-(pq[:,:,3:7]*q[:,:,3:7]).sum(-1).square())*100.
 rotation=(rotation_error*weight).mean()
 velocity=(((pv-v)/vscale).square().mean(-1)*weight).mean()
 joints=(((pq[:,:,7:]-q[:,:,7:])/jscale).square().mean(-1)*weight).mean()
 history=((pred['history'][:,1:]-data['history'][idx])/hscale).square().mean()
 contact=(F.binary_cross_entropy_with_logits(pred['contact_logits'],data['contact'][idx],reduction='none').mean(-1)*weight).mean()
 skill=(F.cross_entropy(pred['phase_logits'].flatten(0,1),phase[idx].flatten(),weight=phase_weights,reduction='none').reshape_as(weight)*weight).mean()
 endpoint=((pq[:,-1,:3]-q[:,-1,:3])/pscale).square().mean()
 handoff_mask=((phase[idx]==2)&(phase[idx-1]!=2)).float()
 handoff_error=((pq[:,:,:3]-q[:,:,:3])/pscale).square().mean(-1)+rotation_error+((pv-v)/vscale).square().mean(-1)+((pq[:,:,7:]-q[:,:,7:])/jscale).square().mean(-1)
 handoff=(handoff_error*handoff_mask).sum()/handoff_mask.sum().clamp_min(1)
 total=xyz+rotation+velocity+joints+.1*history+.2*contact+.2*skill+endpoint+handoff
 return total,dict(position=xyz,rotation=rotation,velocity=velocity,joints=joints,history=history,contact=contact,phase=skill,endpoint=endpoint,handoff=handoff)

@torch.no_grad()
def evaluate():
 scores=[];metrics={}
 for h in horizons:
  accum=[]
  for batch in dev[h].split(64):
   pred=predict(batch,h);loss,parts=losses(pred,batch,h);idx=batch[:,None]+torch.arange(1,h+1,device=device)
   xyz=(pred['qpos'][:,1:,:3]-data['qpos'][idx,:3]).norm(dim=-1)
   joints=(pred['qpos'][:,1:,7:]-data['qpos'][idx,7:]).abs()
   event_error=(pred['phase_logits'].argmax(-1)!=phase[idx]).float().mean()
   accum.append(torch.stack([loss,xyz.mean(),xyz.amax(),joints.mean(),event_error])*len(batch))
  values=(torch.stack(accum).sum(0)/len(dev[h])).cpu().tolist()
  metrics[str(h)]=dict(zip(['loss','xyz_mean_m','batch_max_xyz_mean_m','joint_mae_rad','phase_error_fraction'],values));scores.append(values[0])
 if a.full_entry:
  accum=[];count=0
  for h,ids in full_groups['DEV'].items():
   for batch in ids.split(64):
    pred=predict(batch,h);loss,_=losses(pred,batch,h);accum.append(loss*len(batch));count+=len(batch)
  full_score=float((torch.stack(accum).sum()/count).cpu());metrics['full_entry']={'loss':full_score};return full_score,metrics
 return float(np.mean(scores)),metrics

logs=[];best=float('inf')
for offset in range(a.updates+1):
 update=initial_update+offset
 if offset%25==0 or offset==a.updates:
  model.eval();score,metrics=evaluate();checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),update=update,seed=a.seed)
  torch.save(checkpoint,a.output/f'model_{update:04d}.pt')
  if score<best:
   best=score;torch.save(checkpoint,a.output/'best.pt');(a.output/'best.json').write_text(json.dumps(dict(update=update,score=score,metrics=metrics),indent=2)+'\n')
  logs.append(dict(update=update,wall_s=time.perf_counter()-started,dev_score=score,metrics=metrics))
  for h,values in metrics.items():
   for key,value in values.items():writer.add_scalar(f'DEV/{h}_steps/{key}',value,update)
  (a.output/'log.json').write_text(json.dumps(logs,indent=2)+'\n');writer.flush();print(json.dumps(logs[-1]),flush=True)
 if offset==a.updates:break
 model.train();h=horizons[update%len(horizons)];ids=train[h][torch.randint(len(train[h]),(64,),device=device)]
 if a.full_entry and offset%2==0:
  h=full_lengths[(offset//2)%len(full_lengths)];pool=full_groups['TRAIN'][h];ids=pool[torch.randint(len(pool),(64,),device=device)]
 pred=predict(ids,h);loss,parts=losses(pred,ids,h);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
 # Log detached GPU scalars in one transfer at the scheduled checkpoint cadence.
 if update%25==0:
  values=torch.stack([loss.detach()]+[x.detach() for x in parts.values()]).cpu().tolist()
  for name,value in zip(['total']+list(parts),values):writer.add_scalar('TRAIN/'+name,value,update+1)
torch.save(checkpoint,a.output/'last.pt');writer.close()
(a.output/'status.json').write_text(json.dumps(dict(status='complete',stage='P3 supervised pilot',updates=a.updates,final_update=initial_update+a.updates,best_score=best,wall_s=time.perf_counter()-started))+'\n')
