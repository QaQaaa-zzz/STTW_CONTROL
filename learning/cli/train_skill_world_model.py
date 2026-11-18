"""Bounded supervised full-joint RB model training; frozen control skills."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sttw_control.skill_world_model import SkillWorldModel,JOINT_POSITION_SCALES,JOINT_VELOCITY_SCALES,handoff_focus_loss

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=150)
p.add_argument('--ordered-phases',action='store_true');p.add_argument('--final-learning-rate',type=float);p.add_argument('--handoff-focus-weight',type=float,default=0.);p.add_argument('--learning-rate',type=float);p.add_argument('--joint-displacement',action='store_true');p.add_argument('--entry-predictions',type=Path);p.add_argument('--lock-loss-weight',type=float,default=0.);p.add_argument('--resume',type=Path);p.add_argument('--horizons',type=int,nargs='+',default=[1,5,15,50]);p.add_argument('--balance-phases',action='store_true');p.add_argument('--full-entry',action='store_true')
a=p.parse_args()
if a.final_learning_rate is not None and a.updates<2:p.error('Cosine decay requires at least2 updates')
a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.manual_seed(a.seed);device='cuda';started=time.perf_counter()
manifest=json.loads((a.data/'manifest.json').read_text());raw=np.load(a.data/'sequences.npz');normal=np.load(a.data/'normalization.npz')
tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
data={k:tensor(raw[k]) for k in ['qpos','qvel','history','contact','context']};phase=torch.as_tensor(raw['phase'],device=device,dtype=torch.long)
predicted_data={};is_entry=torch.zeros(len(phase),device=device,dtype=torch.bool)
if a.entry_predictions:
 predicted=np.load(a.entry_predictions);starts=torch.as_tensor(predicted['start'],device=device,dtype=torch.long);is_entry[starts]=True
 for key in ['qpos','qvel','history']:
  predicted_data[key]=data[key].clone();predicted_data[key][starts]=tensor(predicted[key])
initial_update=0
if a.resume:
 resumed=torch.load(a.resume,map_location=device,weights_only=True);model=SkillWorldModel.from_weights(resumed['model']);initial_update=resumed['update']
else:
 model=SkillWorldModel(537,5,*[normal[k] for k in ['input_mean','input_scale','velocity_scale','history_scale']]).to(device)
optimizer=torch.optim.Adam([v for k,v in model.named_parameters() if not k.startswith('joint_displacement.')],lr=3e-4)
if hasattr(model,'joint_displacement'):optimizer.add_param_group(dict(params=model.joint_displacement.parameters()))
if a.resume:optimizer.load_state_dict(resumed['optimizer'])
if a.joint_displacement and not hasattr(model,'joint_displacement'):
 train_pairs=torch.tensor([i for r in manifest['rows'] if r['split']=='TRAIN' for i in range(r['start'],r['start']+r['length']-1)],device=device)
 defect=data['qpos'][train_pairs+1,7:]-data['qpos'][train_pairs,7:]-.5*model.dt*(data['qvel'][train_pairs,6:]+data['qvel'][train_pairs+1,6:])
 scale=defect.square().mean(0).sqrt().clamp_min(1e-5)
 model.enable_joint_displacement(scale);optimizer.add_param_group(dict(params=model.joint_displacement.parameters()))
if a.learning_rate is not None:
 for group in optimizer.param_groups:group['lr']=a.learning_rate
if a.ordered_phases:model.enable_ordered_phases()
initial_lrs=[group['lr'] for group in optimizer.param_groups]
writer=SummaryWriter(str(a.output/'tensorboard'))
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
contract=dict(alpha_train_values=sorted({float(raw['context'][r['start'],4]) for r in manifest['rows'] if r['split']=='TRAIN'}),ordered_phases=hasattr(model,'phase_successor'),phase_supervision='Raw logits CE; constrained probabilities feed next state when ordered',final_learning_rate=a.final_learning_rate,learning_rate_schedule='cosine over this continuation' if a.final_learning_rate is not None else 'constant',handoff_focus_weight=a.handoff_focus_weight,handoff_focus='Balanced phase2-vs-rest BCE plus actual-onset mean squared world-y/.05m, yaw/.05rad, world-vy/.1m_s; unchanged stopping thresholds',joint_displacement=hasattr(model,'joint_displacement'),joint_displacement_scale=model.joint_displacement_scale.cpu().tolist() if hasattr(model,'joint_displacement') else None,joint_displacement_normalization='RMS one-step quadrature defect from within-rollout TRAIN pairs only; zero initialized head',entry_predictions=str(a.entry_predictions) if a.entry_predictions else None,entry_exposure='50percent TRAIN true-start batches replaced by frozen upstream estimated physical/history; DEV true and predicted starts both scored' if a.entry_predictions else None,lock_loss_weight=a.lock_loss_weight,lock_scales='hip/knee0.01rad and0.1rad/s at actual transition labels',rotation_scale_rad=.2,handoff_loss='Mean normalized physical error at actual phase2 transition, weight1; labels only, not rollout inputs',state_schema=str(schema_path),velocity_loss_scale=vscale_values,joint_loss_scale=jscale_values,full_entry=a.full_entry,resume=str(a.resume),initial_update=initial_update,balance_phases=a.balance_phases,phase_weights=phase_weights.cpu().tolist(),seed=a.seed,updates=a.updates,data=str(a.data),batch=64,lr=optimizer.param_groups[0]['lr'],gradient_clip=1.,horizons_steps=horizons,checkpoint_every=25,selection='DEV mean normalized multistep physical/history/contact/phase loss, including transition and endpoint weighting; decision-quality evaluation still required',loss_weights=dict(position=1,rotation=1,velocity=1,joints=1,history=.1,contact=.2,phase=.2,endpoint_position=1),transition_weight=3,normalization='TRAIN only',initialization='model and Adam resume; new sampling RNG' if a.resume else 'random model; no control policy training',device='GPU model and resident tensors; scalar/trajectory transfers only at checkpoint evaluation',stop='finite updates or outer25min wall budget; no automatic extension',history='Predicted controller/RB estimates after initial state; no measured future injected')
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')

def predict(ids,h,use_predicted=False):
 initial={key:data[key][ids] for key in ['qpos','qvel','history']}
 if a.entry_predictions:
  mask=is_entry[ids] & (torch.rand(len(ids),device=device)<.5 if model.training else torch.full_like(is_entry[ids],use_predicted))
  for key in initial:initial[key]=torch.where(mask[:,None],predicted_data[key][ids],initial[key])
 state=model.initialize(initial['qpos'],initial['qvel'],initial['history'][:,:381],initial['history'][:,381:],data['contact'][ids],F.one_hot(phase[ids],5).float(),data['context'][ids])
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
 skill=(F.cross_entropy(pred['raw_phase_logits'].flatten(0,1),phase[idx].flatten(),weight=phase_weights,reduction='none').reshape_as(weight)*weight).mean()
 endpoint=((pq[:,-1,:3]-q[:,-1,:3])/pscale).square().mean()
 handoff_mask=((phase[idx]==2)&(phase[idx-1]!=2)).float()
 handoff_error=((pq[:,:,:3]-q[:,:,:3])/pscale).square().mean(-1)+rotation_error+((pv-v)/vscale).square().mean(-1)+((pq[:,:,7:]-q[:,:,7:])/jscale).square().mean(-1)
 handoff=(handoff_error*handoff_mask).sum()/handoff_mask.sum().clamp_min(1)
 joint_ids=[schema['qpos_names'].index(k) for k in ['hip_joint','knee_joint']];rate_ids=[schema['qvel_names'].index(k) for k in ['hip_joint','knee_joint']]
 lock_error=((pq[:,:,joint_ids]-q[:,:,joint_ids])/.01).square().mean(-1)+((pv[:,:,rate_ids]-v[:,:,rate_ids])/.1).square().mean(-1)
 lock=(lock_error*handoff_mask).sum()/handoff_mask.sum().clamp_min(1)
 focus_event,focus_exit=handoff_focus_loss(pq,pv,pred['phase_logits'],q,v,phase[idx],handoff_mask) if a.handoff_focus_weight else (xyz.new_zeros(()),xyz.new_zeros(()))
 total=xyz+rotation+velocity+joints+.1*history+.2*contact+.2*skill+endpoint+handoff+a.lock_loss_weight*lock+a.handoff_focus_weight*(focus_event+focus_exit)
 return total,dict(position=xyz,rotation=rotation,velocity=velocity,joints=joints,history=history,contact=contact,phase=skill,endpoint=endpoint,handoff=handoff,lock=lock,focus_event=focus_event,focus_exit=focus_exit)

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
  full_score=float((torch.stack(accum).sum()/count).cpu());metrics['full_entry']={'loss':full_score}
  if a.entry_predictions:
   accum=[]
   for h,ids in full_groups['DEV'].items():
    for batch in ids.split(64):
     pred=predict(batch,h,use_predicted=True);loss,_=losses(pred,batch,h);accum.append(loss*len(batch))
   predicted_score=float((torch.stack(accum).sum()/count).cpu());metrics['predicted_entry']={'loss':predicted_score};full_score=(full_score+predicted_score)*.5
  return full_score,metrics
 return float(np.mean(scores)),metrics

logs=[];best=float('inf')
for offset in range(a.updates+1):
 update=initial_update+offset
 if offset%25==0 or offset==a.updates:
  model.eval();score,metrics=evaluate();checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),update=update,seed=a.seed)
  torch.save(checkpoint,a.output/f'model_{update:04d}.pt')
  if score<best:
   best=score;torch.save(checkpoint,a.output/'best.pt');(a.output/'best.json').write_text(json.dumps(dict(update=update,score=score,metrics=metrics),indent=2)+'\n')
  writer.add_scalar('TRAIN/learning_rate',optimizer.param_groups[0]['lr'],update)
  logs.append(dict(update=update,learning_rate=optimizer.param_groups[0]['lr'],wall_s=time.perf_counter()-started,dev_score=score,metrics=metrics))
  for h,values in metrics.items():
   for key,value in values.items():writer.add_scalar(f'DEV/{h}_steps/{key}',value,update)
  (a.output/'log.json').write_text(json.dumps(logs,indent=2)+'\n');writer.flush();print(json.dumps(logs[-1]),flush=True)
 if offset==a.updates:break
 if a.final_learning_rate is not None:
  fraction=.5*(1+math.cos(math.pi*offset/(a.updates-1)))
  for group,lr in zip(optimizer.param_groups,initial_lrs):group['lr']=a.final_learning_rate+(lr-a.final_learning_rate)*fraction
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
