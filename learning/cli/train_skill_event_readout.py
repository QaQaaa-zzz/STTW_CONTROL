"""Train cumulative5ms-event labels on frozen20ms free world-model predictions."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from sttw_control.skill_world_model import SkillWorldModel
from sttw_control.skill_event_readout import SkillEventReadout,event_labels
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--world-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=600);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();device='cuda';tensor=lambda x:torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32)
m=json.loads((a.data/'manifest.json').read_text());rows=m['rows'];z=np.load(a.data/'sequences.npz');n=np.load(a.data/'normalization.npz');starts=np.array([r['start'] for r in rows]);horizon=125;elapsed=np.arange(1,horizon+1)*.02
labels=[];event_times=[];valid=[];provenance=[]
for row in rows:
 d=np.load(row['path']);tr=d['trace'];contact=d['wheel_contact_flags'];mask=np.isin(tr[:,-1],[1,3])&(tr[:,1]>8.8)&~contact[:,0].astype(bool);idx=np.flatnonzero(mask)
 raw_takeoff=float(tr[idx[0],0]) if len(idx) else float('nan');far=float(row['first_far_contact_s']) if row['first_far_contact_s'] is not None else float('nan');origin=float(z['time'][row['start']]);relative=np.array([raw_takeoff,far])-origin
 labels.append(event_labels(elapsed,relative));event_times.append(relative);valid.append(elapsed<=float(z['time'][row['start']+row['length']-1])-origin+1e-5)
 provenance.append(dict(case=row['map']+'/'+row['case_dir'],split=row['split'],source=row['path'],entry_s=origin,takeoff_5ms_s=raw_takeoff if np.isfinite(raw_takeoff) else None,far_contact_s=far if np.isfinite(far) else None))
event_times=np.asarray(event_times);valid=tensor(valid);target=tensor(labels);train=torch.tensor([i for i,r in enumerate(rows) if r['split']=='TRAIN'],device=device);dev=torch.tensor([i for i,r in enumerate(rows) if r['split']=='DEV'],device=device)
world=SkillWorldModel.from_weights(torch.load(a.world_model,map_location=device,weights_only=True)['model']);world.eval()
with torch.no_grad():
 history=tensor(z['history'][starts]);context=tensor(z['context'][starts]);state=world.initialize(tensor(z['qpos'][starts]),tensor(z['qvel'][starts]),history[:,:381],history[:,381:],tensor(z['contact'][starts]),F.one_hot(torch.as_tensor(z['phase'][starts],device=device,dtype=torch.long),5).float(),context)
 prediction=world.rollout_until(state,context,horizon)
 age=tensor(elapsed)[None,:,None].expand(len(rows),-1,1)
 features=torch.cat((prediction['qpos'],prediction['qvel'],prediction['hidden'],age),-1).detach();selected=features[train][valid[train].bool()];mean=selected.mean(0);scale=selected.std(0).clamp_min(.1)
 # These cached features are predicted from the true entry only; no measured future is an input.
 del prediction,world
truth=tensor(event_times);event_finite=torch.isfinite(truth);transition=(age.squeeze(-1)[:,:,None]-truth[:,None,:]).abs()<=.04;weights=(1+4*transition.float())*valid[:,:,None]
contract=dict(world_model=str(a.world_model),world_model_sha256=hashlib.sha256(a.world_model.read_bytes()).hexdigest(),data=str(a.data),events=['first no-wheel-contact sample after RB entry andx>8.8','first far contact from recorded event'],labels='Cumulative event occurrence derived from5ms records, evaluated on20ms clock. Missing events stay all-zero; no motion interpolation.',features='Frozen predicted12qpos+11qvel+128hidden+elapsed since actual RB entry; no measured future or true event time input',normalization='TRAIN frozen-prediction features only',updates=a.updates,seeds=[0,1,2],batch=64,lr=3e-4,clip=1.,checkpoint_every=25,selection='DEV mean absolute event-time error; missing/spurious events penalized by2.5s horizon',unchanged='Physical model weights and phase handoff decisions unchanged; only event readout trained',role='Known topology DEV; not test or proof of sustained flight')
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n');(a.output/'labels.json').write_text(json.dumps(provenance,indent=2)+'\n')
median=np.nanmedian(event_times[train.cpu().numpy()],axis=0);dev_truth=event_times[dev.cpu().numpy()];(a.output/'constant_baseline.json').write_text(json.dumps(dict(train_median_event_s=median.tolist(),dev_mae_s=np.nanmean(np.abs(dev_truth-median),axis=0).tolist()))+'\n')

@torch.no_grad()
def evaluate(model):
 probability,_=model.predict(features[dev],torch.zeros(len(dev),2,device=device));hits=probability>=.5;found=hits.any(1);steps=hits.long().argmax(1)+1;times=steps*.02
 finite=event_finite[dev];error=torch.where(found&finite,(times-truth[dev]).abs(),torch.where(found==finite,0.,2.5));score=error.mean();return float(score.cpu()),probability.cpu().numpy(),found.cpu().numpy(),times.cpu().numpy(),error.cpu().numpy()

summaries=[]
for seed in range(3):
 torch.manual_seed(seed);out=a.output/f'seed{seed}';out.mkdir();writer=SummaryWriter(str(out/'tensorboard'));model=SkillEventReadout(mean,scale).to(device);opt=torch.optim.Adam(model.parameters(),lr=3e-4);best=float('inf');logs=[]
 for update in range(a.updates+1):
  if update%25==0 or update==a.updates:
   model.eval();score,prob,found,times,errors=evaluate(model);checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),seed=seed,update=update);torch.save(checkpoint,out/f'model_{update:04d}.pt');logs.append(dict(update=update,dev_event_score_s=score,wall_s=time.perf_counter()-started));writer.add_scalar('DEV/event_score_s',score,update)
   if score<best:
    best=score;torch.save(checkpoint,out/'best.pt');np.savez_compressed(out/'dev_predictions.npz',probability=prob,event_found=found,event_time_s=times,actual_time_s=dev_truth,elapsed_s=elapsed);(out/'best.json').write_text(json.dumps(dict(update=update,score_s=score,event_mae_with_missing_penalty_s=errors.mean(0).tolist(),missing=(np.isfinite(dev_truth)&~found).sum(0).tolist(),spurious=(~np.isfinite(dev_truth)&found).sum(0).tolist()),indent=2)+'\n')
   (out/'log.json').write_text(json.dumps(logs,indent=2)+'\n');writer.flush()
  if update==a.updates:break
  model.train();ids=train[torch.randint(len(train),(64,),device=device)];logits=model(features[ids]);loss=(F.binary_cross_entropy_with_logits(logits,target[ids],reduction='none')*weights[ids]).sum()/weights[ids].sum();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if update%25==0:writer.add_scalar('TRAIN/event_BCE',float(loss.detach().cpu()),update+1)
 torch.save(checkpoint,out/'last.pt');writer.close();summaries.append(dict(seed=seed,**json.loads((out/'best.json').read_text())));print(json.dumps(summaries[-1]),flush=True)
(a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n');(a.output/'status.json').write_text(json.dumps(dict(status='complete',stage='P3 cumulative contact event readout',wall_s=time.perf_counter()-started))+'\n')
