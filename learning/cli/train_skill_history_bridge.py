"""Bounded ground-to-RB history estimation; true history only at ground start."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sttw_control.playground_arx import states
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec
from sttw_control.skill_history_bridge import SkillHistoryBridge,bridge_features
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--ground',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=600);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();device='cuda';tensor=lambda x:torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32)
entries=json.loads((a.collection/'manifest.json').read_text())['entries'];rows=[];features=[];targets=[];masks=[];pose_errors=[];offset=0
model=load_history_model(a.ground)
for entry in entries:
 source=Path(entry['evaluation']);cfg=json.loads((source/'resolved_config.json').read_text());schema=json.loads((source/'model_state_schema.json').read_text());bank=np.load(cfg['ground_handoff']['reference_path']);records=json.loads((source/'status.json').read_text())['results'];items=[]
 for row in records:
  if row['route']%3:continue
  z=np.load(source/row['case_dir']/'trajectory.npz');j=int(np.flatnonzero(z['trace'][:,-1]==1)[0]);ids=np.arange(0,j+1,4);pose,xi=states(z['qpos'][ids],z['qvel'][ids],schema);h=np.concatenate((z['controller_state'][ids],z['rb_state'][ids]),-1)
  items.append(dict(row=row,pose=pose,velocity=xi,history=h,speed=float(z['trace'][0,18]),source=str(source/row['case_dir']/'trajectory.npz')))
 path_s=tensor(bank['s']);path_xy=tensor(bank['xy'][[x['row']['route'] for x in items]]);path_heading=tensor(bank['heading'][[x['row']['route'] for x in items]]);path_curvature=tensor(bank['curvature'][[x['row']['route'] for x in items]]);speed=tensor([x['speed'] for x in items])
 horizon=max(len(x['pose']) for x in items)-1;n=len(items);routes=[x['row']['route'] for x in items];initial=tensor([x['history'][0] for x in items]);pose=tensor([x['pose'][0] for x in items]);velocity=tensor([x['velocity'][0] for x in items]);state=model.initialize(pose,velocity,velocity,initial[:,:381])
 if model.terrain is not None:model.terrain=TerrainFeatures(**raster_from_spec(cfg['ditch_terrain']),device=device)
 with torch.no_grad():
  fs=[];ps=[]
  for step in range(horizon+1):
   fs.append(bridge_features(state,initial));ps.append(state.pose)
   if step<horizon:state=model.rollout_state(state,path_s,path_xy,path_heading,path_curvature,1,speed_request=speed)['next_state']
  f=torch.stack(fs,1);pred=torch.stack(ps,1)
 for lane,item in enumerate(items):
  length=len(item['history']);features.append(f[lane,:length]);targets.append(tensor(item['history']));error=(pred[lane,:length,:3]-tensor(item['pose'])[:,:3]).norm(dim=-1).cpu().numpy();pose_errors.append(dict(case=entry['layout']+'/'+item['row']['case_dir'],max_xyz_m=float(error.max()),endpoint_xyz_m=float(error[-1])))
  rows.append(dict(case=entry['layout']+'/'+item['row']['case_dir'],split='TRAIN' if entry['map_role']=='TRAIN' else 'DEV',start=offset,length=length,source=item['source']));offset+=length
features=torch.cat(features);target=torch.cat(targets);train=torch.tensor([i for r in rows if r['split']=='TRAIN' for i in range(r['start'],r['start']+r['length'])],device=device);dev=torch.tensor([i for r in rows if r['split']=='DEV' for i in range(r['start'],r['start']+r['length'])],device=device);dev_end=torch.tensor([r['start']+r['length']-1 for r in rows if r['split']=='DEV'],device=device)
mean=features[train].mean(0);scale=features[train].std(0).clamp_min(.1);tmean=target[train].mean(0);tscale=target[train].std(0).clamp_min(.1)
contract=dict(ground=str(a.ground),ground_sha256=hashlib.sha256((a.ground/'best.pt').read_bytes()).hexdigest(),collection=str(a.collection),rows=rows,controller_dim=381,rb_dim=156,features='predicted ground pose12,velocity9,GRU64,elapsed clock,initial537execution history',target='estimated controller381+RB156history, not true future runtime state',normalization='TRAIN only',updates=a.updates,seeds=[0,1,2],batch=64,lr=3e-4,clip=1,selection='DEV entry-point normalized history RMSE; full sequence RMSE also retained',control_policy_updates=0,scope='Canonical downstream0 per approach, whole-map split from existing collection; no TEST claim')
(a.output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n');(a.output/'ground_errors.json').write_text(json.dumps(pose_errors,indent=2)+'\n')
summary=[]
for seed in range(3):
 torch.manual_seed(seed);out=a.output/f'seed{seed}';out.mkdir();writer=SummaryWriter(str(out/'tensorboard'));bridge=SkillHistoryBridge(mean,scale,tmean,tscale).to(device);opt=torch.optim.Adam(bridge.parameters(),lr=3e-4);best=float('inf');logs=[]
 for update in range(a.updates+1):
  if update%25==0 or update==a.updates:
   with torch.no_grad():
    prediction=bridge(features[dev_end]);error=(prediction-target[dev_end])/tscale;score=float(error.square().mean().sqrt().cpu());all_error=((bridge(features[dev])-target[dev])/tscale).square().mean().sqrt();metrics=dict(update=update,entry_rmse=score,controller_rmse=float(error[:,:381].square().mean().sqrt().cpu()),rb_rmse=float(error[:,381:].square().mean().sqrt().cpu()),sequence_rmse=float(all_error.cpu()))
   checkpoint=dict(model=bridge.state_dict(),optimizer=opt.state_dict(),update=update,seed=seed);torch.save(checkpoint,out/f'model_{update:04d}.pt');logs.append(metrics);writer.add_scalar('DEV/entry_history_rmse',score,update);writer.add_scalar('DEV/sequence_history_rmse',metrics['sequence_rmse'],update)
   if score<best:
    best=score;torch.save(checkpoint,out/'best.pt');(out/'best.json').write_text(json.dumps(metrics,indent=2)+'\n');np.savez_compressed(out/'dev_entry_history.npz',predicted=prediction.cpu().numpy(),actual=target[dev_end].cpu().numpy(),scale=tscale.cpu().numpy(),cases=np.array([r['case'] for r in rows if r['split']=='DEV']))
   (out/'log.json').write_text(json.dumps(logs,indent=2)+'\n');writer.flush()
  if update==a.updates:break
  ids=train[torch.randint(len(train),(64,),device=device)];prediction=bridge(features[ids]);loss=((prediction-target[ids])/tscale).square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.);opt.step()
  if update%25==0:writer.add_scalar('TRAIN/history_mse',float(loss.detach().cpu()),update+1)
 torch.save(checkpoint,out/'last.pt');writer.close();summary.append(dict(seed=seed,**json.loads((out/'best.json').read_text())));print(json.dumps(summary[-1]),flush=True)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'status.json').write_text(json.dumps(dict(status='complete',stage='P4 ground-history bridge pilot',wall_s=time.perf_counter()-started))+'\n')
