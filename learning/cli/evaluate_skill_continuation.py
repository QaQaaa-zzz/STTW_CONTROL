"""P3 diagnostic: identical ground predictor from measured vs predicted RB handoff."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch,mujoco
from torch.nn import functional as F
from sttw_control.skill_world_model import SkillWorldModel
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_prediction import BodyEnvelope,rotate,euler
from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interchange',action='store_true');p.add_argument('--oracle',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--ground',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);t0=time.perf_counter();tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
base=a.oracle/'map48';cfg=json.loads((base/'base/resolved_config.json').read_text());world=json.loads((base/'world.json').read_text());schema=json.loads((base/'base/model_state_schema.json').read_text());bank=np.load('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation/approach_departure_turns/routes.npz');normal=np.load(a.data/'normalization.npz')
physics=mujoco.MjModel.from_binary_path(str(base/'base/terrain.mjb'));envelope=BodyEnvelope(physics,dimensions=3)
qorder=[schema['qpos_names'].index(n) for n in ['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint']]
vorder=[schema['steer_dof'],schema['front_dof'],schema['rear_dof']]
def ground_state(q,v):
 inv=q[:,3:7]*tensor([1,-1,-1,-1]);return torch.cat((q[:,:7],q[:,qorder]),-1),torch.cat((rotate(inv,v[:,:3]),v[:,3:6],v[:,vorder]),-1)
inputs=[];truth=[];case_rows=[]
for up in [1,2]:
 for lane,root in [(0,1001),(2,1002)]:
  prefix=base/f'approach{up}/evaluation/rb_run3_route0_h75_v{lane}/trajectory.npz';d=np.load(prefix);j=int(np.flatnonzero(d['trace'][:,-1]==1)[0]);suffix=base/f'departure{up}_0/evaluation/rb_run3_route0_h75_v{lane}/trajectory.npz';tail=np.load(suffix);k=int(np.argmin(abs(d['trace'][:,0]-tail['trace'][0,0])))
  inputs.append(dict(q=d['qpos'][j],v=d['qvel'][j],h=np.r_[d['controller_state'][j],d['rb_state'][j]],contact=d['wheel_contact_flags'][j],phase=d['trace'][j,-1],time=d['trace'][j,0]))
  truth.append(dict(q=tail['qpos'][0],v=tail['qvel'][0],h=tail['controller_state'][0],prev_q=d['qpos'][k-4],prev_v=d['qvel'][k-4],time=tail['trace'][0,0]))
  for down in range(3):
   path=base/f'departure{up}_{down}/evaluation';row=next(x for x in json.loads((path/'status.json').read_text())['results'] if x['case_dir']==suffix.parent.name)
   case_rows.append(dict(upstream=up,root=root,downstream=down,route=3*up+down,actual_success=row['goal'],actual_termination=row['termination'],actual_trajectory=str(path/row['case_dir']/'trajectory.npz')))
context=tensor([[10.3,.3,0.,1.8,0.]]*4);sources={}
q=tensor([r['q'] for r in truth]);v=tensor([r['v'] for r in truth]);pose,velocity=ground_state(q,v);_,previous=ground_state(tensor([r['prev_q'] for r in truth]),tensor([r['prev_v'] for r in truth]));sources['measured']=dict(pose=pose,velocity=velocity,previous=previous,history=tensor([r['h'] for r in truth]),time=tensor([r['time'] for r in truth]),found=torch.ones(4,device='cuda',dtype=torch.bool))
with torch.no_grad():
 for seed in range(3):
  model=SkillWorldModel(537,5,*[normal[k] for k in ['input_mean','input_scale','velocity_scale','history_scale']]).cuda();checkpoint=a.models/f'seed{seed}/best.pt';model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['model']);model.eval()
  history=tensor([r['h'] for r in inputs]);q=tensor([r['q'] for r in inputs]);v=tensor([r['v'] for r in inputs]);state=model.initialize(q,v,history[:,:381],history[:,381:],tensor([r['contact'] for r in inputs]),F.one_hot(torch.ones(4,device='cuda',dtype=torch.long),5).float(),context)
  pred=model.rollout_until(state,context,125);pose,velocity=ground_state(pred['next_state'].qpos,pred['next_state'].qvel);last=pred['event_step']-1;lanes=torch.arange(4,device='cuda');pq=torch.where((last>0)[:,None],pred['qpos'][lanes,(last-1).clamp_min(0)],q);pv=torch.where((last>0)[:,None],pred['qvel'][lanes,(last-1).clamp_min(0)],v);_,previous=ground_state(pq,pv)
  sources[f'seed{seed}']=dict(pose=pose,velocity=velocity,previous=previous,history=pred['next_state'].history[:,:381],time=tensor([r['time'] for r in inputs])+pred['duration_s'],found=pred['event_found'])
 if a.interchange:
  for seed in range(3):
   predicted=sources[f'seed{seed}'];measured=sources['measured']
   sources[f'seed{seed}_state_only']={**predicted,'history':measured['history']}
   sources[f'seed{seed}_history_only']={**measured,'history':predicted['history']}
 ground=load_history_model(a.ground);ground.terrain=TerrainFeatures(**raster_from_spec(world['terrain']),device='cuda') if ground.terrain is not None else None
 rows=json.loads(a.calibration.read_text());cal=next(r for r in rows if r['model']=='M1');edges=tensor(cal['edges']);margins=tensor(cal['axis_margin_m']);rects=world['terrain']['forbidden_rectangles'];boxes=tensor([[r[0],r[2],0.,r[1],r[3],world['terrain'].get('obstacle_height_m',.5)] for r in rects]);bounds=world['bounds_xy_m'];goal=world['goal'];results=[];saved={}
 for name,source in sources.items():
  fields={k:v.repeat_interleave(3,0) for k,v in source.items()};routes=[r['route'] for r in case_rows]
  pred,vel,domain=ground.rollout(fields['pose'],fields['velocity'],fields['previous'],fields['history'],tensor(bank['s']),tensor(bank['xy'][routes]),tensor(bank['heading'][routes]),tensor(bank['curvature'][routes]),250,speed_request=tensor([1.8]*12),return_domain=True)
  lo,hi=envelope(pred);angles=euler(pred[:,:,3:7].reshape(-1,4)).reshape(12,251,3);worldvx=rotate(pred[:,:,3:7].reshape(-1,4),vel[:,:,:3].reshape(-1,3))[:,0].reshape(12,251)
  hold=(pred[:,:,0]>=goal['x_min_m'])&(pred[:,:,0]<=goal['x_max_m'])&(pred[:,:,1].abs()<=goal['abs_y_max_m'])&(worldvx>=goal['min_vx_m_s'])&(domain[:,:,0].abs()<=goal['path_tolerance_m'])&(domain[:,:,1].abs()<=goal['heading_tolerance_rad'])&((vel[:,:,0]-1.8).abs()<=goal['speed_tolerance_m_s'])
  runs=hold.unfold(1,26,1).all(-1);has=runs.any(-1);stop=torch.where(has,runs.long().argmax(-1)+25,torch.full((12,),250,device='cuda'));valid=torch.arange(251,device='cuda')[None,:]<=stop[:,None]
  dynamic=(angles[:,:,0].abs()<=.7)&(angles[:,:,1].abs()<=1.309)&(domain[:,:,0].abs()<=cfg['ground_handoff']['road_half_width_m'])&(domain[:,:,1].abs()<np.pi/2)&(domain[:,:,2]>.05)&(vel[:,:,0]>=.5)
  for margin_name in ['raw','frozen_M1']:
   region=torch.bucketize(pred[:,:,0].contiguous(),edges,right=True).sub(1).clamp(0,len(margins)-1);expansion=margins[region][:,:,None,:] if margin_name=='frozen_M1' else 0
   low=lo-expansion;high=hi+expansion;gap=torch.maximum(boxes[None,None,None,:,:3]-high[:,:,:,None,:],low[:,:,:,None,:]-boxes[None,None,None,:,3:]).amax(-1).amin((-1,-2));road=torch.stack((low[:,:,:,0]-bounds[0],bounds[1]-high[:,:,:,0],low[:,:,:,1]-bounds[2],bounds[3]-high[:,:,:,1]),-1).amin((-1,-2));clear=torch.minimum(gap,road).masked_fill(~valid,float('inf')).amin(-1)
   okay=has&fields['found']&((fields['time']+stop*.02)<=10)&(dynamic|~valid).all(-1)&(clear>=0)
   for i,row in enumerate(case_rows):results.append(dict(row,source=name,margin=margin_name,admitted=bool(okay[i]),clearance_m=float(clear[i]),has_goal=bool(has[i]),dynamic_valid=bool((dynamic[i]|~valid[i]).all()),handoff_time_s=float(fields['time'][i]),goal_time_s=float(fields['time'][i]+stop[i]*.02) if has[i] else None))
  saved[name+'_initial_history']=fields['history'].cpu().numpy();saved[name+'_initial_pose']=fields['pose'].cpu().numpy();saved[name+'_pose']=pred.cpu().numpy();saved[name+'_time']=fields['time'].cpu().numpy();saved[name+'_stop']=stop.cpu().numpy()
np.savez_compressed(a.output/'predictions.npz',**saved);(a.output/'cases.json').write_text(json.dumps(results,indent=2)+'\n');summary=[]
for source in sources:
 for margin in ['raw','frozen_M1']:
  sub=[r for r in results if r['source']==source and r['margin']==margin];summary.append(dict(source=source,margin=margin,cases=len(sub),true_admit=sum(r['admitted'] and r['actual_success'] for r in sub),false_admit=sum(r['admitted'] and not r['actual_success'] for r in sub),false_reject=sum(not r['admitted'] and r['actual_success'] for r in sub),true_reject=sum(not r['admitted'] and not r['actual_success'] for r in sub)))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'contract.json').write_text(json.dumps(dict(role='P3 known DEV map48 handoff interchange diagnosis, not full-chain or TEST',oracle=str(a.oracle),ground=str(a.ground),rb_models=str(a.models),calibration=str(a.calibration),rows=12,upstream=[1,2],roots=[1001,1002],downstream=[0,1,2],notes='Measured source uses actual shared suffix snapshot (up to20ms after event); predicted source uses only real RB entry then predicted event/history. Same ground model, same original10s goal. No new physics. Raw and pre-frozen margins both retained; no post-hoc calibration.',runtime_s=time.perf_counter()-t0),indent=2)+'\n');print(json.dumps(summary))
