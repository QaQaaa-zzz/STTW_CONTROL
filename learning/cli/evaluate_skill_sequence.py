"""P4 first causal full-chain query from common real initial states."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec
from sttw_control.playground_prediction import BodyEnvelope,euler,rotate
from sttw_control.skill_world_model import SkillWorldModel
from sttw_control.skill_history_bridge import SkillHistoryBridge
from sttw_control.skill_sequence_rollout import predict_skill_sequence,native_to_ground
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
selection=json.loads(a.selection.read_text())['models'];base=a.oracle/'map48';cfg=json.loads((base/'base/resolved_config.json').read_text());schema=json.loads((base/'base/model_state_schema.json').read_text());world=json.loads((base/'world.json').read_text());bank=np.load('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/rb_adaptation/approach_departure_turns/routes.npz')
approach=load_history_model(Path(selection['approach']['path']).parent);departure=load_history_model(Path(selection['departure']['path']).parent)
for model in [approach,departure]:
 if model.terrain is not None:model.terrain=TerrainFeatures(**raster_from_spec(world['terrain']),device='cuda')
saved=torch.load(selection['jump']['path'],map_location='cuda',weights_only=True)['model'];jump=SkillWorldModel(537,5,*[saved[k] for k in ['input_mean','input_scale','velocity_scale','history_scale']]).cuda();jump.load_state_dict(saved);jump.eval()
h=torch.load(a.history,map_location='cuda',weights_only=True)['model'];bridge=SkillHistoryBridge(*[h[k] for k in ['mean','scale','target_mean','target_scale']]).cuda();bridge.load_state_dict(h);bridge.eval()
q=[];v=[];history=[];rows=[]
for up in range(3):
 for lane,root in [(0,1001),(2,1002)]:
  source=base/f'approach{up}/evaluation';name=f'rb_run3_route0_h75_v{lane}';z=np.load(source/name/'trajectory.npz')
  for down in range(3):
   q.append(z['qpos'][0]);v.append(z['qvel'][0]);history.append(np.r_[z['controller_state'][0],z['rb_state'][0]])
   label_source=source if up==0 else base/f'departure{up}_{down}/evaluation';actual=next(r for r in json.loads((label_source/'status.json').read_text())['results'] if r['case_dir']==name)
   rows.append(dict(upstream=up,root=root,downstream=down,route=3*up+down,actual_success=actual['goal'],actual_termination=actual['termination'],prefix=str(source/name/'trajectory.npz'),suffix=None if up==0 else str(label_source/name/'trajectory.npz')))
route=[r['route'] for r in rows];n=len(rows)
with torch.no_grad():
 result=predict_skill_sequence(approach,bridge,jump,departure,tensor(q),tensor(v),tensor(history),schema,tensor(bank['s']),tensor(bank['xy'][route]),tensor(bank['heading'][route]),tensor(bank['curvature'][route]),tensor([1.8]*n),8.6,tensor([[10.3,.3,0.,1.8,0.]]*n))
 tail=result['departure'];pose=tail['pose'];vel=tail['velocity'];domain=tail['domain'];goal=world['goal'];wv=rotate(pose[:,:,3:7].reshape(-1,4),vel[:,:,:3].reshape(-1,3))[:,0].reshape(n,-1)
 hold=(pose[:,:,0]>=goal['x_min_m'])&(pose[:,:,0]<=goal['x_max_m'])&(pose[:,:,1].abs()<=goal['abs_y_max_m'])&(wv>=goal['min_vx_m_s'])&(domain[:,:,0].abs()<=goal['path_tolerance_m'])&(domain[:,:,1].abs()<=goal['heading_tolerance_rad'])&((vel[:,:,0]-1.8).abs()<=goal['speed_tolerance_m_s'])
 valid_hold=hold.unfold(1,26,1).all(-1);has_goal=valid_hold.any(-1);stop=torch.where(has_goal,valid_hold.long().argmax(-1)+25,torch.full((n,),250,device='cuda'));goal_time=result['handoff_time_s']+stop*.02
 rb_pose,_=native_to_ground(result['jump']['qpos'].reshape(-1,12),result['jump']['qvel'].reshape(-1,11),schema);rb_pose=rb_pose.reshape(n,125,12)
 all_pose=torch.cat((result['approach_pose'],rb_pose,pose),1);entry_steps=(result['entry_time_s']/.02).round().long();valid=torch.cat((torch.arange(251,device='cuda')[None,:]<=entry_steps[:,None],result['jump']['valid'],torch.arange(251,device='cuda')[None,:]<=stop[:,None]),1)
 physics=mujoco.MjModel.from_binary_path(str(base/'base/terrain.mjb'));lo,hi=BodyEnvelope(physics,dimensions=3)(all_pose);rect=world['terrain']['forbidden_rectangles'];boxes=tensor([[r[0],r[2],0.,r[1],r[3],world['terrain'].get('obstacle_height_m',.5)] for r in rect]);clear=torch.maximum(boxes[None,None,None,:,:3]-hi[:,:,:,None,:],lo[:,:,:,None,:]-boxes[None,None,None,:,3:]).amax(-1).amin((-1,-2));bounds=world['bounds_xy_m'];road=torch.stack((lo[:,:,:,0]-bounds[0],bounds[1]-hi[:,:,:,0],lo[:,:,:,1]-bounds[2],bounds[3]-hi[:,:,:,1]),-1).amin((-1,-2));clear=torch.minimum(clear,road).masked_fill(~valid,float('inf')).amin(-1)
 angles=euler(all_pose[:,:,3:7].reshape(-1,4)).reshape(n,-1,3);posture=((angles[:,:,0].abs()<=.7)&(angles[:,:,1].abs()<=1.309)|~valid).all(-1)
 tail_domain=(domain[:,:,0].abs()<=cfg['ground_handoff']['road_half_width_m'])&(domain[:,:,1].abs()<np.pi/2)&(domain[:,:,2]>.05)&(vel[:,:,0]>=.5)
 domain_ok=(tail_domain|(torch.arange(251,device='cuda')[None,:]>stop[:,None])).all(-1)
 ad=result['approach_domain'];av=result['approach_velocity'];approach_valid=(ad[:,:,0].abs()<=cfg['ground_handoff']['road_half_width_m'])&(ad[:,:,1].abs()<np.pi/2)&(ad[:,:,2]>.05)&(av[:,:,0]>=.5)
 approach_ok=(approach_valid|(torch.arange(251,device='cuda')[None,:]>entry_steps[:,None])).all(-1)
 admitted=approach_ok&result['supported']&has_goal&(goal_time<=10)&(clear>=0)&posture&domain_ok
 metrics=torch.stack((admitted.float(),has_goal.float(),clear,result['entry_time_s'],result['handoff_time_s'],goal_time,posture.float(),domain_ok.float(),result['supported'].float(),approach_ok.float()),-1).cpu().numpy()
 np.savez_compressed(a.output/'predictions.npz',approach=result['approach_pose'].cpu().numpy(),jump=rb_pose.cpu().numpy(),departure=pose.cpu().numpy(),entry_steps=entry_steps.cpu().numpy(),jump_steps=result['jump']['event_step'].cpu().numpy(),departure_steps=stop.cpu().numpy(),entry_history=result['entry_history'].cpu().numpy())
for row,metric in zip(rows,metrics):
 row.update(dict(zip(['admitted','has_goal','clearance_m','entry_time_s','handoff_time_s','goal_time_s','posture_valid','departure_domain_valid','events_found','approach_domain_valid'],map(float,metric))))
(a.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=dict(cases=n,true_admit=sum(bool(r['admitted']) and r['actual_success'] for r in rows),false_admit=sum(bool(r['admitted']) and not r['actual_success'] for r in rows),false_reject=sum(not r['admitted'] and r['actual_success'] for r in rows),true_reject=sum(not r['admitted'] and not r['actual_success'] for r in rows),scope='First C-only raw-envelope DEV diagnostic; no I/G/strong-backtracking comparison or new margin calibration; full approach/departure tracking-domain and all-stage posture/obstacle gates included',runtime_s=time.perf_counter()-started)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'contract.json').write_text(json.dumps(dict(selection=str(a.selection),history=str(a.history),oracle=str(a.oracle),true_data_access='Only first qpos/qvel/controller/RB history used by predictor; future trajectories/status used only as offline labels',limits='Ground locked hip/knee rates assumed zero, flat supported RB entry contract; estimates never modify real controls'),indent=2)+'\n');print(json.dumps(summary))
