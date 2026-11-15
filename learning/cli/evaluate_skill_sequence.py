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
from sttw_control.skill_handoff_history import HandoffHistoryReset
from sttw_control.skill_sequence_rollout import predict_skill_sequence,native_to_ground,HandoffEventGuard
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--handoff-event',choices=['classifier','state'],default='state');p.add_argument('--handoff-history',choices=['learned','execution'],default='execution');p.add_argument('--typical',type=Path);p.add_argument('--diagnostic-entry',choices=['physical','history','both']);p.add_argument('--diagnostic-departure',choices=['physical','history','both']);source=p.add_mutually_exclusive_group(required=True);source.add_argument('--oracle',type=Path);source.add_argument('--evaluation',type=Path);p.add_argument('--selection',type=Path,required=True);p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();
if sum(bool(x) for x in [a.typical,a.diagnostic_entry,a.diagnostic_departure])>1:p.error('Typical, true-entry and true-departure are separate modes')
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
selection=json.loads(a.selection.read_text())['models']
base=a.oracle/'map48' if a.oracle else a.evaluation
physical=base/'base' if a.oracle else base
cfg=json.loads((physical/'resolved_config.json').read_text());schema=json.loads((physical/'model_state_schema.json').read_text())
world=json.loads((base/'world.json' if a.oracle else Path(cfg['world_map'])).read_text());bank=np.load(cfg['ground_handoff']['reference_path'])
approach=load_history_model(Path(selection['approach']['path']).parent);departure=load_history_model(Path(selection['departure']['path']).parent)
for model in [approach,departure]:
 if model.terrain is not None:model.terrain=TerrainFeatures(**raster_from_spec(world['terrain']),device='cuda')
saved=torch.load(selection['jump']['path'],map_location='cuda',weights_only=True)['model'];jump=SkillWorldModel.from_weights(saved);jump.eval()
h=torch.load(a.history,map_location='cuda',weights_only=True)['model'];bridge=SkillHistoryBridge(*[h[k] for k in ['mean','scale','target_mean','target_scale']]).cuda();bridge.load_state_dict(h);bridge.eval()
q=[];v=[];history=[];rows=[]
if a.evaluation:
 for actual in json.loads((a.evaluation/'status.json').read_text())['results']:
  source=a.evaluation/actual['case_dir']/'trajectory.npz';z=np.load(source);route=actual['route'];speed=float(actual['variant']['requested_speed_m_s'])
  q.append(z['qpos'][0]);v.append(z['qvel'][0]);history.append(np.r_[z['controller_state'][0],z['rb_state'][0]])
  rows.append(dict(speed=speed,upstream=route//3,root=actual['variant']['root_variant'],downstream=route%3,route=route,actual_success=actual['goal'],actual_termination=actual['termination'],prefix=str(source),suffix=None))
else:
 for up in (range(1,3) if a.diagnostic_entry or a.diagnostic_departure else range(3)):
  for lane,root in [(0,1001),(2,1002)]:
   source=base/f'approach{up}/evaluation';name=f'rb_run3_route0_h75_v{lane}';z=np.load(source/name/'trajectory.npz')
   for down in range(3):
    q.append(z['qpos'][0]);v.append(z['qvel'][0]);history.append(np.r_[z['controller_state'][0],z['rb_state'][0]])
    label_source=source if up==0 else base/f'departure{up}_{down}/evaluation';actual=next(r for r in json.loads((label_source/'status.json').read_text())['results'] if r['case_dir']==name)
    rows.append(dict(speed=1.8,upstream=up,root=root,downstream=down,route=3*up+down,actual_success=actual['goal'],actual_termination=actual['termination'],prefix=str(source/name/'trajectory.npz'),suffix=None if up==0 else str(label_source/name/'trajectory.npz')))
route=[r['route'] for r in rows];n=len(rows);speeds=tensor([r['speed'] for r in rows])
typical=None
if a.typical:
 frozen=np.load(a.typical);typical={}
 for stage,phase in [('entry',1),('departure',2)]:
  for name in ['qpos','qvel','previous_qpos','previous_qvel','history']:
   typical[stage+'_'+name]=tensor([frozen[f'up{r["upstream"]}_v{r["speed"]:.3f}_phase{phase}_{name}'] for r in rows])
diagnostic=None
if a.diagnostic_entry:
 data=[]
 for row in rows:
  actual=np.load(row['prefix']);idx=int(np.flatnonzero(actual['trace'][:,-1]==1)[0]);data.append((actual,idx))
 diagnostic={}
 if a.diagnostic_entry in ['physical','both']:
  for key in ['qpos','qvel']:diagnostic[key]=tensor([z[key][idx] for z,idx in data])
 if a.diagnostic_entry in ['history','both']:diagnostic['history']=tensor([np.r_[z['controller_state'][idx],z['rb_state'][idx]] for z,idx in data])
departure_diagnostic=None
if a.diagnostic_departure:
 data=[]
 for row in rows:
  actual=np.load(row['prefix']);idx=int(np.flatnonzero(actual['trace'][:,-1]==2)[0]);data.append((actual,idx))
 departure_diagnostic={}
 if a.diagnostic_departure in ['physical','both']:
  for key in ['qpos','qvel']:
   departure_diagnostic[key]=tensor([z[key][idx] for z,idx in data])
   departure_diagnostic['previous_'+key]=tensor([z[key][max(idx-4,0)] for z,idx in data])
 if a.diagnostic_departure in ['history','both']:departure_diagnostic['history']=tensor([z['controller_state'][idx] for z,idx in data])
handoff_reset=None
if a.handoff_history=='execution':
 if not cfg['ground_handoff'].get('sttw_only_after_lock'):p.error('Execution-history reset requires reset-at-return frozen contract')
 handoff_reset=HandoffHistoryReset(cfg,json.loads((physical/'controller_schema.json').read_text()),schema,bank,route,'cuda')
physics=mujoco.MjModel.from_binary_path(str(physical/'terrain.mjb'));guard=HandoffEventGuard(physics,schema,cfg['ground_handoff'],world,speeds)
with torch.no_grad():
 result=predict_skill_sequence(approach,bridge,jump,departure,tensor(q),tensor(v),tensor(history),schema,tensor(bank['s']),tensor(bank['xy'][route]),tensor(bank['heading'][route]),tensor(bank['curvature'][route]),speeds,cfg['ground_handoff']['approach']['jump_start_x_m'],tensor([[world['terrain']['front_x_m'],world['terrain']['width_m'],world['terrain'].get('far_bank_height_m',0.),r['speed'],0.] for r in rows]),typical=typical,diagnostic_entry=diagnostic,handoff_reset=handoff_reset,handoff_guard=guard if a.handoff_event=='state' else None,diagnostic_departure=departure_diagnostic)
 tail=result['departure'];pose=tail['pose'];vel=tail['velocity'];domain=tail['domain'];goal=world['goal'];wv=rotate(pose[:,:,3:7].reshape(-1,4),vel[:,:,:3].reshape(-1,3))[:,0].reshape(n,-1)
 hold=(pose[:,:,0]>=goal['x_min_m'])&(pose[:,:,0]<=goal['x_max_m'])&(pose[:,:,1].abs()<=goal['abs_y_max_m'])&(wv>=goal['min_vx_m_s'])&(domain[:,:,0].abs()<=goal['path_tolerance_m'])&(domain[:,:,1].abs()<=goal['heading_tolerance_rad'])&((vel[:,:,0]-speeds[:,None]).abs()<=goal['speed_tolerance_m_s'])
 valid_hold=hold.unfold(1,26,1).all(-1);has_goal=valid_hold.any(-1);stop=torch.where(has_goal,valid_hold.long().argmax(-1)+25,torch.full((n,),250,device='cuda'));goal_time=result['handoff_time_s']+stop*.02
 rb_pose,_=native_to_ground(result['jump']['qpos'].reshape(-1,12),result['jump']['qvel'].reshape(-1,11),schema);rb_pose=rb_pose.reshape(n,125,12)
 all_pose=torch.cat((result['approach_pose'],rb_pose,pose),1);entry_steps=(result['entry_time_s']/.02).round().long();valid=torch.cat((torch.arange(251,device='cuda')[None,:]<=entry_steps[:,None],result['jump']['valid'],torch.arange(251,device='cuda')[None,:]<=stop[:,None]),1)
 lo,hi=guard.geometry(all_pose)
 rect=world['terrain'].get('forbidden_rectangles',[]);clear=all_pose.new_full(all_pose.shape[:2],float('inf'))
 if rect:
  boxes=tensor([[r[0],r[2],0.,r[1],r[3],world['terrain'].get('obstacle_height_m',.5)] for r in rect]);clear=torch.maximum(boxes[None,None,None,:,:3]-hi[:,:,:,None,:],lo[:,:,:,None,:]-boxes[None,None,None,:,3:]).amax(-1).amin((-1,-2))
 bounds=world['bounds_xy_m'];road=torch.stack((lo[:,:,:,0]-bounds[0],bounds[1]-hi[:,:,:,0],lo[:,:,:,1]-bounds[2],bounds[3]-hi[:,:,:,1]),-1).amin((-1,-2));clear=torch.minimum(clear,road).masked_fill(~valid,float('inf')).amin(-1)
 angles=euler(all_pose[:,:,3:7].reshape(-1,4)).reshape(n,-1,3);posture=((angles[:,:,0].abs()<=.7)&(angles[:,:,1].abs()<=1.309)|~valid).all(-1)
 tail_domain=(domain[:,:,0].abs()<=cfg['ground_handoff']['road_half_width_m'])&(domain[:,:,1].abs()<np.pi/2)&(domain[:,:,2]>.05)&(vel[:,:,0]>=.5)
 domain_ok=(tail_domain|(torch.arange(251,device='cuda')[None,:]>stop[:,None])).all(-1)
 ad=result['approach_domain'];av=result['approach_velocity'];approach_valid=(ad[:,:,0].abs()<=cfg['ground_handoff']['road_half_width_m'])&(ad[:,:,1].abs()<np.pi/2)&(ad[:,:,2]>.05)&(av[:,:,0]>=.5)
 approach_ok=(approach_valid|(torch.arange(251,device='cuda')[None,:]>entry_steps[:,None])).all(-1)
 if a.handoff_event=='classifier':guard(dict(qpos=result['jump']['qpos'],qvel=result['jump']['qvel'],contact=result['jump']['contact_logits'].sigmoid()))
 lane=torch.arange(n,device='cuda');event_index=result['jump']['event_step']-1
 diagnostics={k:v[lane,event_index] for k,v in guard.diagnostics.items()}
 lock_ready,lock_position_error,lock_velocity,support_ready,body_ready,support_geometry=diagnostics.values()
 admitted=support_geometry&support_ready&body_ready&lock_ready&approach_ok&result['supported']&has_goal&(goal_time<=10)&(clear>=0)&posture&domain_ok
 metrics=torch.stack((admitted.float(),has_goal.float(),clear,result['entry_time_s'],result['handoff_time_s'],goal_time,posture.float(),domain_ok.float(),result['supported'].float(),approach_ok.float(),lock_ready.float(),lock_position_error,lock_velocity,support_ready.float(),body_ready.float(),support_geometry.float()),-1).cpu().numpy()
 np.savez_compressed(a.output/'predictions.npz',approach=result['approach_pose'].cpu().numpy(),jump=rb_pose.cpu().numpy(),departure=pose.cpu().numpy(),entry_steps=entry_steps.cpu().numpy(),jump_steps=result['jump']['event_step'].cpu().numpy(),departure_steps=stop.cpu().numpy(),entry_history=result['entry_history'].cpu().numpy(),entry_qpos=result['entry_qpos'].cpu().numpy(),entry_qvel=result['entry_qvel'].cpu().numpy(),handoff_qpos=result['jump']['next_state'].qpos.cpu().numpy(),handoff_contact=result['jump']['next_state'].contact.cpu().numpy(),handoff_phase=result['jump']['next_state'].phase.cpu().numpy(),handoff_qvel=result['jump']['next_state'].qvel.cpu().numpy(),handoff_history=result['jump']['next_state'].history.cpu().numpy(),departure_history=result['departure_history'].cpu().numpy(),departure_previous_velocity=result['departure_previous_velocity'].cpu().numpy(),departure_velocity=vel.cpu().numpy())
for row,metric in zip(rows,metrics):
 row.update(dict(zip(['admitted','has_goal','clearance_m','entry_time_s','handoff_time_s','goal_time_s','posture_valid','departure_domain_valid','events_found','approach_domain_valid','handoff_joints_ready','handoff_joint_error_rad','handoff_joint_speed_rad_s','handoff_support_ready','handoff_body_ready','handoff_support_geometry'],map(float,metric))))
(a.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=dict(cases=n,true_admit=sum(bool(r['admitted']) and r['actual_success'] for r in rows),false_admit=sum(bool(r['admitted']) and not r['actual_success'] for r in rows),false_reject=sum(not r['admitted'] and r['actual_success'] for r in rows),true_reject=sum(not r['admitted'] and not r['actual_success'] for r in rows),scope=('OFFLINE true-departure '+a.diagnostic_departure+' intervention; predicted prefix/event/clock unchanged; NOT causal planning' if a.diagnostic_departure else 'OFFLINE true-entry '+a.diagnostic_entry+' intervention; NOT causal planning' if a.diagnostic_entry else 'I independent TRAIN-typical interfaces' if a.typical else 'C continuous predicted interfaces')+' raw-envelope DEV diagnostic; single-method query; no new margin calibration; full approach/departure tracking-domain and all-stage posture/obstacle gates included',runtime_s=time.perf_counter()-started)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'contract.json').write_text(json.dumps(dict(handoff_event=a.handoff_event,method='diagnostic_departure_'+a.diagnostic_departure if a.diagnostic_departure else 'diagnostic_'+a.diagnostic_entry if a.diagnostic_entry else 'I' if a.typical else 'C',diagnostic_entry=a.diagnostic_entry,diagnostic_departure=a.diagnostic_departure,handoff_history=a.handoff_history,jump_start_x_m=cfg['ground_handoff']['approach']['jump_start_x_m'],typical=str(a.typical) if a.typical else None,selection=str(a.selection),history=str(a.history),oracle=str(a.oracle) if a.oracle else None,evaluation=str(a.evaluation) if a.evaluation else None,true_data_access='OFFLINE intervention reads actual future first phase2 state/history; previous physical velocity is20ms earlier; predicted event and clock unchanged, not deployment' if a.diagnostic_departure else 'OFFLINE intervention reads actual future RB entry; predicted entry clock unchanged; never a deployment result' if a.diagnostic_entry else 'Only first qpos/qvel/controller/RB history used by predictor; future trajectories/status used only as offline labels',limits='Predicted phase2 additionally requires frozen joint lock tolerances; predicted far-wheel contact uses probability>=0.5 and body checks mirror frozen native tolerances with qvel roll-rate proxy. Wheel AABBs must reach the far bank to make predicted far contact geometrically possible. These checks do not certify physical contact or full phase-transition history. Ground locked hip/knee rates assumed zero, flat supported RB entry contract; estimates never modify real controls'),indent=2)+'\n');print(json.dumps(summary))
