"""Audit predicted event-selected handoff state against recorded DEV handoff."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
import torch
from torch.nn import functional as F
from sttw_control.skill_world_model import SkillWorldModel
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);device='cuda';tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
m=json.loads((a.data/'manifest.json').read_text());z=np.load(a.data/'sequences.npz');n=np.load(a.data/'normalization.npz');rows=[r for r in m['rows'] if r['split']=='DEV'];starts=np.array([r['start'] for r in rows]);saved=torch.load(a.model,map_location=device,weights_only=True);model=SkillWorldModel.from_weights(saved['model']);model.eval()
with torch.no_grad():
 history=tensor(z['history'][starts]);context=tensor(z['context'][starts]);state=model.initialize(tensor(z['qpos'][starts]),tensor(z['qvel'][starts]),history[:,:381],history[:,381:],tensor(z['contact'][starts]),F.one_hot(torch.as_tensor(z['phase'][starts],device=device,dtype=torch.long),5).float(),context)
 pred=model.rollout_until(state,context,125)
 final={k:getattr(pred['next_state'],k).cpu().numpy() for k in ['qpos','qvel','history','hidden','contact','phase']};duration=pred['duration_s'].cpu().numpy();found=pred['event_found'].cpu().numpy();trace=pred['qpos'].cpu().numpy();contact_trace=pred['contact_logits'].sigmoid().cpu().numpy();steps=pred['event_step'].cpu().numpy()
records=[];models={}
def boxes(physics,q):
 d=mujoco.MjData(physics);d.qpos[:]=q;mujoco.mj_kinematics(physics,d);out=[]
 for g in range(physics.ngeom):
  if physics.geom_bodyid[g]>0 and (physics.geom_contype[g] or physics.geom_conaffinity[g]):
   rot=d.geom_xmat[g].reshape(3,3);center=d.geom_xpos[g]+rot@physics.geom_aabb[g,:3];extent=abs(rot)@physics.geom_aabb[g,3:];out.append(np.stack((center-extent,center+extent)))
 return np.stack(out)
for lane,row in enumerate(rows):
 indices=np.arange(row['start'],row['start']+row['length']);times=z['time'][indices]
 if row['handoff_s'] is None:
  records.append(dict(map=row['map'],case_dir=row['case_dir'],actual_handoff=False,predicted_handoff=bool(found[lane])));continue
 index=indices[np.argmin(abs(times-row['handoff_s']))];truth=z['qpos'][index];prediction=final['qpos'][lane]
 if row['terrain_model'] not in models:models[row['terrain_model']]=mujoco.MjModel.from_binary_path(row['terrain_model'])
 physics=models[row['terrain_model']];occupancy=np.linalg.norm(boxes(physics,prediction)-boxes(physics,truth),axis=-1).max()
 history_error=(final['history'][lane]-z['history'][index])/n['history_state_scale']
 observed=z['contact'][indices];takeoff_actual=np.flatnonzero(observed[:,0]<.5);takeoff_pred=np.flatnonzero(contact_trace[lane,:steps[lane],0]<.5);far_pred=np.flatnonzero(contact_trace[lane,:steps[lane],1]>=.5)
 event=dict(takeoff_actual_s=float(times[takeoff_actual[0]]) if len(takeoff_actual) else None,takeoff_predicted_s=float(times[0]+.02*(takeoff_pred[0]+1)) if len(takeoff_pred) else None,far_contact_actual_s=row['first_far_contact_s'],far_contact_predicted_s=float(times[0]+.02*(far_pred[0]+1)) if len(far_pred) else None)
 records.append(dict(**event,map=row['map'],case_dir=row['case_dir'],actual_handoff=True,predicted_handoff=bool(found[lane]),event_error_s=float(z['time'][row['start']]+duration[lane]-row['handoff_s']) if found[lane] else None,position_error_m=float(np.linalg.norm(prediction[:3]-truth[:3])),rotation_error_rad=float(2*np.arccos(np.clip(abs(prediction[3:7]@truth[3:7]),0,1))),joint_abs_error_rad=np.abs(prediction[7:]-truth[7:]).tolist(),controller_normalized_rmse=float(np.sqrt(np.mean(history_error[:381]**2))),rb_history_normalized_rmse=float(np.sqrt(np.mean(history_error[381:]**2))),collision_geom_aabb_corner_max_error_m=float(occupancy),actual_source_index=int(index)))
np.savez_compressed(a.output/'predicted_handoff.npz',**final,duration_s=duration,event_found=found,case_ids=np.array([r['map']+'/'+r['case_dir'] for r in rows]))
(a.output/'cases.json').write_text(json.dumps(records,indent=2)+'\n')
success=[r for r in records if r['actual_handoff'] and r['predicted_handoff']]
summary=dict(cases=len(records),predicted_handoffs=len(success),event_mae_s=float(np.mean([abs(r['event_error_s']) for r in success])),event_max_abs_s=max(abs(r['event_error_s']) for r in success),handoff_position_mean_m=float(np.mean([r['position_error_m'] for r in success])),handoff_position_max_m=max(r['position_error_m'] for r in success),occupancy_mean_m=float(np.mean([r['collision_geom_aabb_corner_max_error_m'] for r in success])),scope='125-step max query; GPU predicted phase selects handoff, never actual future event time. Compare event-selected state to actual handoff, includes timing error. AABB corner metric is conservative geometric discrepancy, not safety proof.',model=str(a.model))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
