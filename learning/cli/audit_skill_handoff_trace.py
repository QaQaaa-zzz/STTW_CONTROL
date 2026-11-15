"""Replay frozen RB predictions from saved causal entries and audit lock timing."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.skill_world_model import SkillWorldModel
from sttw_control.skill_sequence_rollout import handoff_joint_errors
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--prediction',type=Path,action='append',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
rows=[];groups=[];started=time.perf_counter()
for group,path in enumerate(a.prediction):
 contract=json.loads((path/'contract.json').read_text());cases=json.loads((path/'cases.json').read_text());z=np.load(path/'predictions.npz')
 physical=Path(contract['oracle'])/'map48/base' if contract['oracle'] else Path(contract['evaluation'])
 cfg=json.loads((physical/'resolved_config.json').read_text());schema=json.loads((physical/'model_state_schema.json').read_text());hand=cfg['ground_handoff']
 world=json.loads(((physical.parent/'world.json') if contract['oracle'] else Path(cfg['world_map'])).read_text())
 checkpoint=json.loads(Path(contract['selection']).read_text())['models']['jump']['path']
 saved=torch.load(checkpoint,map_location='cuda',weights_only=True)['model']
 model=SkillWorldModel(537,5,*[saved[k] for k in ['input_mean','input_scale','velocity_scale','history_scale']]).cuda();model.load_state_dict(saved);model.eval()
 q,v,h=[tensor(z[k]) for k in ['entry_qpos','entry_qvel','entry_history']];n=len(q)
 context=tensor([[world['terrain']['front_x_m'],world['terrain']['width_m'],world['terrain'].get('far_bank_height_m',0.),c['speed'],0.] for c in cases])
 native=mujoco.MjModel.from_binary_path(str(physical/'terrain.mjb'));home=tensor(native.key('initial_state').qpos[[schema['qpos_names'].index(k) for k in ['hip_joint','knee_joint']]])
 with torch.no_grad():
  state=model.initialize(q,v,h[:,:381],h[:,381:],tensor([1,0,0]).expand(n,-1),tensor([0,1,0,0,0]).expand(n,-1),context)
  pred=model.rollout_until(state,context,125)
  error,rate=handoff_joint_errors(pred['qpos'],pred['qvel'],home,schema)
  ready=(error<hand['lock_position_tolerance_rad'])&(rate<hand['lock_velocity_tolerance_rad_s'])
  phase=pred['phase_logits'].argmax(-1);after=torch.arange(125,device='cuda')[None,:]>=pred['event_step'][:,None]-1
  phase_ready=ready&(phase==2);later_ready=ready&after
  ratio=torch.maximum(error/hand['lock_position_tolerance_rad'],rate/hand['lock_velocity_tolerance_rad_s'])
 data={k:pred[k].cpu().numpy() for k in ['qpos','qvel','phase_logits','contact_logits','event_step']}
 data.update(position_error_rad=error.cpu().numpy(),speed_rad_s=rate.cpu().numpy(),ready=ready.cpu().numpy(),phase_ready=phase_ready.cpu().numpy(),later_ready=later_ready.cpu().numpy(),ratio=ratio.cpu().numpy())
 np.savez_compressed(a.output/f'trace_{group:02}.npz',**data)
 terminal_error=float(np.max(abs(pred['next_state'].qpos.cpu().numpy()-z['handoff_qpos'])))
 phase_np=phase.cpu().numpy();first=pred['event_step'].cpu().numpy()
 for i,c in enumerate(cases):
  hit=np.flatnonzero(data['phase_ready'][i]);late=np.flatnonzero(data['later_ready'][i]);mask=phase_np[i]==2
  rows.append(dict(group=group,index=i,root=c['root'],route=c['route'],actual_success=c['actual_success'],classifier_step=int(first[i]),phase_and_joint_step=int(hit[0]+1) if len(hit) else None,later_joint_step=int(late[0]+1) if len(late) else None,min_phase2_tolerance_ratio=float(data['ratio'][i,mask].min()) if mask.any() else None))
 groups.append(dict(group=group,prediction=str(path),checkpoint=checkpoint,cases=n,phase_and_joint_ready=int(data['phase_ready'].any(-1).sum()),later_joint_ready=int(data['later_ready'].any(-1).sum()),terminal_replay_error=terminal_error))
(a.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(dict(groups=groups,runtime_s=time.perf_counter()-started,scope='Known DEV, fixed125-step/2.5s horizon. Post-classifier model steps are diagnostic extrapolation, not verified actual RB continuation. No controls, weights or phase timing changed.'),indent=2)+'\n')
print(json.dumps(groups,indent=2))
