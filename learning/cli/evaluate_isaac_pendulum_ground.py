"""Single-lane ground diagnostic on the common pendulum chassis in Isaac Lab."""
import os,argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.torch_controller import TorchController
cfg=json.loads(Path('runs/isaac_signals_parity/dynamic.json').read_text())
options=json.loads(Path('learning/configs/isaaclab_h75_offroad_selected.json').read_text())['plant_options']
spec=json.loads(a.config.read_text()) if a.config else dict(duration_s=3.,warmup_s=3.,return_s=3.,speed_m_s=2.,steer_slew_rad_s=.8,cases=[dict(name='ecbc',steer_rad=0.,local300=False)])
if spec.get('controller_config'):cfg=json.loads(Path(spec['controller_config']).read_text())
options.update(spec.get('plant_options',{}));radius=options.get('wheel_radius_m',.1);cfg['wheel_radius_m']=radius
substeps=round(.005/options.get('physics_dt',.0002))
cases=spec['cases'];n=len(cases);steps=round(spec['duration_s']/.005)
plant=IsaacLabVectorPlant(a.asset,['plane'],[0]*n,a.output,pendulum_rest=[-1.2,2.499999],pendulum_locked=spec.get('pendulum_locked',False),**options)
if any('controller_wc' in case for case in cases):cfg['wc']=[case.get('controller_wc',cfg['wc']) for case in cases]
controller=TorchController(cfg,n,'cuda:0')
from sttw_control.torch_local_lower import TorchLocalLower
from sttw_control.torch_path_reference import wrap
selected=json.loads(Path('learning/configs/isaaclab_h75_offroad_selected.json').read_text())
lower=TorchLocalLower(selected['lower_actor'],selected['lower_sha256'],n,'cuda:0')
plant.robot.data.default_root_state[:,2]+=torch.tensor([c.get('root_height_offset_m',0.) for c in cases],device='cuda:0')
plant.reset(speed=spec['speed_m_s'])
np.savez(a.output/'reset_state.npz',body_state=plant.rigid.cpu().numpy(),body_names=plant.bodies,joint_position=plant.dofs[:,:,0].cpu().numpy(),joint_names=plant.joints,origin=plant.origin.cpu().numpy())
(a.output/'protocol.json').write_text(json.dumps(dict(asset=str(a.asset),controller=cfg,plant=plant.metadata,spec=spec,lower_actor=selected['lower_actor'],lower_sha256=selected['lower_sha256'],role='ground compatibility DEV; no H75 or training',pendulum_rest_rad=[-1.2,2.499999],max_physics_steps=steps*n*substeps,optimizer_updates=0),indent=2))
device='cuda:0';raw=torch.tensor([spec['speed_m_s'],0.],device=device).repeat(n,1);previous=raw.clone();command=torch.tensor([0.,spec['speed_m_s']/radius],device=device).repeat(n,1)
use_lower=torch.tensor([c['local300'] for c in cases],device=device)
lower_start=torch.tensor([c.get('lower_start_s',spec['warmup_s']) for c in cases],device=device)
eso_enabled=torch.tensor([c.get('eso_enabled',True) for c in cases],device=device)
true_feedback=torch.tensor([c.get('true_speed_feedback',False) for c in cases],device=device)
turn=torch.tensor([c['steer_rad'] for c in cases],device=device)
open_loop=torch.tensor(['open_loop_steer_rate' in c for c in cases],device=device)
pulse=torch.tensor([c.get('open_loop_steer_rate',0.) for c in cases],device=device)
alive=torch.ones(n,dtype=torch.bool,device=device);rows=[];masks=[];references=[];commands=[];contacts=[];start=time.monotonic()
for k in range(steps+1):
 m,pose,v=plant.observe()
 row=torch.cat((torch.full((n,1),k*.005,device=device),pose,v[:,None],m[:,:4],plant.dofs[:,plant.pendulum_servos.indices,0],plant.dofs[:,plant.pendulum_servos.indices,1],m[:,5:7]*radius,plant.roots[:,2:3],controller.disturbance[:,None]),dim=1)
 contacts.append(plant.contact.cpu().numpy().copy())
 rows.append(row.cpu().numpy());masks.append(alive.cpu().numpy().copy());references.append(raw.cpu().numpy().copy());commands.append(command.cpu().numpy().copy())
 alive &= (m[:,0].abs()<=torch.pi/4)&torch.isfinite(row).all(-1)
 if spec.get('save_warmup_state',False) and k==round(spec['warmup_s']/.005) and bool(alive[0]):
  root=plant.roots[0].clone();root[:3]-=plant.origin[0]
  state=dict(root_pose=torch.cat((root[:3],root[[6,3,4,5]])),root_velocity=root[7:],joint_position=plant.dofs[0,:,0],joint_velocity=plant.dofs[0,:,1],previous_reference=previous[0],previous_command=command[0],raw=raw[0],eso=controller.eso[0],disturbance=controller.disturbance[0],gains=controller.gains[0])
  np.savez(a.output/'initial_state.npz',**{name:value.cpu().numpy() for name,value in state.items()},joint_names=plant.joints,physical_tick=k)
 if k==steps or not bool(alive.any()):break
 desired=turn if spec['warmup_s']<=k*.005<spec['return_s'] else torch.zeros_like(turn)
 raw=raw.clone();raw[:,1]+=torch.clamp(desired-raw[:,1],-spec['steer_slew_rad_s']*.005,spec['steer_slew_rad_s']*.005)
 action,obs,finite=lower.prepare(m,v,raw,previous,command,controller)
 applied=torch.where((use_lower & (k*.005>=lower_start))[:,None],action*torch.tensor([1.5,10.],device=device),0.)
 request=torch.stack((torch.where(true_feedback,v,m[:,5]*radius),m[:,2],m[:,3],m[:,0],m[:,1],raw[:,1]),dim=-1)
 out=controller.step(request,eso_enabled)
 steer=(out[:,0]+applied[:,0]).clamp(-3,3);steer=torch.maximum(torch.maximum(torch.full_like(steer,-3),(-.8-m[:,2])/.005),torch.minimum(torch.minimum(torch.full_like(steer,3),(.8-m[:,2])/.005),steer))
 steer=torch.where(open_loop,pulse if .1<=k*.005<.3 else torch.zeros_like(pulse),steer)
 command=torch.stack((steer,(raw[:,0]/radius+applied[:,1]).clamp(-60,60)),dim=-1)
 for _ in range(substeps):plant.step(torch.where(alive[:,None],command,0.))
 _,new_pose,_=plant.observe();lower.finish(applied,wrap(new_pose[:,2]-pose[:,2])/.005);previous=raw.clone()
 if (k+1)%200==0:print(dict(control_steps=k+1,alive=int(alive.sum()),elapsed_s=time.monotonic()-start),flush=True)
trace=np.asarray(rows);valid=np.asarray(masks);np.savez(a.output/'trajectory.npz',trace=trace,valid=valid,contact=np.asarray(contacts),contact_body_names=plant.bodies,reference=np.asarray(references),command=np.asarray(commands),columns=['time','x','y','yaw','speed','roll','roll_rate','steer','steer_rate','hip','knee','hip_velocity','knee_velocity','rear_wheel_speed_estimate','front_wheel_speed_estimate','root_height','eso_disturbance'])
results=[]
for i,case in enumerate(cases):
 tr=trace[valid[:,i],i];results.append(dict(name=case['name'],failed=not bool(alive[i]),duration_s=float(tr[-1,0]),peak_roll_deg=float(np.rad2deg(abs(tr[:,5])).max()),final_speed=float(tr[-1,4]),final_xy=tr[-1,1:3].tolist(),max_pendulum_position_drift_rad=np.max(np.abs(tr[:,9:11]-tr[0,9:11]),axis=0).tolist(),max_pendulum_speed_rad_s=np.max(np.abs(tr[:,11:13]),axis=0).tolist()))
result=dict(status='complete',cases=results,physics_steps=k*substeps*n,elapsed_s=time.monotonic()-start)
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if spec.get('pendulum_locked',False):
 before=plant.robot.data.joint_pos.clone();plant.set_pendulum_locked(False)
 (a.output/'release.json').write_text(json.dumps(dict(state_reset=False,position_change_without_step=float((plant.robot.data.joint_pos-before).abs().max()),restored_limits=plant.robot.data.joint_pos_limits[:,plant.pendulum_servos.indices].cpu().tolist(),original_limits=plant.pendulum_original_limits.cpu().tolist()),indent=2))
os.sync();os._exit(0)
