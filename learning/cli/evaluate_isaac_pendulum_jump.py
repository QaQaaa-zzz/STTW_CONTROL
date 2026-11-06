"""Paired frozen Phase U / RB handoff on the common pendulum bicycle."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
spec=json.loads(a.config.read_text());out=Path(spec['output']);out.mkdir(exist_ok=True,parents=True)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import torch
import isaaclab.sim as sim_utils
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.pendulum_skill import TorchPendulumObservation,TorchPendulumSkill,TorchPendulumGeometry
from sttw_control.torch_controller import TorchController
cfg=json.loads(Path('learning/configs/isaac_pendulum_local300_standard.json').read_text())
options=cfg['plant_options'].copy();options['env_spacing']=20.;options['pendulum_locked']=True
plant=IsaacLabVectorPlant(Path(cfg['asset']),['plane'],[0,0],out,**options)
# Same source step dimensions, translated independently into each lane.
for i,origin in enumerate(plant.origin.cpu().numpy()):
 box=sim_utils.CuboidCfg(size=(4.,2.,.32),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.01,rest_offset=0.),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8))
 box.func(f'/World/JumpStep_{i}',box,translation=(float(origin[0]+5.6),float(origin[1]),0.))
saved=np.load(spec['initial_state'])
state={k:torch.tensor(saved[k],device='cuda:0') for k in ('root_pose','root_velocity','joint_position','joint_velocity')}
state['root_pose']=state['root_pose'].clone();state['root_pose'][0]=spec['start_x'];state['root_pose'][1]=0.
if spec.get('align_heading',False):
 from isaaclab.utils.math import quat_mul,quat_apply
 q=state['root_pose'][3:];w,x,y,z=q
 yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
 rotation=torch.stack((torch.cos(yaw/2),yaw*0,yaw*0,-torch.sin(yaw/2)))
 state['root_pose'][3:]=quat_mul(rotation[None],q[None])[0]
 state['root_velocity'][:3]=quat_apply(rotation[None],state['root_velocity'][:3][None])[0]
 state['root_velocity'][3:]=quat_apply(rotation[None],state['root_velocity'][3:][None])[0]

plant.reset_idx(torch.arange(2,device='cuda:0'),state=state)
plant.set_pendulum_locked(False);plant.set_pendulum_locked(True)
controller=TorchController(json.loads(Path(cfg['controller_config']).read_text()),2,'cuda:0')
policies=[TorchPendulumSkill(Path(spec['bundles'])/s) for s in spec['skills']]
history=TorchPendulumObservation(plant.joints,json.loads(Path('learning/configs/pendulum_common_asset.json').read_text())['mount_rotation_wxyz'],2,'cuda:0')
geometry=TorchPendulumGeometry(json.loads(Path(spec['geometry']).read_text()),plant.bodies)
last=torch.zeros(2,4,device='cuda:0');alive=torch.ones(2,dtype=torch.bool,device='cuda:0');entered=torch.zeros_like(alive);consumed=torch.zeros_like(alive)
active=False;rows=[];actions=[];observations=[];valid=[];start=time.monotonic();steps=round(spec['seconds']/.005)
(out/'protocol.json').write_text(json.dumps(dict(spec=spec,plant=plant.metadata,policies=[s.identity for s in policies],ground_history='last_action zero means no prior jump action; physical FIFO sampled throughout ground approach',initial_translation='common saved snapshot translated to start_x/y0; optional rigid world-yaw alignment rotates pose and both velocities together',role='nominal paired DEV integration; not original donor benchmark',physics_dt=.0002,ground_dt=.005,jump_dt=.02,optimizer_updates=0),indent=2))
target=torch.zeros(2,4,device='cuda:0')
for k in range(steps+1):
 m,pose,speed=plant.observe();root=plant.roots.clone();root[:,:3]-=plant.origin
 distance=3.6-(geometry.frontmost_x(plant.rigid)-plant.origin[:,0])
 row=torch.cat((torch.full((2,1),k*.005,device='cuda:0'),root[:,:3],speed[:,None],m[:,:4],plant.dofs[:,plant.pendulum_servos.indices,0],distance[:,None]),-1)
 rows.append(row);valid.append(alive.clone());actions.append(last.clone())
 alive &= torch.isfinite(row).all(-1)&(m[:,0].abs()<torch.pi/4)&(root[:,2]>0.05)
 if k==steps or not bool(alive.any()):break
 if k%4==0:
  inside=(root[:,0]>=2.5)&(root[:,0]<=4.0);consumed|=entered&~inside;entered|=inside
  signal=(inside&~consumed).float()
  obs=history.advance(root,plant.dofs[:,:,0],plant.dofs[:,:,1],plant.robot.data.body_com_acc_w[:,plant.bodies.index('chasis'),:3],distance,last,signal)
  observations.append(obs)
  if not active and bool(inside.any()):
   plant.set_pendulum_locked(False);active=True
   (out/'handoff.json').write_text(json.dumps(dict(time=k*.005,root=root.cpu().tolist(),position=plant.dofs[:,:,0].cpu().tolist(),velocity=plant.dofs[:,:,1].cpu().tolist(),distance=distance.cpu().tolist()),indent=2))
  if active:
   for i,policy in enumerate(policies):last[i:i+1],target[i:i+1]=policy.infer(obs[i:i+1],plant.dofs[i:i+1,plant.joints.index('knee_joint'),0])
 if active:
  command=torch.stack((torch.zeros_like(speed),target[:,1]),-1)
  for _ in range(25):plant.step(command,pendulum_target=target[:,2:],steering_position_target=target[:,:1],refresh_diagnostics=False)
 else:
  request=torch.stack((m[:,5]*.1,m[:,2],m[:,3],m[:,0],m[:,1],torch.zeros_like(speed)),-1)
  control=controller.step(request,torch.ones_like(alive));command=torch.stack((control[:,0].clamp(-3,3),torch.full_like(speed,spec['approach_speed']/.1)),-1)
  for _ in range(25):plant.step(command,refresh_diagnostics=False)
 plant.refresh()
trace=torch.stack(rows).cpu().numpy();mask=torch.stack(valid).cpu().numpy();np.savez(out/'trajectory.npz',trace=trace,valid=mask,action=torch.stack(actions).cpu().numpy(),observations=torch.stack(observations).cpu().numpy(),columns=['time','x','y','z','speed','roll','roll_rate','steer','steer_rate','hip','knee','obstacle_relative_x'])
result=dict(status='complete',elapsed_s=time.monotonic()-start,physics_lane_steps=k*25*2,cases=[])
for i,name in enumerate(spec['skills']):
 tr=trace[mask[:,i],i];result['cases'].append(dict(skill=name,duration_s=float(tr[-1,0]),physical_termination=not bool(alive[i]),peak_root_z=float(tr[:,3].max()),final_xyz=tr[-1,1:4].tolist(),peak_abs_roll_deg=float(np.rad2deg(abs(tr[:,5])).max()),full_jump_success='not adjudicated; requires clearance and landing evidence'))
(out/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
app.close(wait_for_replicator=False,skip_cleanup=True)
