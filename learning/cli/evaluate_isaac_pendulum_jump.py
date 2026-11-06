"""Paired frozen Phase U / RB handoff on the common pendulum bicycle."""
import os,argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
spec=json.loads(a.config.read_text());out=Path(spec['output']);out.mkdir(exist_ok=True,parents=True)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0',enable_cameras=bool(spec.get('video',False))).app
import torch
import isaaclab.sim as sim_utils
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.pendulum_skill import TorchPendulumObservation,TorchPendulumSkill,TorchPendulumGeometry
from sttw_control.torch_controller import TorchController
cfg=json.loads(Path(spec.get('plant_config','learning/configs/isaac_pendulum_local300_standard.json')).read_text())
options=cfg['plant_options'].copy();options['env_spacing']=20.;options['pendulum_locked']=True
options['physics_dt']=spec.get('physics_dt',.0002)
if 'terrain_base_plane' in spec:options['terrain_base_plane']=spec['terrain_base_plane']
options['gpu_pair_capacities']=spec.get('gpu_pair_capacities')
options['actuator_force_limits']=spec.get('actuator_force_limits')
substeps=round(.005/options['physics_dt'])
if substeps<1 or abs(substeps*options['physics_dt']-.005)>1e-9:raise ValueError('physics_dt must divide ground/sensor period .005')
n=len(spec['skills'])
plant=IsaacLabVectorPlant(Path(cfg['asset']),[spec.get('scene_terrain','plane')],[0]*n,out,**options)
if 'jump_steering_gains' in spec:
 plant.jump_steering_servo.kp.fill_(spec['jump_steering_gains'][0]);plant.jump_steering_servo.kd.fill_(spec['jump_steering_gains'][1])
# Same source step dimensions, translated independently into each lane.
for i,origin in enumerate([] if spec.get('scene_terrain') else plant.origin.cpu().numpy()):
 box=sim_utils.CuboidCfg(size=(4.,2.,.32),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.01,rest_offset=0.),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8))
 box.func(f'/World/JumpStep_{i}',box,translation=(float(origin[0]+5.6),float(origin[1]),0.))
camera=None;writer=None
if spec.get('video',False):
 from isaaclab.sensors.camera import Camera,CameraCfg
 import imageio.v2 as imageio
 light=sim_utils.DomeLightCfg(intensity=2000.);light.func('/World/VideoLight',light)
 camera=Camera(CameraCfg(prim_path='/World/VideoCamera',update_period=0,height=720,width=1280,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=28.,clipping_range=(.05,100.))))
 plant.sim.reset()
 camera.set_world_poses_from_view(torch.tensor([[4.,-5.,2.3]],device='cuda:0'),torch.tensor([[3.8,0.,.35]],device='cuda:0'))
 writer=imageio.get_writer(out/'simulation.mp4',fps=50,codec='libx264',quality=8,macro_block_size=1)
saved=np.load(spec['initial_state'])
state={k:torch.tensor(saved[k],device='cuda:0',dtype=torch.float32) for k in ('root_pose','root_velocity','joint_position','joint_velocity')}
state['root_pose']=state['root_pose'].clone();state['root_pose'][0]=spec['start_x'];state['root_pose'][1]=spec.get('start_y',0.)
if spec.get('align_heading',False) or 'start_heading_rad' in spec:
 from isaaclab.utils.math import quat_mul,quat_apply
 q=state['root_pose'][3:];w,x,y,z=q
 yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
 delta=spec.get('start_heading_rad',0.)-yaw
 rotation=torch.stack((torch.cos(delta/2),yaw*0,yaw*0,torch.sin(delta/2)))
 state['root_pose'][3:]=quat_mul(rotation[None],q[None])[0]
 state['root_velocity'][:3]=quat_apply(rotation[None],state['root_velocity'][:3][None])[0]
 state['root_velocity'][3:]=quat_apply(rotation[None],state['root_velocity'][3:][None])[0]

plant.reset_idx(torch.arange(n,device='cuda:0'),state=state)
plant.set_pendulum_locked(False);plant.set_pendulum_locked(True)
controller=TorchController(json.loads(Path(cfg['controller_config']).read_text()),n,'cuda:0')
policies=[TorchPendulumSkill(Path(spec['bundles'])/s) for s in spec['skills']]
history=TorchPendulumObservation(plant.joints,json.loads(Path(spec.get('model_config','learning/configs/pendulum_common_asset.json')).read_text())['mount_rotation_wxyz'],n,'cuda:0')
geometry=TorchPendulumGeometry(json.loads(Path(spec['geometry']).read_text()),plant.bodies)
last=torch.zeros(n,4,device='cuda:0');alive=torch.ones(n,dtype=torch.bool,device='cuda:0');entered=torch.zeros_like(alive);consumed=torch.zeros_like(alive)
returned=False;landing_ticks=0;landing_ready=False;transition_speed=None
if spec.get('landing_return'):assert n==1, 'Landing return currently uses a single lane'
ground_rows=[]
active=False;rows=[];actions=[];observations=[];valid=[];arm_rows=[];body_rows=[];contact_rows=[];start=time.monotonic();steps=round(spec['seconds']/.005)
(out/'protocol.json').write_text(json.dumps(dict(spec=spec,plant=plant.metadata,policies=[s.identity for s in policies],ground_history='last_action zero means no prior jump action; physical FIFO sampled throughout ground approach',initial_translation='common saved snapshot translated to start_x/y0; optional rigid world-yaw alignment rotates pose and both velocities together',role='nominal paired DEV integration; not original donor benchmark',physics_dt=plant.physics_dt,ground_dt=.005,jump_dt=.02,optimizer_updates=0),indent=2))
ground=None
if spec.get('ground_bundle'):
 from sttw_control.isaac_path_rollout import IsaacPathRollout
 from sttw_control.torch_path_policy import TorchPathPolicy
 from sttw_control.torch_path_reference import TorchPathReference
 ground_saved=np.load(spec.get('ground_initial_state',spec['initial_state']))
 ground_state={key:ground_saved[key] for key in ground_saved.files}
 ground_state.update({key:value.cpu().numpy() for key,value in state.items()})
 np.savez(out/'ground_initial.npz',**ground_state)
 settings=json.loads(Path(spec.get('ground_settings','learning/configs/isaaclab_h75_offroad_selected.json')).read_text())
 settings.update(num_envs=n,alphas=[spec['ground_alpha']]*n,zero_middle=[False]*n,zero_lower=[spec.get('zero_lower',False)]*n,terrains={'plane':'plane'},episode_s=spec['seconds'],initial_state=str(out/'ground_initial.npz'),user_speed_m_s=spec['approach_speed'],auto_reset_horizon=False)
 bundle=Path(spec['ground_bundle']);identity=json.loads((bundle/'identity.json').read_text());ground_cfg=json.loads((bundle/'config.json').read_text())
 actor=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],ground_cfg,n,'cuda:0')
 ground=IsaacPathRollout(bundle,settings,actor,plant=plant)
 ss=np.arange(0,80.01,.02);route=dict(s=ss,xy=np.column_stack((ss+spec['start_x'],np.zeros_like(ss))),heading=np.zeros_like(ss),curvature=np.zeros_like(ss))
 if spec.get('ground_route'):route=dict(np.load(spec['ground_route']))
 np.savez(out/'ground_route.npz',**route)
 ground.begin_connection(TorchPathReference(route,ground.config,ground.device),ground.speed.clone(),ground.alpha.clone())
 if spec.get('ground_route'):
  gm,gpose,gv=plant.observe()
  ground.middle.reset(gm,gpose,gv,ground.previous_reference,ground.previous_reference,ground.previous_command,ground.controller)
 (out/'ground_controller.json').write_text(json.dumps(dict(upper_sha256=ground.upper_sha256,lower_sha256=ground.lower_sha256,alpha=spec['ground_alpha'],physics_dt=plant.physics_dt,control_dt=.005,policy_dt=.02,physical_substeps=ground.physics_substeps),indent=2))
target=torch.zeros(n,4,device='cuda:0')
root_acceleration=torch.zeros(n,3,device='cuda:0')
previous_root_velocity=plant.roots[:,7:10].clone()
for k in range(steps+1):
 if ground is not None and not active and (returned or not spec.get('ground_after_landing_only',False)) and k%4:continue
 m,pose,speed=plant.observe();root=plant.roots.clone();root[:,:3]-=plant.origin
 distance=3.6-(geometry.frontmost_x(plant.rigid)-plant.origin[:,0])
 row=torch.cat((torch.full((n,1),k*.005,device='cuda:0'),root[:,:3],speed[:,None],m[:,:4],plant.dofs[:,plant.pendulum_servos.indices,0],distance[:,None]),-1)
 rows.append(row);valid.append(alive.clone());actions.append(last.clone())
 if spec.get('record_body_states',False):
  body=plant.rigid.clone();body[:,:,:3]-=plant.origin[:,None,:];body_rows.append(body);contact_rows.append(plant.contact.clone())
 if camera is not None and k%4==0:
  plant.sim.render();camera.update(.02,force_recompute=True)
  writer.append_data(camera.data.output['rgb'][0,:,:,:3].cpu().numpy())
 if ground is not None and not active and (returned or not spec.get('ground_after_landing_only',False)):alive &= ground.alive
 alive &= torch.isfinite(row).all(-1)&(m[:,0].abs()<torch.pi/4)&(root[:,2]>0.05)
 if k==steps or not bool(alive.any()):break
 if k%4==0:
  inside=(root[:,0]>=2.5)&(root[:,0]<=4.0);consumed|=entered&~inside;entered|=inside
  signal=(inside&~consumed).float()
  acceleration=root_acceleration if spec.get('root_acceleration_difference',False) else plant.robot.data.body_com_acc_w[:,plant.bodies.index('chasis'),:3]
  obs=torch.cat((history.frames.flatten(1),signal[:,None]),-1) if k==0 and spec.get('zero_initial_history',False) else history.advance(root,plant.dofs[:,:,0],plant.dofs[:,:,1],acceleration,distance,last,signal)
  observations.append(obs)
  if not active and not returned and bool((root[:,0]>=spec.get('handoff_x',2.5)).any()):
   plant.set_pendulum_locked(False);active=True
   (out/'handoff.json').write_text(json.dumps(dict(time=k*.005,root=root.cpu().tolist(),position=plant.dofs[:,:,0].cpu().tolist(),velocity=plant.dofs[:,:,1].cpu().tolist(),distance=distance.cpu().tolist()),indent=2))
  if active:
   for i,policy in enumerate(policies):last[i:i+1],target[i:i+1]=policy.infer(obs[i:i+1],plant.dofs[i:i+1,plant.joints.index('knee_joint'),0])
 if active and spec.get('landing_return'):
  # Contact-triggered handback; retain the physical pose/velocity and lock the current arm posture.
  from isaaclab.utils.math import euler_xyz_from_quat
  ids=[plant.bodies.index(name) for name in ('frontwheel','rearwheel')]
  wheels=plant.rigid[:,ids,:3]-plant.origin[:,None,:]
  roll,pitch,yaw=euler_xyz_from_quat(plant.robot.data.root_link_quat_w)
  pitch=torch.atan2(torch.sin(pitch),torch.cos(pitch))
  on_platform=((wheels[:,:,0]>3.6)&(wheels[:,:,0]<7.6)&(wheels[:,:,1].abs()<.9)&((wheels[:,:,2]-.23).abs()<.025)).all()
  supported=(plant.contact[:,ids,2]>1.).all()
  settled=bool(on_platform & supported & (m[:,0].abs()<.15).all() & (pitch.abs()<.15).all())
  landing_ticks=landing_ticks+1 if settled else 0
  # Latch stable support, then leave the low-speed ECBC gain singularity under jump steering.
  if not landing_ready and landing_ticks*.005>=spec['landing_return']['support_s']:
   landing_ready=True;transition_speed=speed.clone()
   (out/'landing_ready.json').write_text(json.dumps(dict(time=k*.005,speed=speed.cpu().tolist())))
  if landing_ready and spec['landing_return'].get('minimum_handoff_speed_m_s',0)>0:
   transition_speed=(transition_speed+spec['landing_return']['acceleration_m_s2']*.005).clamp_max(spec['landing_return']['speed_m_s'])
   target[:,1]=transition_speed/plant.wheel_radius_m
  if landing_ready and bool((speed>=spec['landing_return'].get('minimum_handoff_speed_m_s',0)).all()) and k%4==0:
   plant.set_pendulum_locked(True);active=False;returned=True;controller.reset(slice(None))
   if ground is not None:
    ground.controller.reset(slice(None));ground.lower.reset(slice(None));ground.alive.fill_(True)
    ground.speed.fill_(spec['landing_return']['speed_m_s'])
    ground.measurement=m;ground.previous_reference=torch.stack((speed,m[:,2]),-1)
    ground.previous_command=torch.stack((m[:,3],m[:,5]),-1)
    ground.middle.reset(m,pose,speed,ground.previous_reference,ground.previous_reference,ground.previous_command,ground.controller)
    ground.begin_connection(ground.route,ground.speed.clone(),ground.alpha.clone())
    nearest=(ground.route.xy[None]-pose[:,None,:2]).square().sum(-1).argmin(-1)
    ground.middle.projection=ground.route.project(pose[:,:2],ground.route.s[nearest],torch.zeros_like(speed))
    ground.middle.follower=ground.route.pursuit(ground.middle.projection['progress'],pose,speed,ground.speed)
    ground.middle.fault=ground.middle.follower['numerical_fault'].clone();ground.middle.domain_exit=ground.middle.follower['invalid'].clone()
   (out/'landing_handoff.json').write_text(json.dumps(dict(time=k*.005,root=root.cpu().tolist(),position=plant.dofs[:,:,0].cpu().tolist(),velocity=plant.dofs[:,:,1].cpu().tolist(),controller_state='ECBC ESO initialized for resumed ground control; no physical reset',arm='locked at current posture'),indent=2))
 if active:
  command=torch.stack((torch.zeros_like(speed),target[:,1]),-1)
  for substep in range(substeps):
   ids=plant.pendulum_servos.indices
   before=torch.cat((torch.full((n,1),k*.005+substep*plant.physics_dt,device='cuda:0'),target[:,2:],plant.dofs[:,ids,0],plant.dofs[:,ids,1]),-1)
   plant.step(command,pendulum_target=target[:,2:],steering_position_target=target[:,:1],refresh_diagnostics=False)
   velocity=plant.robot.data.root_link_lin_vel_w
   root_acceleration=(velocity-previous_root_velocity)/plant.physics_dt;previous_root_velocity=velocity.clone()
   arm_rows.append(torch.cat((before,plant.last_pendulum_torque[:,ids],alive[:,None]),-1))
 elif ground is not None and (returned or not spec.get('ground_after_landing_only',False)):
  result=ground.step();ground_rows.append(result['lower_trace'].clone())
  velocity=plant.robot.data.root_link_lin_vel_w
  root_acceleration=(velocity-previous_root_velocity)/.02;previous_root_velocity=velocity.clone()
 else:
  request=torch.stack((m[:,5]*plant.wheel_radius_m,m[:,2],m[:,3],m[:,0],m[:,1],torch.zeros_like(speed)),-1)
  control=controller.step(request,torch.ones_like(alive));command=torch.stack((control[:,0].clamp(-3,3),torch.full_like(speed,(spec['landing_return']['speed_m_s'] if returned else spec['approach_speed'])/plant.wheel_radius_m)),-1)
  for _ in range(substeps):
   plant.step(command,refresh_diagnostics=False)
   velocity=plant.robot.data.root_link_lin_vel_w
   root_acceleration=(velocity-previous_root_velocity)/plant.physics_dt;previous_root_velocity=velocity.clone()
 if spec.get('record_body_states',False):plant.sensor.update(.005,force_recompute=True)
 plant.refresh()
if ground_rows:np.savez(out/'ground_tracking.npz',trace=torch.cat(ground_rows).cpu().numpy(),columns=['x','y','speed','cross_track','abs_roll','reward','alive','physics_ticks','speed_reference','steer_reference','wheel_speed_estimate'])
if writer is not None:writer.close()
if body_rows:
 np.savez(out/'body_contact.npz',time=torch.stack(rows)[:,0,0].cpu().numpy(),body_state=torch.stack(body_rows).cpu().numpy(),contact=torch.stack(contact_rows).cpu().numpy(),body_names=plant.bodies,valid=torch.stack(valid).cpu().numpy(),semantics='world states relative to lane origin; actual per-body net contact forces; no post-termination samples scored')
if arm_rows:
 np.savez(out/'arm_commands.npz',trace=torch.stack(arm_rows).cpu().numpy(),columns=['time','hip_target','knee_target','hip_position','knee_position','hip_velocity','knee_velocity','hip_command_torque','knee_command_torque','alive_before_step'],semantics='pre-step state and actual clipped PD effort command each physics step; not solver-measured reaction torque')
trace=torch.stack(rows).cpu().numpy();mask=torch.stack(valid).cpu().numpy();np.savez(out/'trajectory.npz',trace=trace,valid=mask,action=torch.stack(actions).cpu().numpy(),observations=torch.stack(observations).cpu().numpy(),columns=['time','x','y','z','speed','roll','roll_rate','steer','steer_rate','hip','knee','obstacle_relative_x'])
result=dict(status='complete',elapsed_s=time.monotonic()-start,physics_lane_steps=k*substeps*n,landing_return_executed=returned,ground_controller=('none; jump policy from reset' if spec.get('handoff_x',2.5)<=spec['start_x'] else (('H75+ECBC (zero lower residual)' if spec.get('zero_lower') else 'H75+Local300') if ground is not None else 'ECBC+ESO')),cases=[])
for i,name in enumerate(spec['skills']):
 tr=trace[mask[:,i],i];result['cases'].append(dict(skill=name,duration_s=float(tr[-1,0]),physical_termination=not bool(alive[i]),peak_root_z=float(tr[:,3].max()),final_xyz=tr[-1,1:4].tolist(),peak_abs_roll_deg=float(np.rad2deg(abs(tr[:,5])).max()),full_jump_success='not adjudicated; requires clearance and landing evidence'))
(out/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
os.sync();os._exit(0)
