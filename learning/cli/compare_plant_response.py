"""Same-input plant response, with no controller or policy feedback."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--servo-mode',choices=['explicit','implicit'],default='explicit');p.add_argument('--backend',choices=['mujoco','lab'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('learning/configs/isaaclab_local300_transfer.json'));a=p.parse_args()
s=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
dt=.0002;t=np.arange(5000)*dt;commands=np.zeros((5000,4,2),np.float32);commands[:,:,1]=23
commands[(t>=.1)&(t<.6),1,1]=28
commands[(t>=.1)&(t<.3),2,0]=.3;commands[(t>=.1)&(t<.3),3,0]=-.3
names=['hold','rear_step','steer_positive','steer_negative'];rows=[]
if a.backend=='mujoco':
 import mujoco
 from sttw_control.model import load_model
 b=load_model();m=b.model;bank=pickle.load(open(s['initial_bank'],'rb'));data=[]
 for _ in names:
  d=mujoco.MjData(m);d.qpos[:]=bank.data.qpos[s['initial_bank_index']];d.qvel[:]=bank.data.qvel[s['initial_bank_index']];mujoco.mj_forward(m,d);data.append(d)
 aid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,k) for k in ['cmd_steering_v','cmd_rearwheel_v']]
 for command in commands:
  lane=[]
  for i,d in enumerate(data):
   d.ctrl[aid]=command[i]*[1,-1];mujoco.mj_step(m,d);force=d.actuator_force[aid].copy();mujoco.mj_forward(m,d)
   w,x,y,z=d.qpos[3:7];roll=-(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))-np.pi/2);mat=d.xmat[b.chassis].reshape(3,3);yaw=np.arctan2(mat[1,0],mat[0,0]);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b.chassis,v,0)
   lane.append([*d.xpos[b.chassis,:2],yaw,roll,v[3]*np.cos(yaw)+v[4]*np.sin(yaw),d.qpos[b.steer_qpos],d.qvel[b.steer_dof],-d.qvel[b.rear_dof],-d.qvel[b.front_dof],*force])
  rows.append(lane)
 identity=b.identity
else:
 from isaaclab.app import AppLauncher
 app=AppLauncher(headless=True,device='cuda:0').app
 import torch
 from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
 plant=IsaacLabVectorPlant('runs/isaac_urdf_drive/vehicle.urdf',['plane'],[0]*4,a.output,servo_mode=a.servo_mode)
 x=np.load('runs/local300_sim2sim/initial_state.npz');state={k:torch.as_tensor(x[k],dtype=torch.float32,device=plant.device) for k in ['root_pose','root_velocity','joint_position','joint_velocity']};order=[list(x['joint_names']).index(k) for k in plant.joints]
 for k in ['joint_position','joint_velocity']:state[k]=state[k][order]
 plant.reset_idx(torch.arange(4,device=plant.device),state=state)
 for command in torch.as_tensor(commands,device=plant.device):
  force=plant.servos.torque(command,plant.dofs[:,:,1])[:,plant.servos.indices].clone();plant.step(command);m,pose,v=plant.observe();rows.append(torch.cat((pose,m[:,0:1],v[:,None],m[:,2:4],m[:,5:7],force),dim=-1))
 rows=torch.stack(rows).cpu().numpy();identity=plant.metadata
np.savez_compressed(a.output/'response.npz',time=t+dt,command=commands,values=np.asarray(rows),cases=names,fields=['x','y','yaw','roll','body_speed','steer','steer_rate','rear_rate','front_rate','steer_torque','rear_torque'])
(a.output/'status.json').write_text(json.dumps(dict(stage='complete',backend=a.backend,steps=20000,identity=identity,torque_semantics=('solver actuator force' if a.backend=='mujoco' else 'prestep explicit servo applied torque' if a.servo_mode=='explicit' else 'prestep servo estimate, not solver applied torque'),scope='open-loop velocity-command response; no ECBC/ESO/policy; original physical limits; no matched solver warmstart'),indent=2))
if a.backend=='lab':plant.close();app.close(wait_for_replicator=False)
