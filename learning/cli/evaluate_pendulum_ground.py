"""Bounded native-model ground compatibility diagnostic, no training."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import mujoco
import jax
import jax.numpy as jp
from sttw_control.controller import ControllerConfig, initial_controller, controller_step
from jit_dvgc.config import load_config
from jit_dvgc.model import load_host_model

p=argparse.ArgumentParser()
p.add_argument('--registry',type=Path,default=Path('learning/configs/pendulum_bike_skill_sources.json'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
r=json.loads(a.registry.read_text())['records']['source']['policy']
raw=json.loads(Path(r['formal_config']).read_text())
b=load_host_model(load_config(Path(raw['inputs']['up_config_path']),runtime_only=True))
m=b.mj_model;idx=b.model_index;dt=float(m.opt.timestep)
c=ControllerConfig(dt=dt)
step=jax.jit(lambda state,row:controller_step(state,row,True,c))
manifest=dict(role='nominal compatibility diagnostic, not skill handoff',model_sha256=b.xml_sha256,physics_dt=dt,controller=asdict(c),duration_s=3,initial_x=-4,requested_speed=2,methods=['fixed_steer','ecbc'],physics_budget=1200,optimizer_updates=0,steering_adapter='integrate commanded rate, clip position to native bounds',hip_knee='native keyframe position held',network='none',failure_roll_deg=45)
(a.output/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
results=[]
for method in manifest['methods']:
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,idx.keyframe_id)
 d.qpos[0]=-4;d.qvel[0]=2
 d.ctrl[:]=[0,2/.07,-1.2,2.5];mujoco.mj_forward(m,d)
 state=initial_controller(c);target=0.;rows=[];failed=False
 for tick in range(601):
  w,x,y,z=d.qpos[3:7];roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
  speed=d.qvel[0]*np.cos(yaw)+d.qvel[1]*np.sin(yaw)
  rows.append([d.time,*d.qpos[:3],roll,yaw,speed,d.qpos[idx.steering_qpos_address],target,*d.ctrl])
  failed=not np.isfinite(d.qpos).all() or abs(roll)>np.pi/4
  if failed or tick==600:break
  if method=='ecbc':
   row=jp.array([speed,d.qpos[idx.steering_qpos_address],d.qvel[idx.steering_dof_address],-roll,-d.qvel[3],0.])
   state,out=step(state,row)
   target=float(np.clip(target+dt*float(out.steer_rate),-.8,.8))
  d.ctrl[:]=[target,2/.07,-1.2,2.5]
  mujoco.mj_step(m,d)
 tr=np.asarray(rows);np.savez(a.output/(method+'.npz'),trace=tr,columns=['t','x','y','z','roll','yaw','speed','steer','target','ctrl_steer','ctrl_rear','ctrl_hip','ctrl_knee'])
 results.append(dict(method=method,duration_s=float(d.time),physics_steps=tick,failed=bool(failed),peak_roll_deg=float(np.rad2deg(abs(tr[:,4])).max()),final_xy=tr[-1,1:3].tolist(),final_speed=float(tr[-1,6])))
(a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
