"""Export the existing MuJoCo standard-case initial state; no physics stepping."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import mujoco
from sttw_control.model import load_model
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=json.loads(a.config.read_text());bank=pickle.load(open(s['initial_bank'],'rb'));i=s['initial_bank_index'];bundle=load_model();model=bundle.model;data=mujoco.MjData(model);data.qpos[:]=bank.data.qpos[i];data.qvel[:]=bank.data.qvel[i];mujoco.mj_forward(model,data)
bid=bundle.chassis;velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,velocity,0)
names=['rearsuspension_joint','steering_joint','rearwheel_joint','frontwheel_joint'];ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
state=dict(root_pose=np.r_[data.xpos[bid],data.xquat[bid]],root_velocity=np.r_[velocity[3:],velocity[:3]],joint_names=np.array(names),joint_position=data.qpos[model.jnt_qposadr[ids]],joint_velocity=data.qvel[model.jnt_dofadr[ids]],eso=bank.controller.eso[i],disturbance=bank.controller.disturbance[i],gains=bank.controller.gains[i],previous_command=bank.actuator.previous[i],previous_reference=bank.governor.current_reference[i],physical_tick=bank.physical_tick[i],raw=bank.raw[i])
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez(a.output,**state);print('exported',a.output,'physical_tick',state['physical_tick'])
