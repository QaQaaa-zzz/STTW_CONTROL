"""Estimate local compliant-contact parameters from original MuJoCo normal forces.

This is a local force/depth and force/velocity approximation, not an exact
translation of MuJoCo solref/solimp into a linear contact model. No physics steps.
"""
import argparse,json,pickle,numpy as np,mujoco
from pathlib import Path
from sttw_control.model import load_model
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b=load_model();m=b.model;cfg=json.loads(a.config.read_text());bank=pickle.load(open(cfg['initial_bank'],'rb'));index=cfg['initial_bank_index'];out=[]
aid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,k) for k in ['cmd_steering_v','cmd_rearwheel_v']]
for dz,dvz in [(0,0),(-.0001,0),(.0001,0),(0,-.01),(0,.01)]:
 d=mujoco.MjData(m);d.qpos[:]=bank.data.qpos[index];d.qvel[:]=bank.data.qvel[index];d.qpos[2]+=dz;d.qvel[2]+=dvz;d.ctrl[aid]=[0,-23];mujoco.mj_forward(m,d);c=[]
 for j in range(d.ncon):
  v=d.contact[j];f=np.zeros(6);mujoco.mj_contactForce(m,d,j,f);names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[k])) for k in [v.geom1,v.geom2]];c.append(dict(bodies=names,dist=float(v.dist),force=f.tolist()))
 out.append(dict(dz=dz,dvz=dvz,contacts=c))
ks=[c['force'][0]/(-c['dist']) for c in out[0]['contacts']];cs=[(a['force'][0]-b['force'][0])/.02 for a,b in zip(out[3]['contacts'],out[4]['contacts'])]
r=dict(model_identity=b.identity,config=str(a.config),initial_bank_index=index,command=[0,23],samples=out,equivalent_stiffness=ks,velocity_derivative_damping=cs,approximation='local secant stiffness and force-velocity derivative; not exact MuJoCo contact equation')
a.output.write_text(json.dumps(r,indent=2));print(json.dumps(dict(k=ks,c=cs)))
