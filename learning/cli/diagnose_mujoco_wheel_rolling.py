"""Analytic source-ellipsoid counterpart of the isolated Isaac wheel diagnostic."""
import argparse,json,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
source=mujoco.MjModel.from_xml_path(str(a.source));bid=source.body('rearwheel').id;gid=next(g for g in range(source.ngeom) if source.geom_bodyid[g]==bid and source.geom_type[g]==mujoco.mjtGeom.mjGEOM_ELLIPSOID)
values=lambda x:' '.join(format(float(v),'.17g') for v in x)
root=ET.Element('mujoco',model='isolated_rb_wheel');ET.SubElement(root,'option',timestep='.005',gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic');world=ET.SubElement(root,'worldbody');ET.SubElement(world,'geom',type='plane',size='0 0 .1',friction='.8 .005 .0001',condim='3')
names=['free_rolling','initial_spin']
for i,name in enumerate(names):
 body=ET.SubElement(world,'body',name=name,pos=f'0 {i*2} .071',quat='.7071067811865476 -.7071067811865476 0 0');ET.SubElement(body,'freejoint');ET.SubElement(body,'inertial',pos=values(source.body_ipos[bid]),quat=values(source.body_iquat[bid]),mass=str(source.body_mass[bid]),diaginertia=values(source.body_inertia[bid]));ET.SubElement(body,'geom',type='ellipsoid',size=values(source.geom_size[gid]),friction='.8 .005 .0001',condim='3',solref=values(source.geom_solref[gid]),solimp=values(source.geom_solimp[gid]))
xml=ET.tostring(root,encoding='unicode');(a.output/'model.xml').write_text(xml);m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
for i,name in enumerate(names):
 j=m.body(name).jntadr[0];adr=m.jnt_dofadr[j];d.qvel[adr]=2. if i==0 else 0.;d.qvel[adr+5]=2/.07
rows=[];start=time.monotonic()
for k in range(201):
 mujoco.mj_forward(m,d);states=[]
 for name in names:
  b=m.body(name).id;vel=np.empty(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,vel,0);states.append(np.r_[d.xpos[b],d.xquat[b],vel[3:],vel[:3]])
 rows.append(states)
 if k<200:mujoco.mj_step(m,d)
trace=np.asarray(rows);np.savez(a.output/'trajectory.npz',state=trace,time=np.arange(201)*.005,names=names,semantics='position wxyz quaternion linear/angular world velocity')
result=dict(status='complete',engine='MuJoCo',source=str(a.source),collision='analytic ellipsoid',physics_steps=400,elapsed_s=time.monotonic()-start,initial_vx=trace[0,:,7].tolist(),final_vx=trace[-1,:,7].tolist(),initial_omega_y=trace[0,:,11].tolist(),final_omega_y=trace[-1,:,11].tolist(),scope='same source wheel mass/inertia/semiaxes; analytic contact, implicitfast .005, condim3 .8 sliding friction; not engine-exact contact equivalence')
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(result)
