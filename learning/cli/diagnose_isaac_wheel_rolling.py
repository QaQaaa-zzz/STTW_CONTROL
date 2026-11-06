"""Isolate source wheel rolling from articulation and controller dynamics."""
import argparse,json,os,time,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--collision',choices=['convexHull','sdf','cylinder'],default='convexHull');p.add_argument('--external-forces-every-iteration',action='store_true');p.add_argument('--sdf-resolution',type=int,default=256);p.add_argument('--friction-correlation-distance',type=float,default=.025);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject,RigidObjectCfg
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Vt,Sdf
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cuda:0',physx=sim_utils.PhysxCfg(solver_type=1,bounce_threshold_velocity=.2,friction_correlation_distance=a.friction_correlation_distance,enable_external_forces_every_iteration=a.external_forces_every_iteration)))
material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.,friction_combine_mode='average');material.func('/World/Material',material)
plane=sim_utils.GroundPlaneCfg(physics_material=material);plane.func('/World/Ground',plane)
link=next(x for x in ET.parse(a.asset).findall('link') if x.get('name')=='rearwheel');mass=float(link.find('inertial/mass').get('value'));inertia=link.find('inertial/inertia');diag=[float(inertia.get(k)) for k in ('ixx','iyy','izz')];objects=[];names=['free_rolling','initial_spin']
for i,name in enumerate(names):
 path='/World/'+name;body=UsdGeom.Xform.Define(sim.stage,path).GetPrim();UsdPhysics.RigidBodyAPI.Apply(body);phys=PhysxSchema.PhysxRigidBodyAPI.Apply(body);phys.CreateLinearDampingAttr(0.);phys.CreateAngularDampingAttr(0.);phys.CreateSolverPositionIterationCountAttr(16);phys.CreateSolverVelocityIterationCountAttr(4);phys.CreateEnableGyroscopicForcesAttr(True)
 props=UsdPhysics.MassAPI.Apply(body);props.CreateMassAttr(mass);props.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag));props.CreateCenterOfMassAttr(Gf.Vec3f(0,0,0))
 if a.collision=='cylinder':
  wheel=UsdGeom.Cylinder.Define(sim.stage,path+'/collision');wheel.CreateRadiusAttr(.07);wheel.CreateHeightAttr(.05);wheel.CreateAxisAttr('Z');wheel.CreateExtentAttr([(-.07,-.07,-.025),(.07,.07,.025)])
  prim=wheel.GetPrim();UsdPhysics.CollisionAPI.Apply(prim);prim.CreateAttribute(PhysxSchema.Tokens.physxCollisionCustomGeometry,Sdf.ValueTypeNames.Bool).Set(True);cfg=PhysxSchema.PhysxCollisionAPI.Apply(prim);cfg.CreateContactOffsetAttr(.001);cfg.CreateRestOffsetAttr(0.);sim_utils.bind_physics_material(str(wheel.GetPath()),'/World/Material')
 for j,col in enumerate([] if a.collision=='cylinder' else link.findall('collision')):
  mesh_file=a.asset.parent/col.find('geometry/mesh').get('filename');vertices=[];faces=[]
  for line in mesh_file.read_text().splitlines():
   fields=line.split()
   if fields and fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
   elif fields and fields[0]=='f':faces.append([int(x.split('/')[0])-1 for x in fields[1:]])
  mesh=UsdGeom.Mesh.Define(sim.stage,f'{path}/collision_{j}');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices,dtype=np.float32)));mesh.CreateFaceVertexCountsAttr([len(x) for x in faces]);mesh.CreateFaceVertexIndicesAttr([x for face in faces for x in face]);UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(a.collision);cfg=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim());cfg.CreateContactOffsetAttr(.001);cfg.CreateRestOffsetAttr(0.);PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim()).CreateHullVertexLimitAttr(64);sim_utils.bind_physics_material(str(mesh.GetPath()),'/World/Material')
  if a.collision=='sdf':sim_utils.define_mesh_collision_properties(str(mesh.GetPath()),sim_utils.SDFMeshPropertiesCfg(sdf_resolution=a.sdf_resolution),stage=sim.stage)
 objects.append(RigidObject(RigidObjectCfg(prim_path=path,spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,i*2.,.071),rot=(math.sqrt(.5),-math.sqrt(.5),0.,0.),lin_vel=(2. if i==0 else 0.,0.,0.),ang_vel=(0.,2/.07,0.)))))
sim.reset()
for obj in objects:
 obj.write_root_state_to_sim(obj.data.default_root_state.clone());obj.reset()
rows=[];start=time.monotonic()
for k in range(201):
 rows.append(torch.stack([obj.data.root_link_state_w[0] for obj in objects]).cpu().numpy())
 if k==200:break
 sim.step(render=False)
 for obj in objects:obj.update(.005)
trace=np.asarray(rows);np.savez(a.output/'trajectory.npz',state=trace,time=np.arange(201)*.005,names=names,semantics='position wxyz quaternion linear/angular world velocity')
result=dict(status='complete',friction_correlation_distance=a.friction_correlation_distance,sdf_resolution=a.sdf_resolution,external_forces_every_iteration=a.external_forces_every_iteration,collision=a.collision,asset=str(a.asset),mass_kg=mass,inertia=diag,wheel_radius_m=.07,wheel_width_m=.05,colliders=1 if a.collision=='cylinder' else len(link.findall('collision')),physics_dt_s=.005,physics_steps=400,elapsed_s=time.monotonic()-start,initial_vx=trace[0,:,7].tolist(),final_vx=trace[-1,:,7].tolist(),initial_omega_y=trace[0,:,11].tolist(),final_omega_y=trace[-1,:,11].tolist(),scope='isolated wheel diagnostic, not bike validation; source inertia; cylinder option preserves outer diameter and width but changes tread profile; no motor or articulation')
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True);os.sync();os._exit(0)
