"""Replay a frozen STTW command with original collision meshes in installed Isaac Lab."""
import argparse,json,hashlib,shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=25);AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
app=AppLauncher(a).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Vt
from sttw_control.torch_controller import TorchVelocityServos
from sttw_control.torch_observation import TorchVehicleSignals
contract=json.loads(a.asset.with_name('contract.json').read_text());old=json.loads(a.receipt.read_text());raw=np.load(a.receipt.with_suffix('.npz'))
# USD prim identifiers cannot contain decimal points from the original OBJ basename.
# Stage identical mesh bytes with valid names; only the URDF filename references change.
staged=a.output/'input';staged.mkdir();tree=ET.parse(a.asset);mapping={}
for mesh_prim in tree.findall('.//mesh'):
    original=mesh_prim.get('filename')
    if original not in mapping:
        mapping[original]=f'collision_mesh_{len(mapping)}.obj';shutil.copy2(a.asset.parent/original,staged/mapping[original])
    mesh_prim.set('filename',mapping[original])
tree.write(staged/'vehicle.urdf',encoding='unicode')
converter=UrdfConverter(UrdfConverterCfg(asset_path=str(staged/'vehicle.urdf'),usd_dir=str(a.output/'usd'),usd_file_name='vehicle.usd',fix_base=False,merge_fixed_joints=False,make_instanceable=False,force_usd_conversion=True,self_collision=False,collider_type='convex_hull',joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.0002,device=a.device,gravity=(0.,0.,-9.81),physx=sim_utils.PhysxCfg(solver_type=1,bounce_threshold_velocity=.2)))
mesh=np.load(a.terrain/'terrain.npz');ground=UsdGeom.Mesh.Define(sim.stage,'/World/Terrain');ground.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh['vertices'].astype(np.float32)));ground.GetFaceVertexCountsAttr().Set([3]*len(mesh['triangles']));ground.GetFaceVertexIndicesAttr().Set(mesh['triangles'].flatten().astype(int).tolist());UsdPhysics.CollisionAPI.Apply(ground.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr('none');PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim()).CreateContactOffsetAttr(.01);PhysxSchema.PhysxCollisionAPI(ground.GetPrim()).CreateRestOffsetAttr(0.)
material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.,friction_combine_mode='average');material.func('/World/Material',material);sim_utils.bind_physics_material('/World/Terrain','/World/Material')
robot=Articulation(ArticulationCfg(prim_path='/World/Robot',spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path,rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.,angular_damping=0.,max_depenetration_velocity=100.,enable_gyroscopic_forces=True),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.01,rest_offset=0.),articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=4,solver_velocity_iteration_count=1)),init_state=ArticulationCfg.InitialStateCfg(pos=tuple(map(float,contract['root_translation'].split())),rot=tuple(contract['root_quat_wxyz']),joint_pos={'.*':0.},joint_vel={'.*':0.}),actuators={'effort':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=1000.,velocity_limit_sim=1000.)}))
for prim in list(sim.stage.Traverse()):
 if str(prim.GetPath()).startswith('/World/Robot') and prim.IsInstance():prim.SetInstanceable(False)
sim_utils.modify_collision_properties('/World/Robot',sim_utils.CollisionPropertiesCfg(contact_offset=.01,rest_offset=0.))
for prim in sim.stage.Traverse():
 if str(prim.GetPath()).startswith('/World/Robot') and prim.HasAPI(UsdPhysics.CollisionAPI):sim_utils.bind_physics_material(str(prim.GetPath()),'/World/Material')
sim.reset();robot.update(.0002);signals=TorchVehicleSignals(contract,robot.joint_names,a.device);servos=TorchVelocityServos(contract,robot.joint_names,a.device)
root=robot.data.default_root_state.clone();root[:,7]=old['drive_speed'];positions=robot.data.default_joint_pos.clone();velocities=torch.zeros_like(positions);velocities[:,[signals.front,signals.rear]]=-old['drive_speed']/.1;robot.write_root_pose_to_sim(root[:,:7]);robot.write_root_velocity_to_sim(root[:,7:]);robot.write_joint_state_to_sim(positions,velocities);robot.reset()
command=torch.tensor(raw['control_trace'][0,12:14][None],device=a.device);states=[];joints=[]
for _ in range(a.steps):
 torque=servos.torque(command,robot.data.joint_vel)+signals.passive_torque(robot.data.joint_pos,robot.data.joint_vel);robot.set_joint_effort_target(torque);robot.write_data_to_sim();sim.step(render=False);robot.update(.0002);state=robot.data.root_link_state_w.clone();state[:,3:7]=state[:,[4,5,6,3]];states.append(state.cpu().numpy());joints.append(torch.stack((robot.data.joint_pos,robot.data.joint_vel),dim=-1).cpu().numpy())
measurement,pose,speed=signals.observe(state,robot.data.joint_pos,robot.data.joint_vel)
result=dict(steps=a.steps,dt=.0002,device=a.device,source_asset=str(a.asset.resolve()),asset_sha256=hashlib.sha256(a.asset.read_bytes()).hexdigest(),terrain=str(a.terrain.resolve()),joints=robot.joint_names,bodies=robot.body_names,masses=robot.root_physx_view.get_masses().cpu().tolist(),materials=robot.root_physx_view.get_material_properties().cpu().tolist(),collision_attributes=[dict(path=str(prim.GetPath()),contact=prim.GetAttribute('physxCollision:contactOffset').Get(),rest=prim.GetAttribute('physxCollision:restOffset').Get()) for prim in sim.stage.Traverse() if str(prim.GetPath()).startswith('/World/Robot') and prim.HasAPI(UsdPhysics.CollisionAPI)],command=command.cpu().tolist(),measurement=measurement.cpu().tolist(),old_measurement=raw['control_trace'][1,1:8].tolist(),pose=pose.cpu().tolist(),speed=speed.cpu().tolist(),scope='installed Isaac Sim 5.1 contact diagnostic; original mesh retained, importer may recook; not closed-loop policy validation')
np.savez_compressed(a.output/'trace.npz',root=np.stack(states),joint=np.stack(joints));(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);app.close()
