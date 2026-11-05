"""Isaac Lab plant implementing the existing STTW GPU rollout interface."""
import json,hashlib,shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Vt
from isaacsim.core.cloner import Cloner,GridCloner
from .torch_controller import TorchVelocityServos
from .torch_observation import TorchVehicleSignals

class IsaacLabVectorPlant:
    def __init__(self,asset,terrain_paths,terrain_ids,output):
        asset=Path(asset);output=Path(output);self.contract=json.loads(asset.with_name('contract.json').read_text());self.device='cuda:0';self.count=len(terrain_ids)
        if hashlib.sha256(asset.read_bytes()).hexdigest()!=self.contract['exported_xml_sha256']:raise ValueError('asset identity mismatch')
        staged=output/'input';staged.mkdir();tree=ET.parse(asset);mapping={}
        for mesh in tree.findall('.//mesh'):
            original=mesh.get('filename')
            if original not in mapping:
                mapping[original]=f'collision_mesh_{len(mapping)}.obj';shutil.copy2(asset.parent/original,staged/mapping[original])
            mesh.set('filename',mapping[original])
        tree.write(staged/'vehicle.urdf',encoding='unicode')
        converter=UrdfConverter(UrdfConverterCfg(asset_path=str((staged/'vehicle.urdf').resolve()),usd_dir=str((output/'usd').resolve()),usd_file_name='vehicle.usd',fix_base=False,merge_fixed_joints=False,make_instanceable=False,force_usd_conversion=True,self_collision=False,collider_type='convex_hull',joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
        self.sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.0002,device=self.device,gravity=(0.,0.,-9.81),physx=sim_utils.PhysxCfg(solver_type=1,bounce_threshold_velocity=.2,gpu_found_lost_pairs_capacity=2**24,gpu_total_aggregate_pairs_capacity=2**24,gpu_found_lost_aggregate_pairs_capacity=2**27)))
        material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.,friction_combine_mode='average');material.func('/World/Material',material)
        cloner=GridCloner(spacing=100.,stage=self.sim.stage);cloner.define_base_env('/World/Envs')
        env_paths=cloner.generate_paths('/World/Envs/Env',self.count)
        origins=np.array([[20.*i,0.,0.] for i in range(self.count)],dtype=np.float32) if self.count<=6 else np.asarray(cloner.get_clone_transforms(self.count)[0],dtype=np.float32)
        self.origin=torch.as_tensor(origins,device=self.device);self.terrain_identity=[];terrain_sources={}
        sim_utils.create_prim(env_paths[0],'Xform')
        for i,terrain_id in enumerate(terrain_ids):
            ground_path=f'/World/Terrain_{i}'
            if terrain_id not in terrain_sources:
                path=Path(terrain_paths[terrain_id]);identity=json.loads((path/'identity.json').read_text())
                if hashlib.sha256((path/'terrain.npz').read_bytes()).hexdigest()!=identity['sha256']:raise ValueError('terrain identity mismatch')
                mesh=np.load(path/'terrain.npz');ground=UsdGeom.Mesh.Define(self.sim.stage,ground_path)
                ground.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh['vertices'].astype(np.float32)));ground.GetFaceVertexCountsAttr().Set([3]*len(mesh['triangles']));ground.GetFaceVertexIndicesAttr().Set(mesh['triangles'].flatten().astype(int).tolist());UsdPhysics.CollisionAPI.Apply(ground.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr('none');collision=PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim());collision.CreateContactOffsetAttr(.01);collision.CreateRestOffsetAttr(0.);sim_utils.bind_physics_material(ground_path,'/World/Material')
                terrain_sources[terrain_id]=(ground_path,identity)
            else:
                prim=self.sim.stage.DefinePrim(ground_path);prim.GetReferences().AddInternalReference(terrain_sources[terrain_id][0]);prim.SetInstanceable(True)
            xform=UsdGeom.Xformable(self.sim.stage.GetPrimAtPath(ground_path));xform.ClearXformOpOrder();xform.AddTranslateOp().Set(tuple(float(v) for v in origins[i]));self.terrain_identity.append(terrain_sources[terrain_id][1])
        self.robot=Articulation(ArticulationCfg(prim_path='/World/Envs/Env_.*/Robot',spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path,activate_contact_sensors=True,rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.,angular_damping=0.,max_depenetration_velocity=100.,enable_gyroscopic_forces=True),articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,solver_position_iteration_count=4,solver_velocity_iteration_count=1)),init_state=ArticulationCfg.InitialStateCfg(pos=tuple(map(float,self.contract['root_translation'].split())),rot=tuple(self.contract['root_quat_wxyz']),joint_pos={'.*':0.},joint_vel={'.*':0.}),actuators={'effort':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=1000.,velocity_limit_sim=1000.)}))
        for prim in list(self.sim.stage.Traverse()):
            if '/Robot/' in str(prim.GetPath()) and prim.IsInstance():prim.SetInstanceable(False)
        sim_utils.modify_collision_properties(f'{env_paths[0]}/Robot',sim_utils.CollisionPropertiesCfg(contact_offset=.01,rest_offset=0.))
        for prim in self.sim.stage.Traverse():
            if '/Robot/' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):sim_utils.bind_physics_material(str(prim.GetPath()),'/World/Material')
        Cloner.clone(cloner,source_prim_path=env_paths[0],prim_paths=env_paths,positions=origins,replicate_physics=True,copy_from_source=False,enable_env_ids=True)
        cloner.filter_collisions(self.sim.cfg.physics_prim_path,'/World/CollisionGroups',env_paths)
        for i in range(self.count):
            self.sim.stage.GetPrimAtPath(f'/World/CollisionGroups/group{i}').GetRelationship('collection:colliders:includes').AddTarget(f'/World/Terrain_{i}')
        self.sensor=ContactSensor(ContactSensorCfg(prim_path='/World/Envs/Env_.*/Robot/.*',update_period=0.,history_length=1));self.sim.reset();self.robot.update(.0002);self.sensor.update(.0002,force_recompute=True)
        self.joints=self.robot.joint_names;self.bodies=self.robot.body_names;self.signals=TorchVehicleSignals(self.contract,self.joints,self.device);self.servos=TorchVelocityServos(self.contract,self.joints,self.device);self.contact_order=[self.sensor.body_names.index(name) for name in self.bodies];self.nonwheel=[i for i,name in enumerate(self.bodies) if name not in ('frontwheel','rearwheel')]
        self.metadata=dict(device=self.device,num_envs=self.count,replicate_physics=True,terrain_asset_count=len(terrain_sources),masses=self.robot.root_physx_view.get_masses().cpu().tolist(),materials=self.robot.root_physx_view.get_material_properties().cpu().tolist(),contact_sensor_bodies=self.sensor.body_names,mesh_filename_mapping=mapping,scope='original meshes imported by Isaac Sim 5.1; contact representation may be recooked')
        self.reset()

    def refresh(self):
        self.roots=self.robot.data.root_link_state_w.clone();self.roots[:,3:7]=self.roots[:,[4,5,6,3]];self.dofs=torch.stack((self.robot.data.joint_pos,self.robot.data.joint_vel),dim=-1);self.rigid=self.robot.data.body_link_state_w.clone();self.rigid[:,:,3:7]=self.rigid[:,:,[4,5,6,3]];self.contact=self.sensor.data.net_forces_w[:,self.contact_order]

    def reset(self,speed=2.3):
        root=self.robot.data.default_root_state.clone();root[:,:3]+=self.origin;root[:,7]=speed;position=self.robot.data.default_joint_pos.clone();velocity=torch.zeros_like(position);velocity[:,[self.signals.front,self.signals.rear]]=-speed/.1;self.robot.write_root_pose_to_sim(root[:,:7]);self.robot.write_root_velocity_to_sim(root[:,7:]);self.robot.write_joint_state_to_sim(position,velocity);self.robot.reset();self.sensor.reset();self.refresh()

    def observe(self):
        measurement,pose,speed=self.signals.observe(self.roots,self.dofs[:,:,0],self.dofs[:,:,1]);return measurement,pose-self.origin,speed

    def step(self,command):
        torque=self.servos.torque(command,self.dofs[:,:,1])+self.signals.passive_torque(self.dofs[:,:,0],self.dofs[:,:,1]);self.robot.set_joint_effort_target(torque);self.robot.write_data_to_sim();self.sim.step(render=False);self.robot.update(.0002);self.sensor.update(.0002,force_recompute=True);self.refresh();return (self.contact[:,self.nonwheel].square().sum(-1)>0).any(-1)

    def close(self):
        self.sim.stop()
