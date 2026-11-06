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
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Vt
from isaacsim.core.cloner import Cloner,GridCloner
from .torch_controller import TorchVelocityServos,TorchPositionServos
from .torch_observation import TorchVehicleSignals

class IsaacLabVectorPlant:
    def __init__(self,asset,terrain_paths,terrain_ids,output,servo_mode="explicit",hull_vertices=None,mesh_collision_on_geometry=False,compliant_wheels=None,friction_correlation_distance=.025,external_forces_every_iteration=False,wheel_collision="convexHull",friction_offset_threshold=.04,env_spacing=None,track_wheel_contacts=False,terrain_layout="external",pendulum_rest=None,pendulum_locked=False,physics_dt=.0002,gpu_pair_capacities=None,actuator_force_limits=None):
        self.servo_mode=servo_mode;self.physics_dt=physics_dt
        pair_capacities=gpu_pair_capacities or [2**24,2**24,2**27]
        asset=Path(asset);output=Path(output);self.contract=json.loads(asset.with_name('contract.json').read_text());self.device='cuda:0';self.count=len(terrain_ids)
        if hashlib.sha256(asset.read_bytes()).hexdigest()!=self.contract['exported_xml_sha256']:raise ValueError('asset identity mismatch')
        if actuator_force_limits:
            actuators={a['name']:a for a in self.contract['actuators']}
            if servo_mode!='explicit' or any(name not in actuators or not 0<float(cap)<=1000 for name,cap in actuator_force_limits.items()):
                raise ValueError('Explicit actuator force overrides require known names and caps in (0,1000] Nm')
            for name,cap in actuator_force_limits.items():actuators[name]['forcerange']=[-float(cap),float(cap)]
        self_collision=self.contract.get('requires_self_collision_filtering', False)
        self.pendulum_rest=pendulum_rest
        if pendulum_locked and pendulum_rest is None:raise ValueError("pendulum_locked requires pendulum_rest")
        self.pendulum_is_locked=False
        if pendulum_rest is not None and servo_mode != 'explicit':
            raise ValueError('Pendulum candidate uses explicit source actuators')
        staged=output/'input';staged.mkdir();tree=ET.parse(asset);mapping={}
        for mesh in tree.findall('.//mesh'):
            original=mesh.get('filename')
            if original not in mapping:
                mapping[original]=f'collision_mesh_{len(mapping)}.obj';shutil.copy2(asset.parent/original,staged/mapping[original])
            mesh.set('filename',mapping[original])
        tree.write(staged/'vehicle.urdf',encoding='unicode')
        converter=UrdfConverter(UrdfConverterCfg(asset_path=str((staged/'vehicle.urdf').resolve()),usd_dir=str((output/'usd').resolve()),usd_file_name='vehicle.usd',fix_base=False,merge_fixed_joints=False,make_instanceable=False,force_usd_conversion=True,self_collision=self_collision,collider_type='convex_hull',joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none',gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.))))
        self.sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.physics_dt,device=self.device,gravity=(0.,0.,-9.81),physx=sim_utils.PhysxCfg(solver_type=1,friction_offset_threshold=friction_offset_threshold,enable_external_forces_every_iteration=external_forces_every_iteration,bounce_threshold_velocity=.2,friction_correlation_distance=friction_correlation_distance,gpu_found_lost_pairs_capacity=pair_capacities[0],gpu_total_aggregate_pairs_capacity=pair_capacities[1],gpu_found_lost_aggregate_pairs_capacity=pair_capacities[2])))
        material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.,friction_combine_mode='average');material.func('/World/Material',material)
        cloner=GridCloner(spacing=100. if env_spacing is None else env_spacing,stage=self.sim.stage);cloner.define_base_env('/World/Envs')
        env_paths=cloner.generate_paths('/World/Envs/Env',self.count)
        origins=np.array([[20.*i,0.,0.] for i in range(self.count)],dtype=np.float32) if self.count<=6 and env_spacing is None else np.asarray(cloner.get_clone_transforms(self.count)[0],dtype=np.float32)
        clone_origins=origins.copy()
        replicated_terrain=terrain_layout=='replicated'
        terrain_offsets=np.zeros((len(terrain_paths),3),dtype=np.float32)
        if replicated_terrain:
            terrain_offsets[:,0]=np.arange(len(terrain_paths))*(env_spacing/len(terrain_paths))
            origins=origins+terrain_offsets[np.asarray(terrain_ids)]
        self.origin=torch.as_tensor(origins,device=self.device);self.terrain_identity=[];terrain_sources={}
        sim_utils.create_prim(env_paths[0],'Xform')
        plane_paths=[]
        if 'plane' in terrain_paths:
            plane=sim_utils.GroundPlaneCfg(physics_material=material);plane.func('/World/Ground',plane);plane_paths=['/World/Ground']
        for i,terrain_id in enumerate(range(len(terrain_paths)) if replicated_terrain else terrain_ids):
            if str(terrain_paths[terrain_id])=='plane':
                self.terrain_identity.append(dict(kind='plane',height_m=0.));continue
            ground_path=f'{env_paths[0]}/Terrain_{i}' if replicated_terrain else f'/World/Terrain_{i}'
            if terrain_id not in terrain_sources:
                path=Path(terrain_paths[terrain_id]);identity=json.loads((path/'identity.json').read_text())
                if hashlib.sha256((path/'terrain.npz').read_bytes()).hexdigest()!=identity['sha256']:raise ValueError('terrain identity mismatch')
                mesh=np.load(path/'terrain.npz');ground=UsdGeom.Mesh.Define(self.sim.stage,ground_path)
                ground.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh['vertices'].astype(np.float32)));ground.GetFaceVertexCountsAttr().Set([3]*len(mesh['triangles']));ground.GetFaceVertexIndicesAttr().Set(mesh['triangles'].flatten().astype(int).tolist());UsdPhysics.CollisionAPI.Apply(ground.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr('none');collision=PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim());collision.CreateContactOffsetAttr(.01);collision.CreateRestOffsetAttr(0.);sim_utils.bind_physics_material(ground_path,'/World/Material')
                terrain_sources[terrain_id]=(ground_path,identity)
            else:
                prim=self.sim.stage.DefinePrim(ground_path);prim.GetReferences().AddInternalReference(terrain_sources[terrain_id][0]);prim.SetInstanceable(True)
            position=terrain_offsets[terrain_id] if replicated_terrain else origins[i]
            xform=UsdGeom.Xformable(self.sim.stage.GetPrimAtPath(ground_path));xform.ClearXformOpOrder();xform.AddTranslateOp().Set(tuple(float(v) for v in position));self.terrain_identity.append(terrain_sources[terrain_id][1])
        if replicated_terrain:self.terrain_identity=[self.terrain_identity[terrain_id] for terrain_id in terrain_ids]
        actuators={'effort':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=0.,damping=0.,effort_limit_sim=1000.,velocity_limit_sim=1000.)}
        if servo_mode == 'implicit':
            actuators={
                'steering':ImplicitActuatorCfg(joint_names_expr=['steering_joint'],stiffness=0.,damping=20.,effort_limit_sim=1.,velocity_limit_sim=1000.),
                'rear':ImplicitActuatorCfg(joint_names_expr=['rearwheel_joint'],stiffness=0.,damping=3.,effort_limit_sim=3.,velocity_limit_sim=1000.),
                'passive':ImplicitActuatorCfg(joint_names_expr=['frontwheel_joint','rearsuspension_joint'],stiffness=0.,damping=0.,effort_limit_sim=1000.,velocity_limit_sim=1000.)}
        self.robot=Articulation(ArticulationCfg(prim_path='/World/Envs/Env_.*/Robot',spawn=sim_utils.UsdFileCfg(usd_path=converter.usd_path,activate_contact_sensors=True,rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.,angular_damping=0.,max_depenetration_velocity=100.,enable_gyroscopic_forces=True),articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=self_collision,solver_position_iteration_count=4,solver_velocity_iteration_count=1)),init_state=ArticulationCfg.InitialStateCfg(pos=tuple(map(float,self.contract['root_translation'].split())),rot=tuple(self.contract['root_quat_wxyz']),joint_pos=({'.*':0.} if pendulum_rest is None else {**{j['name']:0. for j in self.contract['joints']},'hip_joint':pendulum_rest[0],'knee_joint':pendulum_rest[1]}),joint_vel={'.*':0.}),actuators=actuators))
        for prim in list(self.sim.stage.Traverse()):
            if '/Robot/' in str(prim.GetPath()) and prim.IsInstance():prim.SetInstanceable(False)
        if mesh_collision_on_geometry or wheel_collision == "sdf":
            for prim in list(self.sim.stage.Traverse()):
                if '/Robot/' in str(prim.GetPath()) and prim.IsA(UsdGeom.Xform) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                    for child in Usd.PrimRange(prim):
                        if child.IsA(UsdGeom.Mesh):
                            UsdPhysics.CollisionAPI.Apply(child)
                            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(approximation)
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if wheel_collision == "sdf":
            for prim in self.sim.stage.Traverse():
                if '/Robot/' in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    sim_utils.define_mesh_collision_properties(str(prim.GetPath()),sim_utils.SDFMeshPropertiesCfg(sdf_resolution=256),stage=self.sim.stage)
        sim_utils.modify_collision_properties(f'{env_paths[0]}/Robot',sim_utils.CollisionPropertiesCfg(contact_offset=.01,rest_offset=0.))
        for prim in self.sim.stage.Traverse():
            if '/Robot/' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):sim_utils.bind_physics_material(str(prim.GetPath()),'/World/Material')
        if compliant_wheels:
            for wheel,parameters in compliant_wheels.items():
                path='/World/'+wheel+'_material'
                wheel_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.,friction_combine_mode='average',compliant_contact_stiffness=parameters['stiffness'],compliant_contact_damping=parameters['damping'])
                wheel_material.func(path,wheel_material)
                for prim in self.sim.stage.Traverse():
                    if '/Robot/'+wheel+'/' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                        sim_utils.bind_physics_material(str(prim.GetPath()),path)
        if self_collision:
            for left,right in self.contract['filtered_body_pairs']:
                prim=self.sim.stage.GetPrimAtPath(f'{env_paths[0]}/Robot/{left}')
                UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(f'{env_paths[0]}/Robot/{right}')
        self.collision_cooking=[]
        for prim in self.sim.stage.Traverse():
            if '/Robot/' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                hull=PhysxSchema.PhysxConvexHullCollisionAPI(prim)
                previous_limit=hull.GetHullVertexLimitAttr().Get()
                if hull_vertices is not None:
                    hull=PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
                    hull.CreateHullVertexLimitAttr(hull_vertices)
                self.collision_cooking.append(dict(path=str(prim.GetPath()),points=sum(len(UsdGeom.Mesh(child).GetPointsAttr().Get()) for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Mesh)),previous_hull_limit=previous_limit,hull_limit=hull.GetHullVertexLimitAttr().Get(),approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()))
        Cloner.clone(cloner,source_prim_path=env_paths[0],prim_paths=env_paths,positions=clone_origins,replicate_physics=True,copy_from_source=False,enable_env_ids=True)
        if not replicated_terrain:
            cloner.filter_collisions(self.sim.cfg.physics_prim_path,'/World/CollisionGroups',env_paths,global_paths=plane_paths)
            for i in range(self.count):
                if str(terrain_paths[terrain_ids[i]])=='plane':continue
                self.sim.stage.GetPrimAtPath(f'/World/CollisionGroups/group{i}').GetRelationship('collection:colliders:includes').AddTarget(f'/World/Terrain_{i}')
        self.sensor=ContactSensor(ContactSensorCfg(prim_path='/World/Envs/Env_.*/Robot/.*',update_period=0.,history_length=1))
        self.wheel_sensors=[]
        if track_wheel_contacts:
            terrain_filter='/World/Envs/Env_.*/Terrain_.*' if replicated_terrain else '/World/Terrain_.*'
            filters=plane_paths+([terrain_filter] if terrain_sources else [])
            self.wheel_sensors=[ContactSensor(ContactSensorCfg(prim_path=f'/World/Envs/Env_.*/Robot/{wheel}',filter_prim_paths_expr=filters,track_contact_points=True,max_contact_data_count_per_prim=16,update_period=0.)) for wheel in ('frontwheel','rearwheel')]
        self.sim.reset();self.robot.update(self.physics_dt);self.sensor.update(self.physics_dt,force_recompute=True)
        for sensor in self.wheel_sensors:sensor.update(self.physics_dt,force_recompute=True)
        self.joints=self.robot.joint_names;self.bodies=self.robot.body_names;self.signals=TorchVehicleSignals(self.contract,self.joints,self.device);self.servos=TorchVelocityServos(self.contract,self.joints,self.device,additional_actuators=(() if pendulum_rest is None else ("cmd_hip_f","cmd_knee_f")));self.contact_order=[self.sensor.body_names.index(name) for name in self.bodies];self.nonwheel=[i for i,name in enumerate(self.bodies) if name not in ('frontwheel','rearwheel')]
        self.pendulum_servos=None if pendulum_rest is None else TorchPositionServos(self.contract,self.joints,['cmd_hip_f','cmd_knee_f'],self.device)
        self.jump_steering_servo=None if pendulum_rest is None else TorchPositionServos(self.contract,self.joints,['cmd_steering_p'],self.device,allow_inactive=True)
        self.pendulum_target=None if pendulum_rest is None else torch.tensor(pendulum_rest,device=self.device).expand(self.count,-1).clone()
        self.metadata=dict(actuator_force_limits_Nm=actuator_force_limits, gpu_pair_capacities=pair_capacities,physics_dt_s=self.physics_dt,env_origins=origins.tolist(),env_spacing=env_spacing,friction_offset_threshold=PhysxSchema.PhysxSceneAPI(self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)).GetFrictionOffsetThresholdAttr().Get(),wheel_collision=wheel_collision,external_forces_every_iteration=PhysxSchema.PhysxSceneAPI(self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)).GetEnableExternalForcesEveryIterationAttr().Get(),friction_correlation_distance=PhysxSchema.PhysxSceneAPI(self.sim.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)).GetFrictionCorrelationDistanceAttr().Get(),compliant_wheels=compliant_wheels,mesh_collision_on_geometry=mesh_collision_on_geometry,collision_cooking=self.collision_cooking,servo_mode=servo_mode,device=self.device,num_envs=self.count,replicate_physics=True,terrain_asset_count=len(terrain_sources),masses=self.robot.root_physx_view.get_masses().cpu().tolist(),materials=self.robot.root_physx_view.get_material_properties().cpu().tolist(),contact_sensor_bodies=self.sensor.body_names,mesh_filename_mapping=mapping,scope='original meshes imported by Isaac Sim 5.1; contact representation may be recooked')
        self.metadata['self_collision']=dict(enabled=self_collision,filtered_body_pairs=self.contract.get('filtered_body_pairs',[]))
        self.metadata['terrain_layout']=dict(mode=terrain_layout,clone_origins=clone_origins.tolist(),template_offsets=terrain_offsets.tolist(),filtering='native replicated environment IDs' if replicated_terrain else 'explicit collision groups')
        self.metadata['wheel_contact_tracking']=dict(enabled=track_wheel_contacts,body_order=['frontwheel','rearwheel'],aggregation='official ContactSensor mean contact position per body/filter; force_matrix_w per filter',sample_dt_s=self.physics_dt)
        self.reset()
        if self.pendulum_servos is not None:
            self.pendulum_original_limits=self.robot.data.joint_pos_limits[:,self.pendulum_servos.indices].clone()
            self.pendulum_joint_prims=[prim for prim in self.sim.stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in ('hip_joint','knee_joint') and str(prim.GetPath()).startswith('/World/Envs/Env_')]
            self.metadata['pendulum_lock_model']='joint brake with 1e-4 rad half-play, clipped to native limits; not finite motor holding'
            self.set_pendulum_locked(pendulum_locked)

    def set_pendulum_locked(self,locked):
        """Engage at current joint positions or release to native limits; never reset state."""
        indices=self.pendulum_servos.indices
        self.pendulum_target=self.robot.data.joint_pos[:,indices].clone()
        limits=torch.stack((torch.maximum(self.pendulum_target-1e-4,self.pendulum_original_limits[:,:,0]),torch.minimum(self.pendulum_target+1e-4,self.pendulum_original_limits[:,:,1])),dim=-1) if locked else self.pendulum_original_limits
        # Tensor API rewrites every DOF, including unbounded continuous wheel joints.
        # USD notices update only these two joint limits, with native degree units.
        bounds=limits.cpu().numpy()
        for prim in self.pendulum_joint_prims:
            env_id=int(str(prim.GetPath()).split('/Env_')[1].split('/')[0])
            joint_id=0 if prim.GetName()=='hip_joint' else 1
            joint=UsdPhysics.RevoluteJoint(prim)
            joint.GetLowerLimitAttr().Set(float(np.rad2deg(bounds[env_id,joint_id,0])))
            joint.GetUpperLimitAttr().Set(float(np.rad2deg(bounds[env_id,joint_id,1])))
        self.robot.data.joint_pos_limits[:,indices]=limits
        self.pendulum_is_locked=bool(locked)
        self.metadata['pendulum_locked']=bool(locked)

    def refresh(self):
        self.roots=self.robot.data.root_link_state_w.clone();self.roots[:,3:7]=self.roots[:,[4,5,6,3]];self.dofs=torch.stack((self.robot.data.joint_pos,self.robot.data.joint_vel),dim=-1);self.contact=self.sensor.data.net_forces_w[:,self.contact_order]

    @property
    def rigid(self):
        # Full body poses are diagnostics; Lab refreshes its lazy tensor on access.
        state=self.robot.data.body_link_state_w.clone()
        state[:,:,3:7]=state[:,:,[4,5,6,3]]
        return state

    def reset(self,speed=2.3):
        self.reset_idx(torch.arange(self.count,device=self.device),speed=speed)

    def reset_idx(self,env_ids,speed=2.3,state=None):
        """Reset selected lanes without advancing or resetting their neighbours.

        Optional state is the exported MuJoCo bank state, with tensors already
        on this plant's device. Controller/history reset belongs to the task.
        """
        count=len(env_ids)
        if state is None:
            root=self.robot.data.default_root_state[env_ids].clone()
            root[:,7]=speed
            position=self.robot.data.default_joint_pos[env_ids].clone()
            velocity=torch.zeros_like(position)
            velocity[:,[self.signals.front,self.signals.rear]]=-speed/.1
        else:
            root=torch.cat((state['root_pose'],state['root_velocity']),dim=-1).expand(count,-1).clone()
            position=state['joint_position'].expand(count,-1).clone()
            velocity=state['joint_velocity'].expand(count,-1).clone()
        root[:,:3]+=self.origin[env_ids]
        self.robot.write_root_pose_to_sim(root[:,:7],env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(root[:,7:],env_ids=env_ids)
        self.robot.write_joint_state_to_sim(position,velocity,env_ids=env_ids)
        self.robot.reset(env_ids);self.sensor.reset(env_ids)
        for sensor in self.wheel_sensors:sensor.reset(env_ids)
        if self.pendulum_target is not None:self.pendulum_target[env_ids]=position[:,self.pendulum_servos.indices]
        self.refresh()

    def observe(self):
        measurement,pose,speed=self.signals.observe(self.roots,self.dofs[:,:,0],self.dofs[:,:,1]);return measurement,pose-self.origin,speed

    def step(self,command,pendulum_target=None,steering_position_target=None,refresh_diagnostics=True):
        torque=self.signals.passive_torque(self.dofs[:,:,0],self.dofs[:,:,1])
        if self.servo_mode == 'implicit':
            target=torch.maximum(self.servos.control[:,0],torch.minimum(self.servos.control[:,1],command*self.servos.sign))
            self.robot.set_joint_velocity_target(target,joint_ids=self.servos.indices)
        else:
            motor=self.servos.torque(command,self.dofs[:,:,1])
            if steering_position_target is not None:
                motor[:,self.signals.steer]=0.
                motor+=self.jump_steering_servo.torque(steering_position_target,self.dofs[:,:,0],self.dofs[:,:,1])
            torque=torque+motor
        if self.pendulum_servos is not None and not self.pendulum_is_locked:
            if pendulum_target is not None:self.pendulum_target=pendulum_target
            self.last_pendulum_torque=self.pendulum_servos.torque(self.pendulum_target,self.dofs[:,:,0],self.dofs[:,:,1])
            torque=torque+self.last_pendulum_torque
        self.robot.set_joint_effort_target(torque);self.robot.write_data_to_sim();self.sim.step(render=False);self.robot.update(self.physics_dt)
        if not refresh_diagnostics:
            self.dofs=torch.stack((self.robot.data.joint_pos,self.robot.data.joint_vel),dim=-1)
            return None
        self.sensor.update(self.physics_dt,force_recompute=True)
        for sensor in self.wheel_sensors:sensor.update(self.physics_dt,force_recompute=True)
        self.refresh();return (self.contact[:,self.nonwheel].square().sum(-1)>0).any(-1)

    def close(self):
        self.sim.stop()
