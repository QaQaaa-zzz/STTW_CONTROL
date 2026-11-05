"""Persistent batched PhysX GPU plant; original STTW timing and velocity servos.

Vehicles share a terrain tile per type and use distinct collision groups. This
avoids replicating large terrain meshes per environment. Root positions carry
an explicit tile translation; policy poses subtract it once.
"""
import hashlib,json,math
from pathlib import Path
from isaacgym import gymapi,gymtorch
import numpy as np
import torch
from .torch_observation import TorchVehicleSignals
from .torch_controller import TorchVelocityServos

class IsaacVectorPlant:
    def __init__(self,asset_path,terrain_paths,terrain_ids,contact_offset=.01,rest_offset=0.):
        asset_path=Path(asset_path);self.contract=json.loads(asset_path.with_name('contract.json').read_text())
        if hashlib.sha256(asset_path.read_bytes()).hexdigest()!=self.contract['exported_xml_sha256']:raise ValueError('vehicle identity mismatch')
        self.device='cuda:0';self.count=len(terrain_ids);self.gym=gymapi.acquire_gym();params=gymapi.SimParams();params.dt=.0002;params.substeps=1;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.physx.use_gpu=True;params.use_gpu_pipeline=True;params.physx.contact_offset=contact_offset;params.physx.rest_offset=rest_offset;params.physx.num_subscenes=0;params.physx.max_gpu_contact_pairs=2**23;params.physx.default_buffer_size_multiplier=5
        self.sim=self.gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
        if self.sim is None:raise RuntimeError('GPU PhysX creation failed')
        self.terrain_identity=[];copies=max(1,math.ceil(self.count/(len(terrain_paths)*96)));columns=math.ceil(math.sqrt(len(terrain_paths)*copies))
        sites=[(20*(i%columns),20*(i//columns)) for i in range(len(terrain_paths)*copies)]
        lane_sites=[terrain+len(terrain_paths)*((i//len(terrain_paths))%copies) for i,terrain in enumerate(terrain_ids)]
        for i,path in enumerate(list(terrain_paths)*copies):
            path=Path(path);identity=json.loads((path/'identity.json').read_text());data=(path/'terrain.npz').read_bytes()
            if hashlib.sha256(data).hexdigest()!=identity['sha256']:raise ValueError('terrain identity mismatch')
            mesh=np.load(path/'terrain.npz');vertices=mesh['vertices'].astype(np.float32);triangles=mesh['triangles'].astype(np.uint32);p=gymapi.TriangleMeshParams();p.nb_vertices=len(vertices);p.nb_triangles=len(triangles);p.transform.p.x,p.transform.p.y=sites[i];p.static_friction=p.dynamic_friction=self.contract['ground_friction'][0]
            self.gym.add_triangle_mesh(self.sim,vertices.flatten(),triangles.flatten(),p);self.terrain_identity.append(identity)
        options=gymapi.AssetOptions();options.fix_base_link=False;options.collapse_fixed_joints=False;options.override_com=False;options.override_inertia=False
        asset=self.gym.load_asset(self.sim,str(asset_path.parent),asset_path.name,options);self.joints=self.gym.get_asset_dof_names(asset);self.bodies=self.gym.get_asset_rigid_body_names(asset)
        self.signals=TorchVehicleSignals(self.contract,self.joints,self.device);self.servos=TorchVelocityServos(self.contract,self.joints,self.device)
        materials={g['body']:g['friction'] for g in self.contract['vehicle_geoms']};initial_roots=[];initial_dofs=[];self.envs=[];self.actors=[]
        for i,terrain in enumerate(terrain_ids):
            env=self.gym.create_env(self.sim,gymapi.Vec3(0,0,0),gymapi.Vec3(0,0,0),1);pose=gymapi.Transform();xyz=list(map(float,self.contract['root_translation'].split()));xyz[0]+=sites[lane_sites[i]][0];xyz[1]+=sites[lane_sites[i]][1];pose.p=gymapi.Vec3(*xyz);w,x,y,z=self.contract['root_quat_wxyz'];pose.r=gymapi.Quat(x,y,z,w)
            actor=self.gym.create_actor(env,asset,pose,'sttw',i,1);properties=self.gym.get_actor_rigid_shape_properties(env,actor);ranges=self.gym.get_actor_rigid_body_shape_indices(env,actor)
            for name,span in zip(self.bodies,ranges):
                for k in range(span.start,span.start+span.count):properties[k].friction=materials[name][0];properties[k].contact_offset=contact_offset;properties[k].rest_offset=rest_offset
            self.gym.set_actor_rigid_shape_properties(env,actor,properties);dof=self.gym.get_actor_dof_properties(env,actor);dof['driveMode'][:]=gymapi.DOF_MODE_EFFORT;dof['stiffness'][:]=0;dof['damping'][:]=0;self.gym.set_actor_dof_properties(env,actor,dof)
            body=self.gym.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL).copy();joints=self.gym.get_actor_dof_states(env,actor,gymapi.STATE_ALL).copy();initial_roots.append(body.view(np.float32).reshape(-1,13)[0]);initial_dofs.append(joints.view(np.float32).reshape(-1,2));self.envs.append(env);self.actors.append(actor)
            if (i+1)%512==0:print('GPU actors created:',i+1,flush=True)
        self.gym.prepare_sim(self.sim)
        self.roots=gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim));self.dofs=gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.count,len(self.joints),2);self.rigid=gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.count,len(self.bodies),13);self.contact=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.count,len(self.bodies),3)
        if not all(x.is_cuda for x in (self.roots,self.dofs,self.rigid,self.contact)):raise RuntimeError('CPU tensor fallback is forbidden')
        self.initial_roots=torch.tensor(np.stack(initial_roots),device=self.device);self.initial_dofs=torch.tensor(np.stack(initial_dofs),device=self.device);self.origin=torch.zeros(self.count,3,device=self.device);self.origin[:,:2]=torch.tensor([sites[i] for i in lane_sites],device=self.device)
        self.nonwheel=[i for i,n in enumerate(self.bodies) if n not in ('frontwheel','rearwheel')];self.reset()

    def reset(self,speed=2.3):
        self.roots.copy_(self.initial_roots);self.roots[:,7]=speed;self.dofs.copy_(self.initial_dofs);self.dofs[:,[self.signals.front,self.signals.rear],1]=-speed/.1
        self.gym.set_actor_root_state_tensor(self.sim,gymtorch.unwrap_tensor(self.roots));self.gym.set_dof_state_tensor(self.sim,gymtorch.unwrap_tensor(self.dofs.view(-1,2)))

    def observe(self):
        measurement,pose,speed=self.signals.observe(self.roots,self.dofs[:,:,0],self.dofs[:,:,1]);return measurement,pose-self.origin,speed

    def step(self,command):
        torque=self.servos.torque(command,self.dofs[:,:,1])+self.signals.passive_torque(self.dofs[:,:,0],self.dofs[:,:,1]);self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(torque.contiguous()))
        self.gym.simulate(self.sim);self.gym.refresh_actor_root_state_tensor(self.sim);self.gym.refresh_dof_state_tensor(self.sim);self.gym.refresh_rigid_body_state_tensor(self.sim);self.gym.refresh_net_contact_force_tensor(self.sim)
        return (self.contact[:,self.nonwheel].square().sum(-1)>0).any(-1)

    def close(self):
        self.gym.destroy_sim(self.sim)
