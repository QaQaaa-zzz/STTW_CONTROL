"""GPU MuJoCo RB adaptation to the real trench and a settled STTW handoff.

Native actor/action clocks and actuator bounds stay fixed. Training terminates
at the handoff target; only the separate continuous-chain evaluator certifies
STTW/RB/STTW completion. Property randomization is fixed per parallel world.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx.warp.types import DATA_NON_VMAP
from mujoco_playground._src import mjx_env
from jit_dvgc.action_mapping import map_action
from jit_dvgc.geometry import build_geometry_contract, extract_geometry
from jit_dvgc.observation import ObservableGeometry, observable_frame


def load_delay_buffer(root):
    path=Path(root)/'source/isaaclab/isaaclab/utils/buffers'
    spec=importlib.util.spec_from_file_location('sttw_lab_buffers',path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.DelayBuffer


class PlaygroundRBEnv:
    def __init__(self,model,skill,cfg):
        self.device='cuda:0';self.num_envs=n=cfg['num_envs'];self.cfg=cfg
        self.num_obs=76;self.num_privileged_obs=106;self.num_actions=4
        batch=min(n,cfg.get('physics_batch_size',n))
        if n%batch:raise ValueError('Environment count must be divisible by physics batch size')
        self.slices=[slice(i,i+batch) for i in range(0,n,batch)]
        self.max_episode_length=round(cfg['episode_s']/.02)
        self.idx=idx=skill.bundle.model_index
        from jit_dvgc.model import _build_index
        self.idx=idx=_build_index(model)
        mm=mjx.put_model(model,impl='warp');self.model=model
        self.to_jax=lambda x:jax.dlpack.from_dlpack(x.detach().contiguous())
        self.to_torch=torch.utils.dlpack.from_dlpack
        self.rand=lambda shape:torch.rand(shape,device=self.device)
        def uniform(bounds,shape):return bounds[0]+(bounds[1]-bounds[0])*self.rand(shape)
        r=cfg['randomization'];self.randomization=r
        # Same Go2 ranges, expressed through MuJoCo model properties. Absolute
        # friction applies to both explicit pairs and implicit geom contacts.
        mass=uniform(r['mass_scale'],(n,))
        friction=uniform(r['friction_range'],(n,))
        gain=uniform(r['servo_gain_scale'],(n,model.nu))
        com=uniform(r['base_com_m'],(n,3))
        base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,r['base_body'])
        self.properties=tuple(self.to_jax(x) for x in (mass,friction,gain,com))
        def plant(mass,friction,gain,com):
            return mm.replace(body_mass=mm.body_mass*mass,body_inertia=mm.body_inertia*mass,
                body_subtreemass=mm.body_subtreemass*mass,body_invweight0=mm.body_invweight0/mass,
                dof_invweight0=mm.dof_invweight0/mass,body_ipos=mm.body_ipos.at[base].add(com),
                geom_friction=mm.geom_friction.at[:,0].set(friction),
                pair_friction=mm.pair_friction.at[:,:2].set(friction),
                actuator_gainprm=mm.actuator_gainprm*gain[:,None],actuator_biasprm=mm.actuator_biasprm*gain[:,None])
        geometry=build_geometry_contract(model)
        def features(d,last):
            g=extract_geometry(d,geometry)
            geom=ObservableGeometry(g.obstacle_relative_x,d.qpos[2],g.roll,g.pitch,g.yaw)
            frame=observable_frame(d,idx,geom,last,jp.asarray(1.))
            return frame,jp.stack((g.roll,g.pitch,g.yaw))
        self.features=jax.jit(jax.vmap(features))
        self.targets=jax.jit(jax.vmap(lambda action,knee:map_action(action,knee,skill.bundle.action_mapping)))
        self.brake_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,f'ground_brake_{j}') for j in ['hip_joint','knee_joint']])
        def make(q,v,mass,friction,gain,com):
            d=mjx_env.make_data(model,qpos=q,qvel=v,impl='warp',naconmax=64*batch,naccdmax=8*batch,njmax=128)
            d=d.replace(eq_active=d.eq_active.at[self.brake_ids].set(False))
            return mjx.forward(plant(mass,friction,gain,com),d)
        self.make=jax.jit(jax.vmap(make))
        jump=jax.vmap(lambda d,c,m,f,g,o:mjx_env.step(plant(m,f,g,o),d,c,cfg.get('physics_substeps',1)))
        home_opt=mm.opt.replace(timestep=jp.asarray(.001,dtype=mm.opt.timestep.dtype),integrator=type(mm.opt.integrator)(int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)))
        home_step=jax.vmap(lambda d,c,m,f,g,o:mjx_env.step(plant(m,f,g,o).replace(opt=home_opt),d,c,5))
        def reset_one(d,q,v,mask,m,f,g,o):
            candidate=d.replace(qpos=jp.where(mask,q,d.qpos),qvel=jp.where(mask,v,d.qvel),
                qacc_warmstart=jp.where(mask,0.,d.qacc_warmstart),ctrl=jp.where(mask,0.,d.ctrl),time=jp.where(mask,0.,d.time))
            return mjx.forward(plant(m,f,g,o),candidate)
        self.reset_data=jax.jit(jax.vmap(reset_one))
        terrain=jp.asarray(model.geom_type==mujoco.mjtGeom.mjGEOM_HFIELD)
        front=cfg['terrain']['front_x_m']+cfg['terrain']['width_m']
        @jax.jit
        def contacts(d):
            c=d._impl;geoms=c.contact__geom
            active=(jp.arange(geoms.shape[0])<c.nacon[0])&(c.contact__dist<=0)
            active &= jp.any(terrain[jp.clip(geoms,0,model.ngeom-1)]&(geoms>=0),-1)
            worlds=jp.clip(c.contact__worldid,0,batch-1)
            wheels=jp.stack([jp.any(geoms==wid,-1) for wid in (idx.frontwheel_geom_id,idx.rearwheel_geom_id)],axis=-1)
            supported=jp.zeros((batch,2),dtype=jp.int32).at[worlds].max((wheels&active[:,None]).astype(jp.int32)).astype(bool)
            far=jp.zeros((batch,2),dtype=jp.int32).at[worlds].max((wheels&active[:,None]&(c.contact__pos[:,0]>front)[:,None]).astype(jp.int32)).astype(bool)
            return supported,far
        self.contacts=contacts
        @jax.jit
        def advance(d,c,homing,m,f,g,o):
            def only_jump(_):
                out=jump(d,c,m,f,g,o);return out,*contacts(out)
            if cfg.get('shared_physics',False):
                return only_jump(None)
            def only_home(_):
                out=home_step(d,c,m,f,g,o);return out,*contacts(out)
            def mixed(_):
                rd=jump(d,c,m,f,g,o);gd=home_step(d,c,m,f,g,o)
                rs,rf=contacts(rd);gs,gf=contacts(gd)
                def choose(path,r,h):
                    if path[-1].name in DATA_NON_VMAP:return h
                    return jp.where(homing.reshape((batch,)+(1,)*(r.ndim-1)),h,r)
                out=jax.tree_util.tree_map_with_path(choose,rd,gd)
                return out,jp.where(homing[:,None],gs,rs),jp.where(homing[:,None],gf,rf)
            return jax.lax.cond(jp.all(homing),only_home,lambda _:jax.lax.cond(jp.any(homing),mixed,only_jump,None),None)
        self.advance=advance
        self.homing=torch.zeros(n,device=self.device,dtype=torch.bool)
        self.homing_events=torch.zeros((),device=self.device)
        self.far_wheels=torch.zeros(n,2,device=self.device,dtype=torch.bool)
        self.home_kp=torch.tensor(model.actuator_gainprm[2:,0],device=self.device,dtype=torch.float32)
        self.ctrl_lo=torch.tensor(model.actuator_ctrlrange[:,0],device=self.device,dtype=torch.float32)
        self.ctrl_hi=torch.tensor(model.actuator_ctrlrange[:,1],device=self.device,dtype=torch.float32)
        self.home=torch.tensor(model.key_qpos[idx.keyframe_id,[idx.hip_qpos_address,idx.knee_qpos_address]],device=self.device,dtype=torch.float32)
        self.joint_q=[idx.hip_qpos_address,idx.knee_qpos_address];self.joint_v=[idx.hip_dof_address,idx.knee_dof_address]
        self.initial_q=torch.tensor(np.load(cfg['entry_bank'])['qpos'],device=self.device,dtype=torch.float32)
        self.initial_v=torch.tensor(np.load(cfg['entry_bank'])['qvel'],device=self.device,dtype=torch.float32)
        DelayBuffer=load_delay_buffer(cfg['isaaclab_checkout'])
        self.delays={name:DelayBuffer(round(r['latency_s'][1]/.005),n,self.device) for name in ['action','sensor']}
        for b in self.delays.values():b.set_time_lag(torch.randint(round(r['latency_s'][0]/.005),round(r['latency_s'][1]/.005)+1,(n,),device=self.device))
        self.offset=uniform(r['joint_zero_offset_rad'],(n,3));self.noise=torch.tensor(r['frame_noise'],device=self.device)
        self.history=torch.zeros(n,3,25,device=self.device);self.last=torch.zeros(n,4,device=self.device)
        self.length=torch.zeros(n,device=self.device,dtype=torch.long);self.hold=torch.zeros(n,device=self.device)
        self.airborne=torch.zeros(n,device=self.device,dtype=torch.bool)
        self.stats=torch.zeros(4,device=self.device)
        self.capture_weights=torch.tensor([cfg['reward'].get(key,0.) for key in ('landing_lateral_speed_weight','landing_yaw_rate_weight','landing_steer_rate_weight')],device=self.device)
        self.capture_penalties=torch.zeros(3,device=self.device)
        q,v=self.sample_initial();q,v=self.to_jax(q),self.to_jax(v)
        self.data=[self.make(q[sl],v[sl],*[x[sl] for x in self.properties]) for sl in self.slices]
        self.obs=torch.zeros(n,76,device=self.device);self.critic=torch.zeros(n,106,device=self.device)
        self.update_observation()

    def field(self,name):
        return jp.concatenate([getattr(d,name) for d in self.data])

    def frames(self,action):
        values=[self.features(d,action[sl]) for d,sl in zip(self.data,self.slices)]
        return tuple(jp.concatenate([v[i] for v in values]) for i in range(2))

    def sample_initial(self):
        ids=torch.randint(len(self.initial_q),(self.num_envs,),device=self.device)
        q=self.initial_q[ids].clone();v=self.initial_v[ids].clone()
        q[:,1]+=(2*self.rand((self.num_envs,))-1)*.05
        v[:,0]+=(2*self.rand((self.num_envs,))-1)*.1
        return q,v

    def update_observation(self):
        q=self.to_torch(self.field('qpos'));v=self.to_torch(self.field('qvel'))
        _,angle=self.frames(self.to_jax(self.last));self.angle=self.to_torch(angle)
        signal=((q[:,0]>=9.5)&(q[:,0]<=11.)).float()
        self.obs=torch.cat((self.history.flatten(1),signal[:,None]),1)
        self.critic=torch.cat((self.obs,q,v,self.angle,v[:,:3],torch.zeros(self.num_envs,1,device=self.device)),1)

    def step(self,latent):
        action=torch.tanh(latent);q=self.to_torch(self.field('qpos'))
        target=self.to_torch(self.targets(self.to_jax(action),self.field('qpos')[:,self.idx.knee_qpos_address]))
        start_x=q[:,0].clone()
        for substep in range(4):
            push=(self.length>0)&(self.length%round(self.randomization['push_interval_s']/.02)==0)&(substep==0)
            velocity=self.to_torch(self.field('qvel')).clone()
            for axis,col in [('x',0),('y',1),('roll',3),('pitch',4),('yaw',5)]:
                lo,hi=self.randomization['push_velocity_range'][axis]
                velocity[:,col]+=push*(lo+(hi-lo)*self.rand((self.num_envs,)))
            requested=target.clone();observed_action=action.clone()
            if self.cfg.get('physical_homing',False):
                roll,pitch,yaw=self.angle.unbind(-1)
                body_speed=velocity[:,0]*torch.cos(yaw)+velocity[:,1]*torch.sin(yaw)
                gyro=self.to_torch(self.field('sensordata'))[:,self.idx.sensor_addresses[5]]
                enter=self.far_wheels.all(-1)&self.airborne&(roll.abs()<.15)&(pitch.abs()<.2)&(velocity[:,2].abs()<.5)&(gyro.abs()<.6)&(body_speed>=1.5)&~self.homing
                self.homing |= enter;self.homing_events+=enter.sum()
                bias=self.to_torch(self.field('qfrc_bias'))[:,self.joint_v]
                home_target=(self.home+bias/(self.home_kp*self.to_torch(self.properties[2])[:,2:])).clamp(self.ctrl_lo[2:],self.ctrl_hi[2:])
                requested[:,2:]=torch.where(self.homing[:,None],home_target,target[:,2:])
                delta=requested[:,2:]-self.home
                span=torch.where(delta>=0,self.ctrl_hi[2:]-self.home,self.home-self.ctrl_lo[2:]).clamp_min(1e-6)
                observed_action[:,2:]=torch.where(self.homing[:,None],(delta/span).clamp(-1,1),action[:,2:])
            velocity=self.to_jax(velocity);ctrl=self.to_jax(self.delays['action'].compute(requested));homing=self.to_jax(self.homing)
            result=[self.advance(d.replace(qvel=velocity[sl]),ctrl[sl],homing[sl],*[x[sl] for x in self.properties]) for d,sl in zip(self.data,self.slices)]
            self.data=[x[0] for x in result]
            supported,far=(jp.concatenate([x[i] for x in result]) for i in [1,2])
            self.far_wheels=self.to_torch(far)
            self.airborne |= ~self.to_torch(supported).any(-1)
            frame,angle=self.frames(self.to_jax(observed_action));self.angle=self.to_torch(angle)
            noisy=self.to_torch(frame).clone();noisy[:,9:12]+=self.offset
            noisy+=(2*torch.rand_like(noisy)-1)*self.noise
            delayed=self.delays['sensor'].compute(noisy)
        self.last=observed_action
        self.history=torch.cat((self.history[:,1:],delayed[:,None]),1)
        self.length+=1;self.update_observation()
        q=self.to_torch(self.field('qpos'));v=self.to_torch(self.field('qvel'));roll,pitch,yaw=self.angle.unbind(-1)
        grounded=self.to_torch(far).all(-1)&self.airborne
        pose_error=(q[:,self.joint_q]-self.home).square().sum(-1)
        joint_speed=v[:,self.joint_v].square().sum(-1)
        ready=grounded&(v[:,3:6].abs().amax(-1)<.3)&(roll.abs()<.08)&(pitch.abs()<.1)&(yaw.abs()<.15)&(v[:,1].abs()<.2)&(v[:,2].abs()<.2)&(v[:,0]>1.4)&(v[:,0]<2.2)&(pose_error<self.cfg['reward'].get('handoff_pose_norm_max_rad',.15)**2)&(joint_speed<self.cfg['reward'].get('handoff_joint_speed_norm_max_rad_s',.5)**2)
        self.hold=torch.where(ready,self.hold+.02,0.)
        success=self.hold>=.2
        failure=(roll.abs()>.7)|(pitch.abs()>1.309)|(q[:,2]<0)|(q[:,1].abs()>.6)|(q[:,0]>13.5)|~torch.isfinite(q).all(-1)
        timeout=self.length>=self.max_episode_length
        done=success|failure|timeout
        reward_cfg=self.cfg['reward']
        progress=(q[:,0]-start_x).clamp(-.1,.02*reward_cfg.get('progress_speed_cap_m_s',5.))
        reward=2*progress+.02*(torch.exp(-4*(v[:,0]-1.8).square())-2*roll.square()-pitch.square()-2*yaw.square()-2*q[:,1].square())
        reward-=.02*reward_cfg.get('speed_error_weight',0.)*(v[:,0]-1.8).square()
        reward+=grounded*.02*(2*torch.exp(-8*pose_error)-.1*joint_speed)
        capture_terms=torch.stack((v[:,1].square(),v[:,5].square(),v[:,self.idx.steering_dof_address].square()),-1)
        capture_penalty=grounded[:,None]*.02*capture_terms*self.capture_weights
        reward-=capture_penalty.sum(-1)
        self.capture_penalties+=capture_penalty.sum(0)
        reward+=success*10.-(failure|timeout)*5.
        reward=torch.nan_to_num(reward,nan=-5.,posinf=-5.,neginf=-5.)
        self.stats+=torch.stack((success.sum(),failure.sum(),timeout.sum(),done.sum()))
        # Reset all closed-loop history together; no host state/action transfer.
        initial_q,initial_v=self.sample_initial()
        initial_q,initial_v,reset=self.to_jax(initial_q),self.to_jax(initial_v),self.to_jax(done)
        self.data=[self.reset_data(d,initial_q[sl],initial_v[sl],reset[sl],*[x[sl] for x in self.properties]) for d,sl in zip(self.data,self.slices)]
        self.history=torch.where(done[:,None,None],0.,self.history);self.last=torch.where(done[:,None],0.,self.last)
        self.length=torch.where(done,0,self.length);self.hold=torch.where(done,0.,self.hold);self.airborne &= ~done
        self.homing &= ~done;self.far_wheels &= ~done[:,None]
        for buffer in self.delays.values():buffer.reset(done)
        self.update_observation()
        return self.obs,self.critic,reward,done,{'time_outs':torch.zeros_like(timeout)}
