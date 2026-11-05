"""Persistent GPU path-control rollout with original 5000/200/50 Hz scheduling."""
import json
from pathlib import Path
import numpy as np
import torch
from .torch_controller import TorchController
from .torch_local_lower import TorchLocalLower
from .torch_path_policy import TorchPathPolicy
from .torch_path_reference import TorchPathReference,wrap
from .torch_middle_control import TorchMiddleControl
from .torch_path_reward import costs,failure_reward

class IsaacPathRollout:
    def __init__(self,bundle,settings,actor,plant=None,lower=None):
        bundle=Path(bundle);self.config=json.loads((bundle/'config.json').read_text());identity=json.loads((bundle/'identity.json').read_text());n=settings['num_envs'];self.n=n;self.device='cuda:0';self.horizon=round(settings['episode_s']/.02)
        terrain_count=len(settings['terrains']);terrain_ids=[i%terrain_count for i in range(n)];self.terrain=torch.tensor(terrain_ids,device=self.device);self.alpha=torch.tensor([settings['alphas'][(i//terrain_count)%len(settings['alphas'])] for i in range(n)],device=self.device);self.diagnostics=settings.get('diagnostics',False);self.zero_middle=torch.tensor(settings.get('zero_middle',[False]*n),device=self.device)
        if plant is None:
            from .isaac_vector_env import IsaacVectorPlant
            plant=IsaacVectorPlant(settings['asset'],list(settings['terrains'].values()),terrain_ids)
        self.plant=plant;self.cc=json.loads(Path(settings['controller_config']).read_text());self.controller=TorchController(self.cc,n,self.device)
        self.lower=lower if lower is not None else TorchLocalLower(settings['lower_actor'],settings['lower_sha256'],n,self.device);self.policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],self.config,n,self.device);self.policy.weights=actor.weights
        self.route=TorchPathReference(np.load(bundle/'route.npz'),self.config,self.device);self.speed=torch.full((n,),2.3,device=self.device);self.middle=TorchMiddleControl(self.policy,self.route,identity['frame_scales'],self.alpha,self.speed);self.scale=torch.tensor([1.5,10.],device=self.device);self.episode=0
        self.randomization=None;self.randomization_properties=None;self.push_events=0
        if settings.get('randomization'):
            from .isaaclab_randomization import randomize_vehicle_properties,VehicleControlRandomization
            self.randomization_properties=randomize_vehicle_properties(self.plant,settings['randomization'])
            self.randomization=VehicleControlRandomization(self.plant,settings['randomization'])
        self.initial_state=None;self.initial_physical_tick=0
        self.initial_alpha=self.alpha.clone();self.switch=settings.get('alpha_switch')
        if self.switch is not None:
            self.switch_target=torch.tensor(self.switch['target_alpha'],device=self.device)
            self.switch_mode=torch.tensor([{'independent':0,'common':1,'steer_priority':2}[mode] for mode in self.switch['execution_modes']],device=self.device)
        if settings.get('initial_state'):
            data=np.load(settings['initial_state']);order=[list(data['joint_names']).index(k) for k in self.plant.joints]
            self.initial_state={key:torch.as_tensor(data[key],device=self.device,dtype=torch.float32) for key in ('root_pose','root_velocity','eso','disturbance','gains','previous_reference','previous_command','raw')}
            for key in ('joint_position','joint_velocity'):self.initial_state[key]=torch.as_tensor(data[key][order],device=self.device,dtype=torch.float32)
            self.initial_physical_tick=int(data['physical_tick'])
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.controller.reset(slice(None));self.lower.reset(slice(None))
        if self.initial_state is None:
            self.plant.reset();self.previous_reference=torch.stack((self.speed,torch.zeros_like(self.speed)),dim=-1);self.previous_command=torch.stack((torch.zeros_like(self.speed),self.speed/.1),dim=-1);raw=self.previous_reference
        else:
            state=self.initial_state
            self.plant.reset_idx(torch.arange(self.n,device=self.device),state=state)
            for key in ('eso','disturbance','gains'):getattr(self.controller,key)[:]=state[key]
            self.previous_reference=state['previous_reference'].expand(self.n,-1).clone();self.previous_command=state['previous_command'].expand(self.n,-1).clone();raw=state['raw'].expand(self.n,-1)
        self.alive=torch.ones(self.n,device=self.device,dtype=torch.bool);self.tick=0
        self.alpha.copy_(self.initial_alpha);self.policy.execution_mode.zero_();self.switched=torch.zeros_like(self.alive);self.switch_tick=torch.full((self.n,),-1,device=self.device,dtype=torch.long)
        m,pose,v=self.plant.observe();self.measurement=m
        if self.randomization is not None:self.randomization.reset(torch.arange(self.n,device=self.device))
        self.middle.reset(m,pose,v,raw,self.previous_reference,self.previous_command,self.controller)

    @torch.no_grad()
    def update_preference(self):
        if self.switch is not None:
            event=self.alive&~self.switched&(self.middle.projection['progress']>=self.switch['progress_s_m'])
            self.alpha.copy_(torch.where(event,self.switch_target,self.alpha))
            self.policy.execution_mode.copy_(torch.where(event,self.switch_mode,self.policy.execution_mode))
            self.switch_tick=torch.where(event,self.tick*4,self.switch_tick);self.switched|=event

    @torch.no_grad()
    def observation(self):
        self.update_preference()
        _,pose,_=self.plant.observe();obs=self.middle.observation(pose)[0];obs=torch.nan_to_num(obs);clock=torch.full((self.n,1),1-self.tick/self.horizon,device=self.device);return obs,torch.cat((obs,clock),dim=-1)

    @torch.no_grad()
    def step(self,std):
        self.update_preference()
        valid=self.alive.clone();initial_offsets=self.policy.offsets.clone();reward=torch.zeros(self.n,device=self.device);new_failure=torch.zeros_like(valid);logs=[];details=[];first_action=None
        for lower_tick in range(4):
            if self.randomization is not None:
                physical_control_tick=self.tick*4+lower_tick;push_ticks=round(self.randomization.config['push_interval_s']/.005)
                if physical_control_tick>0 and physical_control_tick%push_ticks==0:
                    from types import SimpleNamespace
                    from isaaclab.envs.mdp.events import push_by_setting_velocity
                    ids=self.alive.nonzero(as_tuple=False).flatten()
                    push_by_setting_velocity(SimpleNamespace(scene={'robot':self.plant.robot}),ids,self.randomization.config['push_velocity_range']);self.plant.refresh();self.push_events+=len(ids)
            m,pose,v=self.plant.observe();governed=self.middle.before(pose,noise_std=std,zero_mask=self.zero_middle)
            if lower_tick==0:first_action=self.middle.latent.clone()
            policy_m=m if self.randomization is None else self.measurement
            action,_,finite=self.lower.prepare(policy_m,v,governed,self.previous_reference,self.previous_command,self.controller);applied=action*self.scale
            signals=self.plant.signals;position=self.plant.dofs[:,:,0];row=torch.stack((m[:,5]*.1,m[:,2],m[:,3],m[:,0],m[:,1],governed[:,1]),dim=-1);output=self.controller.step(row,torch.full_like(valid,(self.initial_physical_tick+self.tick*4+lower_tick)*.005>3.))
            steer=(output[:,0]+applied[:,0]).clamp(-3,3);steer=torch.maximum(torch.maximum(torch.full_like(steer,-3),(-.8-position[:,signals.steer])/.005),torch.minimum(torch.minimum(torch.full_like(steer,3),(.8-position[:,signals.steer])/.005),steer));command=torch.stack((steer,(governed[:,0]/.1+applied[:,1]).clamp(-60,60)),dim=-1)
            pre_failure=self.middle.fault|self.middle.domain_exit|~finite|~torch.isfinite(m).all(-1)|(m[:,0].abs()>.7);just_failed=self.alive&pre_failure;new_failure|=just_failed;self.alive&=~pre_failure;active_lower=self.alive.clone();peak=m[:,0].abs();endpoint=self.plant.roots.clone();stop_tick=torch.full((self.n,),25,device=self.device,dtype=torch.long)
            wheel_points=[];wheel_forces=[]
            for physics_tick in range(25):
                motor_command=torch.where(self.alive[:,None],torch.nan_to_num(command),torch.zeros_like(command))
                contact=self.plant.step(motor_command if self.randomization is None else self.randomization.command(motor_command))
                if self.randomization is not None:self.measurement=self.randomization.observe(self.plant.observe()[0])
                if self.diagnostics and getattr(self.plant,'wheel_sensors',[]):
                    wheel_points.append(torch.stack([s.data.contact_pos_w[:,0] for s in self.plant.wheel_sensors],dim=1).clone())
                    wheel_forces.append(torch.stack([s.data.force_matrix_w[:,0] for s in self.plant.wheel_sensors],dim=1).clone())
                q=self.plant.roots[:,3:7];x,y,z,w=q.unbind(-1);roll=-(torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))-torch.pi/2);peak=torch.where(self.alive,torch.maximum(peak,roll.abs()),peak)
                failure=contact|~torch.isfinite(self.plant.roots).all(-1);hit=self.alive&failure;stop_tick=torch.where(hit,physics_tick+1,stop_tick);endpoint=torch.where(self.alive[:,None],self.plant.roots,endpoint);new_failure|=hit;self.alive&=~failure
            post_m,post_pose,post_v=self.plant.signals.observe(endpoint,self.plant.dofs[:,:,0],self.plant.dofs[:,:,1]);post_pose=post_pose-self.plant.origin
            self.lower.finish(applied,wrap(post_pose[:,2]-pose[:,2])/.005);offsets=self.policy.offsets.clone();self.middle.after(post_m if self.randomization is None else self.measurement,post_pose,post_v,governed,command,applied,self.controller)
            projection=self.middle.projection;curvature=self.route.at(projection['progress'])[2];ratio=self.controller._system(self.speed)[3];chi=((torch.atan(self.cc['wheelbase']*curvature/np.cos(self.cc['caster']))/ratio).abs()-.18).div(.12).clamp(0,1)
            terms=costs(alpha=self.alpha,chi=chi,ev=post_v-self.speed,ey=projection['cross_track'],eh=wrap(projection['heading']-post_pose[:,2]),peak_roll=peak,roll_rate=post_m[:,1],speed=post_v,offsets=offsets,offset_rate=torch.zeros_like(offsets),config=self.config)
            reward+=torch.where(active_lower,torch.nan_to_num(terms['reward']),0);bad=(post_m[:,0].abs()>.7)|self.middle.fault|self.middle.domain_exit;new_failure|=self.alive&bad;self.alive&=~bad
            logs.append(torch.cat((post_pose[:15,:2],post_v[:15,None],projection['cross_track'][:15,None],peak[:15,None],terms['reward'][:15,None],active_lower[:15,None],stop_tick[:15,None],governed[:15],post_m[:15,5:6]*.1),dim=-1));self.previous_reference=governed;self.previous_command=command
            if self.diagnostics:
                details.append(dict(observation=self.middle.obs.clone(),nominal=self.middle.nominal.clone(),
                    progress=projection['progress'].clone(),pose=post_pose.clone(),measurement=post_m.clone(),policy_measurement=(post_m if self.randomization is None else self.measurement).clone(),
                    target=self.middle.flags['target'].clone(),offsets=self.policy.offsets.clone(),filtered=self.policy.filtered.clone(),
                    rate_clipped=self.middle.flags['rate_clipped'].clone(),alive=self.alive.clone(),
                    alpha=self.alpha.clone(),switch_tick=self.switch_tick.clone(),execution_mode=self.policy.execution_mode.clone(),channel_progress=self.middle.flags['channel_progress'].clone(),
                    contact_force=self.plant.contact.clone(),body_state=self.plant.rigid.clone(),
                    reward_normal=terms['reward'].clone(),**{f'{group}_{key}':value.clone() for group in ('raw','effective') for key,value in terms[group].items()}))
                if wheel_points:
                    details[-1].update(wheel_contact_points=torch.stack(wheel_points),wheel_contact_forces=torch.stack(wheel_forces))
        rate=(self.policy.offsets-initial_offsets)/.02;c=self.config['path_reward'];rate_cost=c['offset_rate_weight']*((rate/torch.tensor(c['offset_rate_scales'],device=self.device))**2).sum(-1);reward-=c['scale']*.02*rate_cost.clamp_max(c['component_caps']['offset_rate'])
        reward=torch.where(new_failure,failure_reward(torch.full_like(reward,self.horizon-self.tick),self.config),reward);reward=torch.where(valid,reward,0);self.tick+=1;horizon_done=self.tick==self.horizon;done=new_failure|torch.full_like(valid,horizon_done)
        result=dict(action=first_action,reward=reward,done=done,valid=valid,failure=new_failure,lower_trace=torch.stack(logs),episode=self.episode,episode_tick=self.tick)
        if self.diagnostics:result['diagnostics']=details
        if horizon_done:self.episode+=1;self.reset()
        return result
