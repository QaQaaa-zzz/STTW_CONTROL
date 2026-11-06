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
        terrain_count=len(settings['terrains']);terrain_ids=settings.get('lane_terrain_ids',[i%terrain_count for i in range(n)]);self.terrain=torch.tensor(terrain_ids,device=self.device);self.alpha=torch.tensor(settings.get('lane_alphas',[settings['alphas'][(i//terrain_count)%len(settings['alphas'])] for i in range(n)]),device=self.device);self.diagnostics=settings.get('diagnostics',False);self.trace_lanes=settings.get('trace_lanes',min(n,15));self.zero_middle=torch.tensor(settings.get('zero_middle',[False]*n),device=self.device)
        if plant is None:
            from .isaac_vector_env import IsaacVectorPlant
            plant=IsaacVectorPlant(settings['asset'],list(settings['terrains'].values()),terrain_ids)
        self.zero_lower=torch.tensor(settings.get('zero_lower',[False]*n),device=self.device)
        self.upper_sha256=identity['files']['actor.npz'];self.lower_sha256=settings['lower_sha256']
        self.plant=plant;self.wheel_radius_m=getattr(plant,'wheel_radius_m',.1);self.cc=json.loads(Path(settings['controller_config']).read_text());self.cc['wheel_radius_m']=self.wheel_radius_m;self.controller=TorchController(self.cc,n,self.device)
        self.lower=lower if lower is not None else TorchLocalLower(settings['lower_actor'],settings['lower_sha256'],n,self.device);self.policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],self.config,n,self.device);self.policy.weights=actor.weights
        self.physics_dt=getattr(self.plant,'physics_dt',.0002);self.physics_substeps=round(.005/self.physics_dt)
        if self.physics_substeps<1 or abs(self.physics_substeps*self.physics_dt-.005)>1e-9:raise ValueError('physics step must divide STTW .005 s control period')
        self.route=TorchPathReference(np.load(bundle/'route.npz'),self.config,self.device);self.speed=torch.as_tensor(settings["lane_speeds"],device=self.device,dtype=torch.float32) if "lane_speeds" in settings else torch.full((n,),float(settings.get("user_speed_m_s",2.3)),device=self.device);self.middle=TorchMiddleControl(self.policy,self.route,identity['frame_scales'],self.alpha,self.speed);self.scale=torch.tensor([1.5,10.],device=self.device);self.episode=0
        self.randomization=None;self.randomization_properties=None;self.push_events=0
        if settings.get('randomization'):
            from .isaaclab_randomization import randomize_vehicle_properties,VehicleControlRandomization
            self.randomization_properties=randomize_vehicle_properties(self.plant,settings['randomization'])
            self.randomization=VehicleControlRandomization(self.plant,settings['randomization'])
        self.initial_state=None;self.initial_physical_tick=0
        self.auto_reset_horizon=settings.get('auto_reset_horizon',True)
        self.training_pulse=None
        self.initial_alpha=self.alpha.clone();self.switch=settings.get('alpha_switch')
        self.preference_route=self.route if self.switch and self.switch.get('progress_frame')=='original_route' else None
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
            self.plant.reset();self.previous_reference=torch.stack((self.speed,torch.zeros_like(self.speed)),dim=-1);self.previous_command=torch.stack((torch.zeros_like(self.speed),self.speed/self.wheel_radius_m),dim=-1);raw=self.previous_reference
        else:
            state=self.initial_state
            self.plant.reset_idx(torch.arange(self.n,device=self.device),state=state)
            for key in ('eso','disturbance','gains'):getattr(self.controller,key)[:]=state[key]
            self.previous_reference=state['previous_reference'].expand(self.n,-1).clone();self.previous_command=state['previous_command'].expand(self.n,-1).clone();raw=state['raw'].expand(self.n,-1)
        self.alive=torch.ones(self.n,device=self.device,dtype=torch.bool);self.tick=0
        self.task_tick=torch.zeros(self.n,device=self.device,dtype=torch.long);self.physical_tick=torch.full_like(self.task_tick,self.initial_physical_tick);self.since_reset_tick=torch.zeros_like(self.task_tick)
        self.alpha.copy_(self.initial_alpha);self.policy.execution_mode.zero_();self.switched=torch.zeros_like(self.alive);self.switch_tick=torch.full((self.n,),-1,device=self.device,dtype=torch.long)
        m,pose,v=self.plant.observe();self.measurement=m
        if self.randomization is not None:self.randomization.reset(torch.arange(self.n,device=self.device))
        self.middle.reset(m,pose,v,raw,self.previous_reference,self.previous_command,self.controller)
        self.preference_progress=self.middle.projection['progress'].clone()

    @torch.no_grad()
    def update_preference(self):
        if self.switch is not None:
            progress=self.middle.projection['progress']
            if self.preference_route is not None:
                _,pose,speed=self.plant.observe()
                self.preference_progress=self.preference_route.project(pose[:,:2],self.preference_progress,speed.abs()*.02)['progress']
                progress=self.preference_progress
            event=self.alive&~self.switched&(progress>=self.switch['progress_s_m'])
            self.alpha.copy_(torch.where(event,self.switch_target,self.alpha))
            self.policy.execution_mode.copy_(torch.where(event,self.switch_mode,self.policy.execution_mode))
            self.switch_tick=torch.where(event,self.tick*4,self.switch_tick);self.switched|=event

    @torch.no_grad()
    def observation(self):
        self.update_preference()
        _,pose,_=self.plant.observe();obs=self.middle.observation(pose)[0];obs=torch.nan_to_num(obs);clock=1-self.task_tick[:,None]/(self.horizon*4);return obs,torch.cat((obs,clock),dim=-1)

    @torch.no_grad()
    def step(self,std=None,latent=None):
        self.update_preference()
        valid=self.alive.clone();start_task_tick=self.task_tick.clone();initial_offsets=self.policy.offsets.clone();reward=torch.zeros(self.n,device=self.device);new_failure=torch.zeros_like(valid);logs=[];details=[];first_action=None
        for lower_tick in range(4):
            if self.randomization is not None:
                push_ticks=round(self.randomization.config['push_interval_s']/.005)
                ids=(self.alive&(self.since_reset_tick>0)&(self.since_reset_tick%push_ticks==0)).nonzero(as_tuple=False).flatten()
                if len(ids):
                    from types import SimpleNamespace
                    from isaaclab.envs.mdp.events import push_by_setting_velocity
                    push_by_setting_velocity(SimpleNamespace(scene={'robot':self.plant.robot}),ids,self.randomization.config['push_velocity_range']);self.plant.refresh();self.push_events+=len(ids)
            m,pose,v=self.plant.observe();governed=self.middle.before(pose,noise_std=std,zero_mask=self.zero_middle,latent_override=latent)
            if lower_tick==0:first_action=self.middle.latent.clone()
            policy_m=m if self.randomization is None else self.measurement
            action,_,finite=self.lower.prepare(policy_m,v,governed,self.previous_reference,self.previous_command,self.controller);applied=torch.where(self.zero_lower[:,None],0.,action*self.scale)
            signals=self.plant.signals;position=self.plant.dofs[:,:,0];row=torch.stack((m[:,5]*self.wheel_radius_m,m[:,2],m[:,3],m[:,0],m[:,1],governed[:,1]),dim=-1);output=self.controller.step(row,self.physical_tick*.005>3.)
            steer=(output[:,0]+applied[:,0]).clamp(-3,3);steer=torch.maximum(torch.maximum(torch.full_like(steer,-3),(-.8-position[:,signals.steer])/.005),torch.minimum(torch.minimum(torch.full_like(steer,3),(.8-position[:,signals.steer])/.005),steer));command=torch.stack((steer,(governed[:,0]/self.wheel_radius_m+applied[:,1]).clamp(-60,60)),dim=-1)
            pre_failure=self.middle.fault|self.middle.domain_exit|~finite|~torch.isfinite(m).all(-1)|(m[:,0].abs()>.7);just_failed=self.alive&pre_failure;new_failure|=just_failed;self.alive&=~pre_failure;active_lower=self.alive.clone();peak=m[:,0].abs();endpoint=self.plant.roots.clone();stop_tick=torch.full((self.n,),self.physics_substeps,device=self.device,dtype=torch.long)
            wheel_points=[];wheel_forces=[]
            for physics_tick in range(self.physics_substeps):
                if self.training_pulse is not None:self.training_pulse(physics_tick)
                motor_command=torch.where(self.alive[:,None],torch.nan_to_num(command),torch.zeros_like(command))
                contact=self.plant.step(motor_command if self.randomization is None else self.randomization.command(motor_command))
                if self.randomization is not None:self.measurement=self.randomization.observe(self.plant.observe()[0])
                if self.diagnostics and getattr(self.plant,'wheel_sensors',[]):
                    wheel_points.append(torch.stack([s.data.contact_pos_w[:,0] for s in self.plant.wheel_sensors],dim=1).clone())
                    wheel_forces.append(torch.stack([s.data.force_matrix_w[:,0] for s in self.plant.wheel_sensors],dim=1).clone())
                q=self.plant.roots[:,3:7];x,y,z,w=q.unbind(-1);roll=-(torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))-torch.pi/2);peak=torch.where(self.alive,torch.maximum(peak,roll.abs()),peak)
                failure=contact|~torch.isfinite(self.plant.roots).all(-1);hit=self.alive&failure;stop_tick=torch.where(hit,physics_tick+1,stop_tick);endpoint=torch.where(self.alive[:,None],self.plant.roots,endpoint);new_failure|=hit;self.alive&=~failure
            post_m,post_pose,post_v=self.plant.signals.observe(endpoint,self.plant.dofs[:,:,0],self.plant.dofs[:,:,1]);post_pose=post_pose-self.plant.origin
            self.physical_tick+=1;self.task_tick+=1;self.since_reset_tick+=1
            self.lower.finish(applied,wrap(post_pose[:,2]-pose[:,2])/.005);offsets=self.policy.offsets.clone();self.middle.after(post_m if self.randomization is None else self.measurement,post_pose,post_v,governed,command,applied,self.controller)
            projection=self.middle.projection;curvature=self.route.at(projection['progress'])[2];ratio=self.controller._system(self.speed)[3];chi=((torch.atan(self.cc['wheelbase']*curvature/np.cos(self.cc['caster']))/ratio).abs()-.18).div(.12).clamp(0,1)
            terms=costs(alpha=self.alpha,chi=chi,ev=post_v-self.speed,ey=projection['cross_track'],eh=wrap(projection['heading']-post_pose[:,2]),peak_roll=peak,roll_rate=post_m[:,1],speed=post_v,offsets=offsets,offset_rate=torch.zeros_like(offsets),config=self.config)
            reward+=torch.where(active_lower,torch.nan_to_num(terms['reward']),0);bad=(post_m[:,0].abs()>.7)|self.middle.fault|self.middle.domain_exit;new_failure|=self.alive&bad;self.alive&=~bad
            logs.append(torch.cat((post_pose[:self.trace_lanes,:2],post_v[:self.trace_lanes,None],projection['cross_track'][:self.trace_lanes,None],peak[:self.trace_lanes,None],terms['reward'][:self.trace_lanes,None],active_lower[:self.trace_lanes,None],stop_tick[:self.trace_lanes,None],governed[:self.trace_lanes],post_m[:self.trace_lanes,5:6]*self.wheel_radius_m),dim=-1));self.previous_reference=governed;self.previous_command=command
            if self.diagnostics:
                details.append(dict(observation=self.middle.obs.clone(),nominal=self.middle.nominal.clone(),
                    progress=projection['progress'].clone(),pose=post_pose.clone(),measurement=post_m.clone(),policy_measurement=(post_m if self.randomization is None else self.measurement).clone(),
                    target=self.middle.flags['target'].clone(),offsets=self.policy.offsets.clone(),filtered=self.policy.filtered.clone(),
                    held_channels=self.middle.flags['held_channels'].clone(),rate_clipped=self.middle.flags['rate_clipped'].clone(),alive=self.alive.clone(),
                    domain_exit=self.middle.domain_exit.clone(),policy_fault=self.middle.fault.clone(),alpha=self.alpha.clone(),switch_tick=self.switch_tick.clone(),execution_mode=self.policy.execution_mode.clone(),channel_progress=self.middle.flags['channel_progress'].clone(),
                    contact_force=self.plant.contact.clone(),body_state=self.plant.rigid.clone(),
                    reward_normal=terms['reward'].clone(),**{f'{group}_{key}':value.clone() for group in ('raw','effective') for key,value in terms[group].items()}))
                if wheel_points:
                    details[-1].update(wheel_contact_points=torch.stack(wheel_points),wheel_contact_forces=torch.stack(wheel_forces))
        rate=(self.policy.offsets-initial_offsets)/.02;c=self.config['path_reward'];rate_cost=c['offset_rate_weight']*((rate/torch.tensor(c['offset_rate_scales'],device=self.device))**2).sum(-1);reward-=c['scale']*.02*rate_cost.clamp_max(c['component_caps']['offset_rate'])
        reward=torch.where(new_failure,failure_reward(self.horizon-start_task_tick//4,self.config),reward);reward=torch.where(valid,reward,0);self.tick+=1;horizon_done=self.task_tick>=self.horizon*4;done=new_failure|horizon_done
        result=dict(action=first_action,reward=reward,done=done,valid=valid,failure=new_failure,lower_trace=torch.stack(logs),episode=self.episode,episode_tick=self.tick,tracking_error=torch.stack((post_v-self.speed,projection['cross_track'],post_m[:,0]),dim=-1))
        if self.diagnostics:result['diagnostics']=details
        if self.auto_reset_horizon and self.tick==self.horizon:self.episode+=1;self.reset()
        return result


    @torch.no_grad()
    def capture_connection_state(self,lane):
        """Capture a nominal original-STTW exit at a 50 Hz boundary.

        Includes exposed physical and controller state, not hidden PhysX solver
        caches. Source terrain/model and target scene must be matched by caller.
        The six-connection experiment does not use randomized training snapshots.
        """
        if self.randomization is not None or self.switch is not None or self.plant.pendulum_rest is not None or self.middle.tick%4:
            raise ValueError('Connection capture requires nominal original STTW at a policy boundary')
        groups={
            'controller':(self.controller,('eso','disturbance','gains')),
            'lower':(self.lower,('frames','mask','previous_applied','yaw_rate')),
            'policy':(self.policy,('filtered','offsets','execution_mode','hold_channels')),
            'middle':(self.middle,('frames','mask','pose','yaw_rate','nominal','rates','fault','domain_exit')),
            'rollout':(self,('previous_reference','previous_command','measurement','alive','task_tick','physical_tick','since_reset_tick','alpha','speed','switched','switch_tick','preference_progress')),
        }
        root=self.plant.roots[lane].clone();root[:3]-=self.plant.origin[lane]
        physical=dict(root_pose=torch.cat((root[:3],root[[6,3,4,5]])),root_velocity=root[7:],joint_position=self.plant.dofs[lane,:,0],joint_velocity=self.plant.dofs[lane,:,1])
        state={name:{key:getattr(obj,key)[lane].detach().cpu().clone() for key in keys} for name,(obj,keys) in groups.items()}
        state['physical']={key:value.detach().cpu().clone() for key,value in physical.items()}
        for group in ('projection','follower'):
            state[group]={key:value[lane].detach().cpu().clone() for key,value in getattr(self.middle,group).items()}
        return dict(state=state,upper_sha256=self.upper_sha256,lower_sha256=self.lower_sha256,
            asset_sha256=self.plant.contract['exported_xml_sha256'],joint_names=list(self.plant.joints),
            controller_config=self.cc,rollout_tick=self.tick,middle_tick=self.middle.tick,
            channel_intervention_mode=self.policy.channel_intervention_mode,
            scope='exposed physical plus controller snapshot; PhysX solver/contact cache not serialized')

    @torch.no_grad()
    def restore_connection_state(self,snapshot):
        """Broadcast one saved exit into this matched nominal batch, without warmup.

        Preserves histories/ESO/filter clocks. Bind the next task geometry after
        restoration; do not reset controllers merely because the route changes.
        """
        identity=(self.upper_sha256,self.lower_sha256,self.plant.contract['exported_xml_sha256'],list(self.plant.joints),self.cc)
        source=tuple(snapshot[k] for k in ('upper_sha256','lower_sha256','asset_sha256','joint_names','controller_config'))
        if identity!=source or self.randomization is not None or self.switch is not None or self.plant.pendulum_rest is not None:
            raise ValueError('Connection replay requires the same nominal policy/model/controller interfaces')
        states=snapshot['state'];to=lambda v:v.to(self.device)
        self.plant.reset_idx(torch.arange(self.n,device=self.device),state={k:to(v) for k,v in states['physical'].items()})
        for name,obj in (('controller',self.controller),('lower',self.lower),('policy',self.policy),('middle',self.middle),('rollout',self)):
            for key,value in states[name].items():
                target=getattr(obj,key);target.copy_(to(value).expand_as(target))
        for group in ('projection','follower'):
            target=getattr(self.middle,group)
            for key,value in states[group].items():target[key].copy_(to(value).expand_as(target[key]))
        self.tick=snapshot['rollout_tick'];self.middle.tick=snapshot['middle_tick']
        self.policy.channel_intervention_mode=snapshot['channel_intervention_mode']

    @torch.no_grad()
    def begin_connection(self,route,speeds,alphas):
        """Assign a new A-to-B task while retaining inherited control dynamics."""
        self.route=route;self.middle.route=route
        self.speed.copy_(torch.as_tensor(speeds,device=self.device));self.alpha.copy_(torch.as_tensor(alphas,device=self.device))
        _,pose,speed=self.plant.observe()
        self.middle.projection=route.project(pose[:,:2],torch.zeros_like(speed),torch.zeros_like(speed))
        self.middle.follower=route.pursuit(self.middle.projection['progress'],pose,speed,self.speed)
        self.preference_progress=self.middle.projection['progress'].clone()
        self.task_tick.zero_();self.tick=0
        # middle.nominal/filtered/offsets/history and physical_tick are inherited.
        # Original reference publication applies its usual slew on the next step.
