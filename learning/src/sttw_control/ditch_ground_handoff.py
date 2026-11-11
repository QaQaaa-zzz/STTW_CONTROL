"""Optional measured landing handback to the existing H75 ground stack."""
import json
from pathlib import Path
import numpy as np
import torch


class DitchGroundHandoff:
    def __init__(self, env, spec, output):
        from .isaac_path_rollout import IsaacPathRollout
        from .torch_path_policy import TorchPathPolicy
        from .torch_path_reference import TorchPathReference
        n=env.num_envs
        self.env,self.spec,self.output=env,spec,Path(output)
        self.leg_hold=spec.get('leg_hold','joint_brake')
        self.handoff_stage=spec.get('handoff_stage',3)
        self.handoff_event=spec.get('handoff_event','stage')
        if self.handoff_event not in ('stage','first_far_contact'):
            raise ValueError('Unknown landing handoff event')
        self.task_continuation=spec.get('continue_to_horizon',False)
        if self.handoff_stage not in (3,4):
            raise ValueError('Handoff stage must be capture3 or completed hold4')
        if self.leg_hold not in ('joint_brake','finite_motor'):
            raise ValueError('Unknown handoff leg hold mode')
        settings=json.loads(Path(spec['settings']).read_text())
        settings.update(num_envs=n,alphas=[spec['alpha']],zero_middle=[spec.get('zero_middle',False)]*n,zero_lower=[True]*n,
                        terrains={'ditch':'shared'},episode_s=env.max_episode_length*env.dt,
                        initial_state=None,user_speed_m_s=spec['speed_m_s'],auto_reset_horizon=False,
                        record_control_commands=True)
        bundle=Path(spec['bundle']);identity=json.loads((bundle/'identity.json').read_text())
        config=json.loads((bundle/'config.json').read_text())
        if 'nominal_steer_slew_rad_s' in spec:
            if spec['nominal_steer_slew_rad_s']<=0:raise ValueError('Positive nominal steering reference slew required')
            config['follower']['nominal_steer_slew_rad_s']=spec['nominal_steer_slew_rad_s']
        actor=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],config,n,env.device)
        self.ground=IsaacPathRollout(bundle,settings,actor,plant=env.plant)
        reference_path=spec.get("reference_path")
        if reference_path:
            with np.load(reference_path) as saved:
                route={key:saved[key] for key in ("s","xy","heading","curvature")}
        else:
            ss=np.arange(0,30.01,.02)
            route=dict(s=ss,xy=np.column_stack((ss-5.,np.zeros_like(ss))),heading=np.zeros_like(ss),curvature=np.zeros_like(ss))
        self.ground.begin_connection(TorchPathReference(route,config,env.device,route_ids=spec.get('reference_route_ids')),self.ground.speed.clone(),self.ground.alpha.clone())
        self.active=False;self.commands=[];self.command_times=[];self.command_phases=[]
        self.history=[];self.record_history=spec.get("record_controller_history",False)
        self.approach=spec.get('approach')
        self.approaching=self.approach is not None
        self.approach_initialized=False
        (self.output/'ground_identity.json').write_text(json.dumps(dict(upper_sha256=self.ground.upper_sha256,
            lower_sha256=self.ground.lower_sha256,zero_lower=True,zero_middle=spec.get('zero_middle',False),controller=('ECBC/ESO + original follower; H75 correction zero' if spec.get('zero_middle',False) else 'H75 + ECBC/ESO; lower residual zero'),
            follower=config['follower'],reference=reference_path or 'Original world-local y=0, no reanchoring at landing',control_dt_s=.005,policy_dt_s=.02,
            feedback='Existing ground clean-state interface; frozen randomized physical properties retained; jump noise/delay only before handback',
            pendulum=self.leg_hold),indent=2))

    def step(self, latent):
        if self.env.num_envs != 1:
            raise ValueError('Use VectorDitchHandoff for batched stepping')
        e=self.env;g=self.ground
        if self.approaching:
            if not self.approach_initialized:
                g.resume_ground_lanes(e.all_ids,self.approach['speed_m_s'],self.approach['alpha'])
                self.approach_initialized=True
            if float(e.physical()[0][0,0])>=self.approach['jump_start_x_m']:
                self.approaching=False
                np.savez_compressed(self.output/'approach_entry.npz',root=e.physical()[0].cpu().numpy(),
                    observation=e.obs_buf.cpu().numpy(),joint_state=e.plant.dofs.cpu().numpy(),
                    time_s=float(e.episode_length_buf[0])*e.dt)
        ready=(bool(e.verified_jump.observed_far_contact[0]) if self.handoff_event=='first_far_contact'
               else int(e.verified_jump.scorer.stage[0])>=self.handoff_stage)
        if not self.approaching and not self.active and ready:
            before=e.physical()[0].clone()
            if self.leg_hold=='joint_brake':
                e.plant.set_pendulum_locked(True)
            else:
                # Preserve native limits and use existing finite-torque position servos.
                indices=e.plant.pendulum_servos.indices
                e.plant.pendulum_target=e.plant.dofs[:,indices,0].clone()
                e.plant.pendulum_target=torch.maximum(e.lo[2:],torch.minimum(e.hi[2:],e.plant.pendulum_target))
            g.resume_ground_lanes(e.all_ids,self.spec['speed_m_s'],self.spec['alpha'])
            g.physical_tick.fill_(round(float(e.episode_length_buf[0])*e.dt/.005))
            self.active=True
            (self.output/'handoff.json').write_text(json.dumps(dict(time_s=float(e.episode_length_buf[0])*e.dt,
                root=before.cpu().tolist(),stage=int(e.verified_jump.scorer.stage[0]),event=self.handoff_event,physical_reset=False,leg_hold=self.leg_hold,
                leg_target=e.plant.pendulum_target.cpu().tolist(),
                unchanged_root=bool(torch.equal(before,e.physical()[0]))),indent=2))
        if not self.active and not self.approaching:
            transition=e.step(latent)
            if self.handoff_stage==4 and bool(transition[4]['jump_success'].all()) and int(e.episode_length_buf[0])<e.max_episode_length:
                obs,priv,reward,done,info=transition
                return obs,priv,reward,torch.zeros_like(done),info
            return transition
        reward=torch.zeros(1,device=e.device);code=torch.zeros(1,device=e.device,dtype=torch.long);tick=0
        def physics(command):
            nonlocal reward,code,tick
            old_velocity=e.plant.roots[:,7:10].clone()
            result=e.plant.step(command,refresh_diagnostics=True)
            inc,done,success,stage=e.verified_jump.sample(e,torch.zeros(1,4,device=e.device),tick==e.physics_substeps-1)
            terminal=e.verified_jump.physical_failure if self.task_continuation or self.handoff_stage==4 else done
            reason=e.verified_jump.end_code if self.task_continuation or self.handoff_stage==4 else torch.where(success,13,e.verified_jump.end_code)
            code=torch.where(terminal&(code==0),reason,code)
            reward+=inc;tick+=1
            if self.approaching and tick==e.physics_substeps:
                from .phase_u_events import advance_events
                from .jump_heading import heading_features
                root=e.physical()[0]
                acceleration=(root[:,7:10]-old_velocity)/e.plant.physics_dt
                reference=e.cfg.get('policy_obstacle_reference_x_m',e.ditch['front_x_m'])
                distance=reference-(e.geometry.frontmost_x(e.plant.rigid)-e.plant.origin[:,0])
                # Existing chain convention: zero jump action history during ground approach.
                frame=e.history.make_frame(root,e.plant.dofs[:,:,0],e.plant.dofs[:,:,1],acceleration,distance,e.last_action)
                e.events=advance_events(e.events,dict(x=root[:,0],z=root[:,2],vertical_velocity=root[:,9],physical_failure=e.verified_jump.physical_failure,forward_velocity=root[:,7]),e.task)
                e.obs_buf=torch.cat((e.history.advance_frame(e.randomization.sensor_frame(frame),e.events['jump_signal'].float()),heading_features(root)),-1)
                e.privileged_obs_buf=e.physical()[2]
            return result
        g.physics_step_override=physics
        if self.record_history:
            # Observation-only trace; no solver snapshot, restoration, or control mutation.
            groups={
                'controller':(g.controller,('eso','disturbance','gains')),
                'lower':(g.lower,('frames','mask','previous_applied','yaw_rate')),
                'policy':(g.policy,('filtered','offsets','execution_mode','hold_channels')),
                'middle':(g.middle,('frames','mask','pose','yaw_rate','nominal','rates','fault','domain_exit')),
                'rollout':(g,('previous_reference','previous_command','measurement','alpha','speed','task_tick','physical_tick')),
            }
            row={group+'_'+key:getattr(obj,key).detach().cpu().numpy().copy()
                 for group,(obj,keys) in groups.items() for key in keys}
            row.update(root_xyzw=e.physical()[0].detach().cpu().numpy().copy(),
                       joint_state=e.plant.dofs.detach().cpu().numpy().copy())
            self.history.append(row)
        result=g.step();self.commands.append(result['executed_commands'].cpu().numpy())
        self.command_times.append(float(e.episode_length_buf[0])*e.dt)
        self.command_phases.append('approach' if self.approaching else 'continuation')
        scorer=e.verified_jump.scorer
        supervisor_failure=result['failure']&~scorer.done
        # A ground supervisor stop is a task failure, not a physical-success exit.
        # Close the same finite-horizon balance once, cancelling residual potential.
        reward-=torch.where(supervisor_failure,100.+scorer.phi,0.)
        scorer.phi.masked_fill_(supervisor_failure,0.)
        scorer.done|=supervisor_failure
        if bool(result['failure'].any()):
            (self.output/'ground_stop.json').write_text(json.dumps(dict(
                time_s=float(e.episode_length_buf[0]+1)*e.dt,
                domain_exit=g.middle.domain_exit.cpu().tolist(),numerical_fault=g.middle.fault.cpu().tolist(),
                ground_failure=result['failure'].cpu().tolist(),root=e.physical()[0].cpu().tolist(),
                tracking_error=result['tracking_error'].cpu().tolist()),indent=2))
        e.platform_held.copy_(scorer.success);e.episode_length_buf+=1;e.episode_return+=reward
        if self.task_continuation or self.handoff_stage==4:
            done=(code!=0)|result['failure']|(e.episode_length_buf>=e.max_episode_length)
            code=torch.where((e.episode_length_buf>=e.max_episode_length)&(code==0)&~result['failure'],9,code)
        else:
            done=scorer.done.clone()|result['failure']
        reason=torch.where(g.middle.domain_exit,15,torch.where(g.middle.fault,16,17))
        code=torch.where(result['failure']&(code==0),reason,code)
        info=dict(unscaled_reward=reward,end_code=code,reward_parts={'verified_jump':reward,'episode_return_override':torch.zeros_like(reward)},
                  jump_stage=scorer.stage.clone(),jump_success=scorer.success.clone())
        return e.obs_buf,e.privileged_obs_buf,reward*e.task['ppo']['reward_scaling'],done,info


class VectorDitchHandoff(DitchGroundHandoff):
    """Batched physical jump/ground execution for collection and recovery training.

    All lanes share one PhysX scene. Phase switches only select existing motor
    interfaces; neither touchdown nor switching resets the physical robot.
    The inherited single-lane evaluator remains the historical reference.
    """
    def __init__(self, env, spec, output, *, auto_reset=False):
        super().__init__(env,spec,output)
        if self.leg_hold!='finite_motor' or self.handoff_event!='first_far_contact':
            raise ValueError('Vector continuation uses finite motors and first far contact')
        self.auto_reset=auto_reset
        initialization=spec.get('observer_initialization_lanes',['cold']*env.num_envs)
        if len(initialization)!=env.num_envs or any(v not in ('cold','measured') for v in initialization):
            raise ValueError('Observer initialization requires one cold/measured mode per lane')
        self.measured_observer=torch.tensor([v=='measured' for v in initialization],device=env.device)
        route_ids=spec.get('handoff_route_ids')
        self.handoff_route_ids=None
        if route_ids is not None:
            if (not self.ground.route.batched or len(route_ids)!=env.num_envs
                    or any(i<0 or i>=len(self.ground.route.xy) for i in route_ids)):
                raise ValueError('Handoff routes require one valid route-bank index per lane')
            self.handoff_route_ids=torch.as_tensor(route_ids,device=env.device,dtype=torch.long)
            self.initial_route_ids=self.ground.route.route_ids.clone()
        self.phase=torch.ones(env.num_envs,device=env.device,dtype=torch.long)
        self.finished=torch.zeros_like(self.phase,dtype=torch.bool)
        self.reached_goal=self.finished.clone();self.push_seen=self.finished.clone()
        self.nonzero_push=any(lo!=0 or hi!=0 for lo,hi in env.rand['push_velocity_range'].values())
        self.potential=torch.zeros(env.num_envs,device=env.device)
        self.handoff_time=torch.full_like(self.potential,float('nan'))
        self.entry_time=torch.full_like(self.potential,float('nan'))
        self.terminal_code=torch.zeros_like(self.phase)
        self.ground.record_control_commands=False
        self.goal=spec['goal']
        self.road_half_width=spec.get('road_half_width_m',.6)
        self.road_violation=self.finished.clone()
        self.ground_only=torch.as_tensor(spec.get('ground_only_lanes',[False]*env.num_envs),device=env.device,dtype=torch.bool)
        reference=env.cfg.get('task_reference_path',spec.get('reference_path'))
        self.road=None
        if reference:
            with np.load(reference) as data:self.road=torch.as_tensor(data['xy'],device=env.device,dtype=torch.float32)
        self.record_events=env.cfg.get('record_event_states',False)
        self.physics_contact_modes=[];self.physics_roots=[];self.physics_valid=[]
        self.event_names=('flight_confirmed','clearance_confirmed','far_contact','sttw_handoff','goal')
        self.event_time=torch.full((env.num_envs,5),float('nan'),device=env.device)
        self.event_root=torch.full((env.num_envs,5,13),float('nan'),device=env.device)
        self.event_road_violation=torch.zeros(env.num_envs,5,device=env.device,dtype=torch.bool)
        self.downstream_planner=None
        if 'handoff_route_planner' in spec:
            from .handoff_route_planner import HandoffRoutePlanner
            self.downstream_planner=HandoffRoutePlanner(self,spec['handoff_route_planner'])
        self.reset()

    def reset(self):
        self.reset_idx(self.env.all_ids)
        return self.env.obs_buf,self.env.privileged_obs_buf

    def reset_idx(self,ids):
        e=self.env;g=self.ground
        e.reset_idx(ids)
        if self.downstream_planner is not None:
            for lane in ids.cpu().tolist():
                if lane in self.downstream_planner.pending:
                    self.downstream_planner.pending[lane][1]['status']='reset_before_publication'
                    del self.downstream_planner.pending[lane]
        if self.handoff_route_ids is not None:
            g.route.route_ids[ids]=self.initial_route_ids[ids]
        self.phase[ids]=torch.where(self.ground_only[ids],3,0 if self.approach else 1)
        self.finished[ids]=False;self.reached_goal[ids]=False;self.road_violation[ids]=False;self.push_seen[ids]=False
        self.event_time[ids]=float('nan');self.event_root[ids]=float('nan');self.event_road_violation[ids]=False
        self.potential[ids]=0.;self.terminal_code[ids]=0
        self.handoff_time[ids]=float('nan');self.entry_time[ids]=float('nan')
        self.entry_time[ids]=0. if not self.approach else float('nan')
        speed=self.approach['speed_m_s'] if self.approach else self.spec['speed_m_s']
        alpha=self.approach['alpha'] if self.approach else self.spec['alpha']
        g.resume_ground_lanes(ids,speed,alpha)
        g.physical_tick[ids]=0
        e.plant.set_steering_position_modes(self.phase==1)

    @torch.no_grad()
    def step(self,latent):
        from .phase_u_events import advance_events
        from .jump_heading import heading_features
        from isaaclab.envs.mdp.events import push_by_setting_velocity
        from types import SimpleNamespace
        e=self.env;g=self.ground;p=e.plant;alive=~self.finished
        root=e.physical()[0]
        if self.downstream_planner is not None:
            self.downstream_planner.publish_due(root)
        if self.approach:
            enter=alive&(self.phase==0)&(root[:,0]>=self.approach['jump_start_x_m'])
            self.phase[enter]=1;self.entry_time[enter]=e.episode_length_buf[enter]*e.dt
        handing=alive&(self.phase==1)&e.verified_jump.observed_far_contact
        ids=handing.nonzero().flatten()
        if len(ids):
            indices=p.pendulum_servos.indices
            hold=p.dofs[:,indices,0].clamp(min=e.lo[2:],max=e.hi[2:])
            p.pendulum_target[ids]=hold[ids]
            if self.downstream_planner is not None:
                self.downstream_planner.query(ids,root,e.obs_buf)
            if self.handoff_route_ids is not None:
                g.route.route_ids[ids]=self.handoff_route_ids[ids]
            g.resume_ground_lanes(ids,self.spec['speed_m_s'],self.spec['alpha'],observer_init_mask=self.measured_observer[ids])
            g.physical_tick[ids]=e.episode_length_buf[ids]*4
            self.phase[ids]=2;self.handoff_time[ids]=e.episode_length_buf[ids]*e.dt
            self.event_time[ids,3]=self.handoff_time[ids];self.event_root[ids,3]=root[ids];self.event_road_violation[ids,3]=self.road_violation[ids]
        jumping=(self.phase==1)&alive;grounding=alive&~jumping
        p.set_steering_position_modes(jumping)
        action=torch.tanh(latent);target=e.targets(action)
        push_ticks=round(e.rand['push_interval_s']/e.dt)
        ids=(alive&(e.episode_length_buf>0)&(e.episode_length_buf%push_ticks==0)).nonzero().flatten()
        if len(ids):
            push_by_setting_velocity(SimpleNamespace(scene={'robot':p.robot}),ids,e.rand['push_velocity_range'])
            self.push_seen[ids]=self.nonzero_push
        tick=0;failed=torch.zeros_like(alive);code=torch.zeros_like(self.phase)
        measured=None
        def physics(command):
            nonlocal tick,failed,code,measured
            valid_before=alive&~failed
            before=p.robot.data.root_link_lin_vel_w.clone()
            delivered=e.randomization.command(target)
            actual=torch.where(jumping[:,None],torch.stack((torch.zeros_like(delivered[:,0]),delivered[:,1]),-1),command)
            actual=torch.where(alive[:,None],actual,0.)
            leg=torch.where(jumping[:,None],delivered[:,2:],p.pendulum_target)
            result=p.step(actual,pendulum_target=leg,steering_position_target=delivered[:,:1],refresh_diagnostics=True)
            e.verified_jump.sample(e,torch.where(jumping[:,None],action,0.),tick==e.physics_substeps-1)
            now=e.verified_jump.physical_failure&~failed
            code=torch.where(now,e.verified_jump.end_code,code);failed|=now
            tick+=1
            state=e.verified_jump.last_root
            if self.road is None:distance=state[:,1].abs()
            else:
                road=self.road[g.route.route_ids] if self.road.ndim==3 else self.road
                delta=state[:,None,:2]-road[...,:-1,:];segment=road[...,1:,:]-road[...,:-1,:]
                fraction=((delta*segment).sum(-1)/segment.square().sum(-1)).clamp(0,1)
                distance=(delta-fraction[:,:,None]*segment).square().sum(-1).amin(-1).sqrt()
            self.road_violation|=alive&(distance>self.road_half_width)
            if self.record_events:
                self.physics_contact_modes.append(e.verified_jump.last_contact_mode.clone())
                self.physics_roots.append(state.clone());self.physics_valid.append(valid_before.clone())
                scorer=e.verified_jump.scorer
                for j,occurred in enumerate((scorer.stage>=1,scorer.stage>=2,e.verified_jump.observed_far_contact)):
                    take=alive&~failed&occurred&torch.isnan(self.event_time[:,j])
                    self.event_time[take,j]=(e.episode_length_buf[take]+tick/e.physics_substeps)*e.dt
                    self.event_root[take,j]=state[take]
                    self.event_road_violation[take,j]=self.road_violation[take]
            if tick%e.sensor_substeps==0:
                state=e.physical()[0]
                distance=e.cfg.get('policy_obstacle_reference_x_m',e.ditch['front_x_m'])-(e.geometry.frontmost_x(p.rigid)-p.origin[:,0])
                acceleration=(p.robot.data.root_link_lin_vel_w-before)/p.physics_dt
                frame=e.history.make_frame(state,p.dofs[:,:,0],p.dofs[:,:,1],acceleration,distance,torch.where(jumping[:,None],action,e.last_action))
                measured=e.randomization.sensor_frame(frame)
            return result
        g.physics_step_override=physics
        result=g.step()
        failed|=grounding&result['failure']
        ground_reason=torch.where(g.middle.domain_exit,15,torch.where(g.middle.fault,16,17))
        code=torch.where(grounding&result['failure']&(code==0),ground_reason,code)
        root,contact,_=e.physical()
        e.events=advance_events(e.events,dict(x=root[:,0],z=root[:,2],vertical_velocity=root[:,9],forward_velocity=root[:,7],physical_failure=failed),e.task)
        e.last_action.copy_(torch.where(jumping[:,None],action,e.last_action))
        e.obs_buf=e.history.advance_frame(measured,e.events['jump_signal'].float())
        if e.num_obs==80:
            features=heading_features(root)
            noise=torch.as_tensor(e.cfg['heading_observation_noise'],device=e.device)
            e.obs_buf=torch.cat((e.obs_buf,features+(2*torch.rand_like(features)-1)*noise),-1)
        e.privileged_obs_buf=e.history.privileged(e.obs_buf[:,:76],root,p.dofs[:,:,0],p.dofs[:,:,1],contact)
        e.episode_length_buf+=1
        goal=((self.phase==2)|(self.phase==3))&(root[:,0]>=self.goal['x_min_m'])&(root[:,0]<=self.goal['x_max_m'])&(root[:,1].abs()<=self.goal['abs_y_max_m'])&(root[:,7]>=self.goal['min_vx_m_s'])
        timeout=e.episode_length_buf>=e.max_episode_length
        done=alive&(failed|goal|timeout)
        success=done&goal&~failed&~self.road_violation
        self.event_time[success,4]=e.episode_length_buf[success]*e.dt;self.event_root[success,4]=root[success]
        self.reached_goal|=success
        # Telescoping shaping; completed episodic return is exactly +100/-100.
        # Timeout cannot earn a better return by avoiding a difficult jump.
        stage=e.verified_jump.scorer.stage
        phi=50.*((root[:,0]-1.5)/(self.goal['x_min_m']-1.5)).clamp(0,1)+10.*(stage>=1)+10.*(stage>=2)+10.*(self.phase==2)
        reward=torch.where(alive,phi-self.potential,0.)
        reward=torch.where(done,torch.where(success,100.,-100.)-self.potential,reward)
        self.potential=torch.where(alive,phi,self.potential)
        e.episode_return+=reward
        code=torch.where(success,13,code)
        code=torch.where(done&goal&self.road_violation&~failed,18,code)
        code=torch.where(done&(code==0),9,code)
        self.terminal_code=torch.where(done,code,self.terminal_code)
        self.finished|=done
        info=dict(time_outs=torch.zeros_like(reward),unscaled_reward=reward,end_code=code,
                  task_success=success,jump_stage=stage.clone(),jump_success=success,
                  handoff=(self.phase==2),jump_active=jumping,road_violation=self.road_violation.clone(),reward_parts={'task':reward})
        ids=done.nonzero().flatten()
        if len(ids):
            info['episode']=dict(reward=e.episode_return[ids].mean(),length=e.episode_length_buf[ids].float().mean(),task_success=success[ids].float().mean(),physical_failure=failed[ids].float().mean(),handoff=(self.phase[ids]==2).float().mean(),push_seen=self.push_seen[ids].float().mean())
            if self.auto_reset:self.reset_idx(ids)
        e.total_physics_steps+=e.physics_substeps*e.num_envs
        return e.obs_buf,e.privileged_obs_buf,reward*e.task['ppo']['reward_scaling'],done,info
