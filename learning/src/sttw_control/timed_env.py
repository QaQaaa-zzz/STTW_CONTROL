"""Random command tracking against an independent time-indexed reference pose."""
from dataclasses import replace
import jax
import jax.numpy as jp
from .env import RecoveryEnv
from .controller import initial_controller
from .observation import initial_history
from .recovery import initial_recovery
from .timed_reference import (schedule, command_at, raw_request_at, recovery_entry_at,
                              advance_reference, errors, geometry_table,
                              project_geometry, project_committed_geometry)
from .tracking_reward import transition,initial_return
from . import priority_return_v2 as v2


class TimedRecoveryEnv(RecoveryEnv):
    def control_reference(self,pose,reference_pose,reference_command,path_features=None):
        c=self.config;r=c.timed_reference
        features=errors(pose,reference_pose,reference_command) if path_features is None else path_features
        v,w=reference_command
        if self.config.tracking.geometric:w=v*features[2]
        requested_yaw=w-r.yaw_feedback*features[1]+r.lateral_feedback*features[0]
        steer=jp.arctan(c.controller.wheelbase*requested_yaw/(jp.maximum(v,.1)*jp.cos(c.controller.caster)))
        return jp.array([jp.clip(steer,-r.max_steer,r.max_steer),v])

    def prepare_timed(self,controller,actuator,history,measurement,tick,pose,alpha,ref_pose,ref_command,yaw,tracking_state,path_features=None,eso_enabled=False,base_output_scale=None,priority_v2_frame=None):
        feature=errors(pose,ref_pose,ref_command)
        return self._prepare(controller,actuator,history,measurement,tick,pose,alpha,
            command_override=self.control_reference(pose,ref_pose,ref_command,path_features),tracking_state=tracking_state,
            path_features_override=feature[:3] if path_features is None else path_features,timed_frame=jp.array([feature[3],ref_command[1],yaw]),
            eso_enabled=eso_enabled,base_output_scale=base_output_scale,priority_v2_frame=priority_v2_frame)

    def _event_for(self, commands, fallback):
        if not self.config.timed_reference.training_mix:
            return fallback
        scenario=commands[0,6].astype(jp.int32)
        sign=jp.where(commands[1,2] < 0.,-1.,1.)
        start=jp.rint(3./self.config.controller.dt)
        duration=jp.rint(.5/self.config.controller.dt)
        return jp.where(scenario==2,jp.array([start,start+duration,0.,2.*sign,0.,0.]),jp.zeros(6))

    def _begin_task(self, state, commands, event, *, reset_memory, preparation_failed=False):
        c=self.config
        ref=jp.array([c.speed_reference,0.]);pose=state.pose
        raw=raw_request_at(0,commands,c.controller.dt)
        table=jp.zeros((self.horizon+1,6)).at[0].set(jp.concatenate((jp.zeros(1),pose,ref))) if c.tracking.geometric else None
        geometric=errors(pose,pose,ref)[:3] if table is not None else None
        geometry=None
        if c.timed_reference.mode == 'geometry':
            geometry=geometry_table(commands,c.timed_reference,c.speed_reference,
                                    pose,c.horizon_seconds,c.controller.dt)
        controller=initial_controller(c.controller) if reset_memory else state.controller
        history=initial_history(c.observation) if reset_memory else state.history
        tracking=initial_return()
        v2_state=v2.initial_state() if c.observation.include_priority_v2 else None
        v2_frame=None
        if v2_state is not None:
            speed=jp.dot(jp.asarray(state.data.qvel[:3]),jp.asarray(state.data.xmat[self.bundle.chassis]).reshape(3,3)[:,0]) if c.forward_speed_source=="true" else state.measurement[5]*.1
            v2_frame=jp.concatenate([jp.atleast_1d(speed),v2.observation_context(v2_state,t=0.,episode_end=c.horizon_seconds,actual_progress=0.)])
        ctrl,h,obs,base,roll=self.prepare_timed(controller,state.actuator,history,
            state.measurement,jp.int32(0),pose,state.priority_alpha,pose,ref,jp.asarray(0.),
            tracking,geometric,eso_enabled=state.eso_enabled,
            base_output_scale=c.actuator.base_output_scale,priority_v2_frame=v2_frame)
        components=jax.tree.map(jp.zeros_like,state.tracking_components)
        raw_costs=({k:jp.zeros_like(v) for k,v in components.items() if k not in ("alive","deadline","failure")} if c.tracking.objective=="soft_budget_v1" else None)
        if c.tracking.objective=="priority_return_v2":
            result=v2.reward_terms(alpha=state.priority_alpha,speed_error=0.,lateral_error=0.,heading_error=0.)
            components=jax.tree.map(jp.zeros_like,result["reward_parts"])
            raw_costs=jax.tree.map(jp.zeros_like,result["raw_costs"])
        failed=jp.asarray(preparation_failed)
        return state.replace(controller=ctrl,history=h,obs=obs,base=base,reference=roll,event=event,
            command_schedule=commands,reference_pose=pose,reference_command=ref,yaw_rate=jp.asarray(0.),
            raw_reference_request=raw,geometric_table=table,geometric_features=geometric,
            path_segment=jp.int32(0),reference_geometry=geometry,path_progress=jp.asarray(0.),
            priority_v2_state=v2_state,physical_failed=failed,
            tracking_state=tracking,tracking_components=components,
            tracking_raw_costs=raw_costs,recovery=initial_recovery(),
            tick=jp.int32(0),reward=jp.asarray(0.),done=failed,terminated=failed,
            truncated=jp.bool_(False),end_code=jp.where(failed,5,0),preparation_failed=failed,
            active_base_output_scale=jp.asarray(c.actuator.base_output_scale))

    def _sample_task(self,state,key):
        commands=schedule(jax.random.fold_in(key,51),self.config.timed_reference,self.config.speed_reference)
        from .priority import sample_alpha
        alpha=sample_alpha(key,self.config.priority)
        state=state.replace(priority_alpha=alpha)
        event=self._event_for(commands,state.event)
        return self._begin_task(state,commands,event,reset_memory=not bool(self.config.preparation_seconds),
                                preparation_failed=state.preparation_failed)

    def prepare_state(self,seed=0):
        """Compute the physical/controller/history state before task time zero."""
        s=super().reset(seed)
        key=jax.random.PRNGKey(seed) if isinstance(seed,int) else seed
        commands=schedule(jax.random.fold_in(key,51),self.config.timed_reference,self.config.speed_reference)
        steps=int(round(self.config.preparation_seconds/self.config.controller.dt))
        if not steps:
            return s
        state=self._begin_task(s,commands,jp.zeros(6),reset_memory=True)
        straight=commands.at[:,1].set(self.config.speed_reference).at[:,2].set(0.)
        if straight.shape[1]>=6:straight=straight.at[:,5].set(0.).at[:,6].set(0.)
        prep=state.replace(command_schedule=straight,event=jp.zeros(6),
                           active_base_output_scale=jp.asarray(self.config.preparation_base_output_scale))
        if self.backend=='cpu':
            for _ in range(steps):
                prep=self.step(prep,jp.zeros(2))
        else:
            prep,_=jax.lax.scan(lambda carry,_:(self.step(carry,jp.zeros(2)),None),prep,None,length=steps)
        failed=prep.done|~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in jax.tree.leaves((prep.controller,prep.actuator,prep.history,prep.measurement,prep.pose))]))
        prep=prep.replace(eso_enabled=prep.eso_enabled|(steps*self.config.controller.dt>self.config.eso_start))
        return prep.replace(preparation_failed=failed)

    def reset_from_prepared(self,prepared,seed):
        """Start a new randomized task without recomputing the closed-loop preparation."""
        key=jax.random.PRNGKey(seed) if isinstance(seed,int) else seed
        return self._sample_task(prepared,key)

    def reset(self,seed=0,*,reference_id=None):
        if reference_id is not None:raise ValueError('timed training samples commands, not reference IDs')
        key=jax.random.PRNGKey(seed) if isinstance(seed,int) else seed
        prepared=self.prepare_state(key)
        return self.reset_from_prepared(prepared,key)

    def _advance(self,state,measurement,actuator,action,physical_contact,physics_finite,true_speed,pose):
        c=self.config;dt=c.controller.dt;tick=state.tick+1
        ref_pose=advance_reference(state.reference_pose,state.reference_command,dt)
        ref_command=command_at(tick,state.reference_command,state.command_schedule,dt,c.timed_reference)
        raw_request=raw_request_at(tick,state.command_schedule,dt)
        used_command=state.reference_command
        feature=errors(pose,ref_pose,used_command)
        progress=state.path_progress;segment=state.path_segment;geometric=state.geometric_features
        table=state.geometric_table
        if c.tracking.geometric:
            # Only commit the newly issued path interval. No reset-time future
            # integration and no path rebasing to the actual vehicle.
            arc=table[state.tick,0]+state.reference_command[0]*dt
            table=table.at[tick].set(jp.concatenate((jp.atleast_1d(arc),ref_pose,ref_command)))
            geometric,progress,segment=project_committed_geometry(pose,table,segment,progress,jp.linalg.norm(pose[:2]-state.pose[:2]),tick)
        elif c.timed_reference.mode == 'geometry':
            window=2.*jp.linalg.norm(pose[:2]-state.pose[:2])+c.timed_reference.projection_margin
            progress,ref_pose,curvature=project_geometry(pose,state.reference_geometry,progress,window)
            # Full-curve mode uses the original reset-integrated path tangent.
            used_command=jp.array([state.reference_command[0],curvature*state.reference_command[0]])
            ref_command=jp.array([ref_command[0],curvature*ref_command[0]])
            feature=errors(pose,ref_pose,used_command)
        reward_feature=geometric if c.tracking.geometric else feature
        dpsi=pose[2]-state.pose[2];yaw=jp.arctan2(jp.sin(dpsi),jp.cos(dpsi))/dt
        leaves=jax.tree.leaves((measurement,actuator,action,true_speed,pose,yaw,ref_pose,ref_command))
        invalid=~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in leaves]))|~jp.asarray(physics_finite)
        failed=invalid|(jp.abs(measurement[0])>c.roll_failure)|physical_contact
        scenario=(state.command_schedule[0,6].astype(jp.int32)
                  if c.timed_reference.training_mix else jp.int32(-1))
        recovery_trigger=(recovery_entry_at(state.tick,state.command_schedule,dt)
                          if c.timed_reference.training_mix else False)
        diagnostic_config=replace(c.tracking,objective="soft_budget_v1") if c.tracking.objective=="priority_return_v2" else c.tracking
        reward_result=transition(state.tracking_state,roll=measurement[0],roll_rate=measurement[1],
            speed_error=true_speed-state.reference_command[0],yaw_rate_error=yaw-state.reference_command[1],
            lateral_error=reward_feature[0],heading_error=reward_feature[1],longitudinal_error=feature[3],action=action,
            alpha=state.priority_alpha,dt=dt,alive_rate=c.alive_reward_rate,failure_penalty=c.failure_penalty,
            failed=failed,enabled=state.tick*dt>=c.tracking.start_seconds,config=diagnostic_config,
            recovery_trigger=recovery_trigger,clock_from_departure=scenario!=1,return_raw=diagnostic_config.objective=='soft_budget_v1')
        tracking,parts=reward_result[:2]
        raw_costs=reward_result[2] if len(reward_result)==3 else None
        if raw_costs is not None:failed=failed|(parts['failure']<0)
        v2_state=state.priority_v2_state
        v2_frame=None
        if c.observation.include_priority_v2:
            v2_state,events=v2.advance(v2_state,t=tick*dt,dt=dt,episode_end=c.horizon_seconds,
                reference_speed=ref_command[0],reference_yaw=ref_command[1],published_yaw_request=raw_request[1],previous_reference_speed=state.reference_command[0],
                reference_progress=table[tick,0],actual_progress=progress,
                speed_error=true_speed-used_command[0],lateral_error=reward_feature[0],heading_error=reward_feature[1],
                roll=measurement[0],roll_rate=measurement[1],failed=failed)
            if c.tracking.objective=="priority_return_v2":
                _,parts,raw_costs=transition(state.tracking_state,roll=measurement[0],roll_rate=measurement[1],
                    speed_error=true_speed-used_command[0],lateral_error=reward_feature[0],heading_error=reward_feature[1],
                    action=action,alpha=state.priority_alpha,dt=dt,alive_rate=0.,failure_penalty=200.,failed=failed,
                    enabled=True,config=c.tracking,priority_v2_state=v2_state,priority_v2_events=events,
                    task_penalty_already_paid=state.priority_v2_state.penalty_paid,return_raw=True)
            speed=true_speed if c.forward_speed_source=="true" else measurement[5]*.1
            v2_frame=jp.concatenate([jp.atleast_1d(speed),v2.observation_context(v2_state,t=tick*dt,episode_end=c.horizon_seconds,actual_progress=progress)])
        eso_enabled=state.eso_enabled|(tick*c.controller.dt>c.eso_start)
        ctrl,h,obs,base,roll=self.prepare_timed(state.controller,actuator,state.history,measurement,tick,pose,
            state.priority_alpha,ref_pose,ref_command,yaw,tracking,geometric,eso_enabled=eso_enabled,
            base_output_scale=state.active_base_output_scale,priority_v2_frame=v2_frame)
        bad=~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in jax.tree.leaves((ctrl,obs,base))]))
        invalid=invalid|bad;failed=failed|bad
        if v2_state is not None:v2_state=v2_state.replace(physical_failed=failed,task_complete=v2_state.task_complete & ~failed)
        parts={k:jp.where(bad,-c.failure_penalty if k=='failure' else 0.,v) for k,v in parts.items()}
        timeout=(tick>=self.horizon)&~failed
        code=jp.where(invalid,3,jp.where(physical_contact,4,jp.where(failed,1,jp.where(timeout,2,0))))
        recovery=state.recovery.replace(balance_recovered=tracking.hold>=c.tracking.hold_seconds,
            task_recovered=tracking.credited & ~tracking.pending & ~tracking.deadline_missed & ~failed)
        return state.replace(controller=ctrl,history=h,obs=jp.nan_to_num(obs),base=base,reference=roll,
            actuator=actuator,measurement=measurement,pose=pose,tick=tick,reward=sum(parts.values()),
            done=failed|timeout,terminated=failed,truncated=timeout,end_code=code,recovery=recovery,
            physical_failed=failed,priority_v2_state=v2_state,
            tracking_state=tracking,tracking_components=parts,tracking_raw_costs=raw_costs,reference_pose=ref_pose,
            reference_command=ref_command,raw_reference_request=raw_request,yaw_rate=yaw,
            path_progress=progress,path_segment=segment,geometric_features=geometric,geometric_table=table,
            eso_enabled=eso_enabled)
