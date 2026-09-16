"""Random command tracking against an independent time-indexed reference pose."""
import jax
import jax.numpy as jp
from .env import RecoveryEnv
from .controller import initial_controller
from .observation import initial_history
from .timed_reference import schedule,command_at,advance_reference,errors
from .tracking_reward import transition,initial_return


class TimedRecoveryEnv(RecoveryEnv):
    def control_reference(self,pose,reference_pose,reference_command):
        c=self.config;r=c.timed_reference
        features=errors(pose,reference_pose,reference_command)
        v,w=reference_command
        requested_yaw=w-r.yaw_feedback*features[1]+r.lateral_feedback*features[0]
        steer=jp.arctan(c.controller.wheelbase*requested_yaw/(jp.maximum(v,.1)*jp.cos(c.controller.caster)))
        return jp.array([jp.clip(steer,-r.max_steer,r.max_steer),v])

    def prepare_timed(self,controller,actuator,history,measurement,tick,pose,alpha,ref_pose,ref_command,yaw,tracking_state):
        feature=errors(pose,ref_pose,ref_command)
        return self._prepare(controller,actuator,history,measurement,tick,pose,alpha,
            command_override=self.control_reference(pose,ref_pose,ref_command),tracking_state=tracking_state,
            path_features_override=feature[:3],timed_frame=jp.array([feature[3],ref_command[1],yaw]))

    def reset(self,seed=0,*,reference_id=None):
        if reference_id is not None:raise ValueError('timed training samples commands, not reference IDs')
        s=super().reset(seed)
        key=jax.random.PRNGKey(seed) if isinstance(seed,int) else seed
        commands=schedule(jax.random.fold_in(key,51),self.config.timed_reference,self.config.speed_reference)
        ref=jp.array([self.config.speed_reference,0.]);pose=s.pose
        ctrl,h,obs,base,roll=self.prepare_timed(initial_controller(self.config.controller),s.actuator,
            initial_history(self.config.observation),s.measurement,s.tick,s.pose,s.priority_alpha,
            pose,ref,jp.asarray(0.),initial_return())
        return s.replace(controller=ctrl,history=h,obs=obs,base=base,reference=roll,
            command_schedule=commands,reference_pose=pose,reference_command=ref,yaw_rate=jp.asarray(0.))

    def _advance(self,state,measurement,actuator,action,physical_contact,physics_finite,true_speed,pose):
        c=self.config;dt=c.controller.dt;tick=state.tick+1
        ref_pose=advance_reference(state.reference_pose,state.reference_command,dt)
        ref_command=command_at(tick,state.reference_command,state.command_schedule,dt,c.timed_reference)
        feature=errors(pose,ref_pose,state.reference_command)
        dpsi=pose[2]-state.pose[2];yaw=jp.arctan2(jp.sin(dpsi),jp.cos(dpsi))/dt
        leaves=jax.tree.leaves((measurement,actuator,action,true_speed,pose,yaw,ref_pose,ref_command))
        invalid=~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in leaves]))|~jp.asarray(physics_finite)
        failed=invalid|(jp.abs(measurement[0])>c.roll_failure)|physical_contact
        tracking,parts=transition(state.tracking_state,roll=measurement[0],roll_rate=measurement[1],
            speed_error=true_speed-state.reference_command[0],yaw_rate_error=yaw-state.reference_command[1],
            lateral_error=feature[0],heading_error=feature[1],longitudinal_error=feature[3],action=action,
            alpha=state.priority_alpha,dt=dt,alive_rate=c.alive_reward_rate,failure_penalty=c.failure_penalty,
            failed=failed,enabled=state.tick*dt>=c.tracking.start_seconds,config=c.tracking)
        ctrl,h,obs,base,roll=self.prepare_timed(state.controller,actuator,state.history,measurement,tick,pose,
            state.priority_alpha,ref_pose,ref_command,yaw,tracking)
        bad=~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in jax.tree.leaves((ctrl,obs,base))]))
        invalid=invalid|bad;failed=failed|bad
        parts={k:jp.where(bad,-c.failure_penalty if k=='failure' else 0.,v) for k,v in parts.items()}
        timeout=(tick>=self.horizon)&~failed
        code=jp.where(invalid,3,jp.where(physical_contact,4,jp.where(failed,1,jp.where(timeout,2,0))))
        recovery=state.recovery.replace(balance_recovered=tracking.hold>=c.tracking.hold_seconds,
            task_recovered=tracking.credited & ~tracking.pending & ~tracking.deadline_missed & ~failed)
        return state.replace(controller=ctrl,history=h,obs=jp.nan_to_num(obs),base=base,reference=roll,
            actuator=actuator,measurement=measurement,pose=pose,tick=tick,reward=sum(parts.values()),
            done=failed|timeout,terminated=failed,truncated=timeout,end_code=code,recovery=recovery,
            tracking_state=tracking,tracking_components=parts,reference_pose=ref_pose,
            reference_command=ref_command,yaw_rate=yaw)
