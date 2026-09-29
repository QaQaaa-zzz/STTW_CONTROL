"""V3 finite task: direct command corrections, four unchanged physical ticks."""
import jax
import jax.numpy as jp
from flax import struct
from .teleop_env import TeleopEnv
from .controller import _system
from .direct_command_policy import (correction_tick,raw_context,initial_history,make_frame,push_history,assemble_observation)
from .direct_command_reward import interval_cost,failure_reward
from .direct_command_scenarios import schedule,publish_command

@struct.dataclass
class DirectState:
    physical: object
    history: object
    offsets: object
    settle_clock: object
    command_rates: object
    yaw_rate: object
    previous_bounded: object
    env_id: object
    episode_index: object
    rows: object
    family: object
    slew: object
    alpha: object
    tick: object
    fault: object
    lower: object=None

class DirectCommandEnv:
    def __init__(self,spec):
        self.spec=spec;self.physics=TeleopEnv(config=spec,backend='mjx')
        self.cc=self.physics.cc;self.ac=self.physics.ac;self.zero_log=None
        self.lower=None
        if spec.get('lower_controller') is not None:
            from .frozen_lower_controller import FrozenLowerController
            self.lower=FrozenLowerController(spec['lower_controller'])
    def reset(self,snapshot,env_id,episode_index,alpha,rows=None,slew=None):
        raw=jp.stack((snapshot.raw[0],jp.asarray(0.)))
        sampled,family,drawn_slew=schedule(env_id,episode_index,raw[0],self.spec)
        rows=sampled if rows is None else rows;slew=drawn_slew if slew is None else slew
        pose=self.physics.helpers.pose(snapshot.data)
        p=snapshot.replace(reference_pose=pose,yaw_unwrapped=pose[2],yaw_wrapped=pose[2],raw=raw,
            governor=snapshot.governor.replace(current_reference=raw,last_goal=raw))
        issued,rates,_=publish_command(raw,rows,jp.int32(0),slew,self.cc.dt)
        state=DirectState(p.replace(raw=issued),initial_history(self.spec),jp.zeros(2),jp.asarray(0.),rates,
            jp.asarray(0.),jp.zeros(2),jp.asarray(env_id,jp.int32),jp.asarray(episode_index,jp.int32),rows,family,slew,
            jp.asarray(alpha),jp.int32(0),~((alpha==0)|(alpha==1)))
        if self.lower is not None:state=state.replace(lower=self.lower.initial(pose))
        return self.record_frame(state)
    def observation(self,s):
        chi,g,eligible,_=raw_context(s.physical.raw,s.command_rates,s.settle_clock,self.cc,self.spec)
        debt=s.physical.reference_pose[2]-s.physical.yaw_unwrapped
        actor,clipped,fault=assemble_observation(s.history,s.alpha,debt,chi,g,s.settle_clock,s.offsets,self.spec)
        critic=jp.concatenate([actor,jp.asarray([(800-s.tick//4)/800.])])
        return actor,critic,fault|s.fault,clipped
    def record_frame(self,s):
        m,_,v=self.physics.observe(s.physical.data)
        _,_,a4,_,_=_system(m[5]*.1,s.physical.controller.gains,self.cc)
        m=m.at[4].set(s.yaw_rate)
        frame=make_frame(measurement=m,forward_speed=v,previous_governed=s.physical.governor.current_reference,
            previous_final_command=s.physical.actuator.previous,previous_bounded_residual=s.previous_bounded,
            raw=s.physical.raw,raw_rates=s.command_rates,eso_equilibrium_shift=-s.physical.controller.disturbance/a4,cc=self.cc)
        return s.replace(history=push_history(s.history,frame),fault=s.fault|(~jp.all(jp.isfinite(frame))&~s.physical.failed))
    def _tick(self,s,z,bypass):
        p=s.physical;raw=p.raw
        chi,g,eligible,next_clock=raw_context(raw,s.command_rates,s.settle_clock,self.cc,self.spec)
        offsets,governed,flags=correction_tick(s.offsets,z,raw,self.spec)
        offsets=jp.where(bypass,jp.zeros(2),offsets);governed=jp.where(bypass,raw,governed)
        previous=p.actuator.previous
        override=None;lower=s.lower;lower_fault=jp.bool_(False)
        if self.lower is not None:
            from .closed_loop_kernel import preview_controls,controls
            measurement,pose,_=self.physics.observe(p.data)
            preview=preview_controls(p.controller,measurement,raw,governed,p.physical_tick*self.cc.dt>3.,self.cc)
            lower,lower_action,lower_obs,lower_flags=self.lower.prepare(lower,measurement,pose,governed,preview[2],previous)
            lower_fault=~lower_flags['finite']
            safe_action=jp.where(lower_fault,jp.zeros(2),lower_action)
            override=controls(p.controller,p.actuator,measurement,raw,governed,p.physical_tick*self.cc.dt>3.,bypass,
                              self.cc,self.ac,lower_action=safe_action,previewed=preview)
        p,log=self.physics._step(p,governed,bypass,exact_governed=True,control_override=override)
        if self.lower is not None:
            measurement,pose,speed=self.physics.observe(p.data)
            lower=self.lower.after_step(lower,governed,pose,measurement,speed,safe_action,p.failed,s.tick)
            log.update(lower_action=lower_action,lower_path_features=lower_flags['path_features'],
                       lower_endpoint_extension=lower_flags['path_endpoint_extension'],lower_fault=lower_fault)
        yaw_rate=(p.yaw_unwrapped-s.physical.yaw_unwrapped)/self.cc.dt
        costs=interval_cost(alpha=s.alpha,chi=chi,g=g,raw=raw,
            actual_speed=log['actual_forward_speed'],actual_steer=log['actual_delta'],
            heading_error=log['e_psi_unwrapped'],roll=log['phi'],roll_rate=log['phi_dot'],
            executed_offsets=offsets,final_command=log['final_command'],previous_final_command=previous,spec=self.spec)
        _,_,target=publish_command(raw,s.rows,s.tick,s.slew,self.cc.dt)
        next_raw,next_rates,_=publish_command(raw,s.rows,s.tick+1,s.slew,self.cc.dt)
        next_raw=jp.where(p.failed,raw,next_raw)
        next_rates=jp.where(p.failed,s.command_rates,next_rates)
        log.update(target=target,raw_rates=s.command_rates,offsets=offsets,chi=chi,g=g,settle_clock=s.settle_clock,
            eligible=eligible,alpha=s.alpha,family=s.family,episode_index=s.episode_index,env_id=s.env_id,
            active_tick=jp.bool_(True),time=s.tick*self.cc.dt,yaw_rate=yaw_rate,
            latent_z=z,raw_cost=costs['raw_cost'],effective_cost=costs['effective_cost'],
            cap_fraction=costs['cap_fraction'],tick_reward=costs['reward'],policy_fault=s.fault,
            raw_components=costs['raw_components'],effective_components=costs['effective_components'])
        for name,value in flags.items():log['reference_'+name]=value
        return s.replace(physical=p.replace(raw=next_raw),offsets=offsets,settle_clock=next_clock,
            command_rates=next_rates,yaw_rate=yaw_rate,previous_bounded=log['applied_residual'],tick=s.tick+1,
            lower=lower,fault=s.fault|lower_fault),log
    def set_log_template(self,s):
        shape=jax.eval_shape(lambda s:self._tick(s,jp.zeros(2),False),s)[1]
        self.zero_log=jax.tree.map(lambda x:jp.zeros(x.shape,x.dtype),shape)
    def policy_step(self,s,z,bypass=False):
        s=s.replace(fault=s.fault|~jp.all(jp.isfinite(z)))
        j=s.tick//4
        def tick(c,_):
            return jax.lax.cond(c.physical.failed|c.fault,
                lambda c:(c,self.zero_log),lambda c:self._tick(c,z,bypass),c)
        end,logs=jax.lax.scan(tick,s,None,length=4)
        end=self.record_frame(end)
        normal=jp.sum(logs['tick_reward']);failed=end.physical.failed
        reward=jp.where(failed,failure_reward(800-j,self.spec),normal)
        active=jp.sum(logs['active_tick'].astype(jp.int32));last=jp.maximum(active-1,0)
        # Full failure replacement once, including when failure occurs inside repeat.
        failure_rewards=jp.zeros(4).at[last].set(reward)
        logs['scored_tick_reward']=jp.where(failed,failure_rewards,logs['tick_reward'])
        logs['failure_cost']=jp.where(failed,-failure_rewards,jp.zeros(4))
        logs['scored_effective_components']=jax.tree.map(lambda x:jp.where(failed,jp.zeros_like(x),x),logs['effective_components'])
        logs['finite_task_end']=jp.full((4,),(end.tick>=3200)&~failed&~end.fault)
        done=failed|end.fault|(end.tick>=3200)
        final_obs=self.observation(end)
        return end,reward,done,logs,final_obs
