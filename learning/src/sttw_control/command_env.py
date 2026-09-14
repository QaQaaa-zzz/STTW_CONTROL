"""Command tracking task using the shared ECBC, actuator and physics implementation."""
import jax
import jax.numpy as jp
from .env import RecoveryEnv
from .controller import initial_controller
from .observation import initial_history
from .motion_commands import sample_schedule,reference_at,reward_terms

class CommandRecoveryEnv(RecoveryEnv):
    def requested(self,tick,schedule):
        return reference_at(tick*self.config.controller.dt,schedule)

    def command(self,tick,pose=None,schedule=None):
        # Reference conversion only, not a residual action allocator. The original
        # yaw request remains observable and is the reward/evaluation target.
        c=self.config
        v,r,_=self.requested(tick,schedule) if schedule is not None else jp.array([c.speed_reference,0.,.5])
        steer=jp.arctan(c.controller.wheelbase*r/(v*jp.cos(c.controller.caster)))
        return jp.array([steer,v])

    def prepare_command(self,controller,actuator,history,measurement,tick,pose,alpha,schedule,yaw):
        cmd=self.command(tick,pose,schedule)
        raw=self.requested(tick,schedule)
        return self._prepare(controller,actuator,history,measurement,tick,pose,alpha,cmd,jp.array([raw[1],yaw]))

    def reset(self,seed=0):
        s=super().reset(seed)
        key=jax.random.PRNGKey(seed) if isinstance(seed,int) else seed
        schedule=sample_schedule(jax.random.fold_in(key,51),self.config.motion_commands,self.config.speed_reference)
        # Initial simulated speed is explicit; fixed panels must agree with it.
        alpha=jp.where(self.config.motion_commands.dynamic_alpha,schedule[0,3],s.priority_alpha)
        ctrl,h,obs,base,ref=self.prepare_command(initial_controller(self.config.controller),s.actuator,initial_history(self.config.observation),s.measurement,s.tick,s.pose,alpha,schedule,jp.asarray(0.))
        return s.replace(controller=ctrl,history=h,obs=obs,base=base,reference=ref,priority_alpha=alpha,command_schedule=schedule,yaw_rate=jp.asarray(0.),priority_locked=jp.bool_(False))

    def _advance(self,state,measurement,actuator,action,physical_contact,physics_finite,true_speed,pose):
        c=self.config;tick=state.tick+1
        dpsi=pose[2]-state.pose[2]
        yaw=jp.arctan2(jp.sin(dpsi),jp.cos(dpsi))/c.controller.dt
        upcoming=self.requested(tick,state.command_schedule)
        alpha=jp.where(c.motion_commands.dynamic_alpha & ~state.priority_locked,upcoming[2],state.priority_alpha)
        controller,history,obs,base,ref=self.prepare_command(state.controller,actuator,state.history,measurement,tick,pose,alpha,state.command_schedule,yaw)
        leaves=jax.tree.leaves((controller,actuator,obs,measurement,action,true_speed,yaw))
        invalid=~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in leaves]))|~jp.asarray(physics_finite)
        fallen=jp.abs(measurement[0])>c.roll_failure
        failed=invalid|fallen|physical_contact;timeout=(tick>=self.horizon)&~failed
        raw=self.requested(state.tick,state.command_schedule)
        terms=reward_terms(measurement[0],measurement[1],true_speed-raw[0],yaw-raw[1],action,state.priority_alpha,c.motion_commands)
        reward=jp.where(failed,-c.failure_penalty,c.controller.dt*(c.alive_reward_rate-sum(terms.values())))
        code=jp.where(invalid,3,jp.where(physical_contact,4,jp.where(fallen,1,jp.where(timeout,2,0))))
        return state.replace(controller=controller,actuator=actuator,history=history,measurement=measurement,pose=pose,reference=ref,base=base,obs=jp.nan_to_num(obs),tick=tick,reward=reward,done=failed|timeout,terminated=failed,truncated=timeout,end_code=code,priority_alpha=alpha,yaw_rate=yaw)


def make_command_validator(env,actor,scale,config):
    keys=jp.stack([jax.random.PRNGKey(s) for a in env.config.priority.validation_alphas for s in config.validation_seeds])
    alphas=jp.repeat(jp.asarray(env.config.priority.validation_alphas),len(config.validation_seeds))
    @jax.jit
    def validate(params,zero=False):
        state=jax.vmap(env.reset)(keys)
        state=jax.vmap(env.set_priority)(state,alphas)
        def tick(s,_):
            active=~s.done
            a=actor.apply(params['actor'],s.obs/scale)
            n=jax.vmap(env.step)(s,jp.where(zero,jp.zeros_like(a),a))
            raw=jax.vmap(env.requested)(s.tick,s.command_schedule)
            speed=jp.sum(n.data.qvel[:,:3]*n.data.xmat[:,env.bundle.chassis,:,0],axis=-1)
            return n,(active,(speed-raw[:,0])**2,(n.yaw_rate-raw[:,1])**2,jp.abs(n.measurement[:,0]),n.reward,s.tick*env.config.controller.dt<jp.minimum(1.,s.command_schedule[:,1,0]) if s.command_schedule.shape[1]>1 else s.tick*env.config.controller.dt<1.)
        last,(active,speed,yaw,roll,reward,initial)=jax.lax.scan(tick,state,None,length=env.horizon)
        count=jp.maximum(jp.sum(active,axis=0),1)
        first=active&initial
        first_count=jp.maximum(jp.sum(first,axis=0),1)
        return dict(episode_return=jp.sum(jp.where(active,reward,0),axis=0),initial_speed_rmse=jp.sqrt(jp.sum(jp.where(first,speed,0),axis=0)/first_count),initial_yaw_rmse=jp.sqrt(jp.sum(jp.where(first,yaw,0),axis=0)/first_count),speed_rmse=jp.sqrt(jp.sum(jp.where(active,speed,0),axis=0)/count),yaw_rmse=jp.sqrt(jp.sum(jp.where(active,yaw,0),axis=0)/count),failed=last.terminated,steps=count,roll_peak=jp.max(jp.where(active,roll,0),axis=0),roll_exceed_fraction=jp.sum(active&(roll>env.config.motion_commands.roll_working_limit),axis=0)/count)
    return validate
