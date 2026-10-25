"""Opt-in live original-path baseline after a tracking-domain exit.

This is a new recovery-task contract, not a change to historical termination.
No Actor is evaluated. Zero correction targets still pass through the original
environment's filter, rate limits, reference bounds and frozen lower kernel.
"""
import math
import jax
import jax.numpy as j
from .path_command_reward import failure_reward


def make_live_path_step(env, *, horizon_s):
    """Advance up to four real lower ticks and update history once if active.

    Keep the accumulated domain-exit flag as telemetry. Contact, numerical
    faults and the fixed horizon still stop execution permanently. Do not feed
    these domain-exited observations to an old trained Actor without a separate
    policy-domain contract. Returned scores use physical rather than domain
    termination and are not interchangeable with old domain-terminal rewards.
    """
    dt=float(env.cc.dt)
    upper=float(env.config['timing']['upper_dt_s'])
    if (not math.isfinite(horizon_s) or horizon_s<=0 or dt<=0
            or not math.isclose(upper,4*dt,abs_tol=1e-10)):
        raise ValueError('positive horizon and four lower ticks per upper interval required')
    limit=round(horizon_s/dt)
    if limit%4 or not math.isclose(limit*dt,horizon_s,abs_tol=1e-8):
        raise ValueError('horizon must align to complete upper intervals')
    reward_config=env.config['path_reward']

    def step(state,path,speed,pulse):
        def lower_tick(s,_):
            stopped=s.physical.failed|s.fault|(s.tick>=limit)
            return jax.lax.cond(stopped,lambda x:(x,env.zero_log),
                lambda x:env.tick(x,j.zeros(2),path,speed,pulse,horizon=horizon_s),s)
        end,logs=jax.lax.scan(lower_tick,state,None,length=4)
        active=logs['active_tick']
        n=j.sum(active)
        end=jax.lax.cond(n>0,env.record_frame,lambda x:x,end)
        rate=(end.offsets-state.offsets)/upper
        raw=reward_config['offset_rate_weight']*j.sum(
            (rate/j.asarray(reward_config['offset_rate_scales']))**2)
        effective=j.minimum(raw,reward_config['component_caps']['offset_rate'])
        logs['raw_components']['offset_rate']=j.where(active,raw,0.)
        logs['effective_components']['offset_rate']=j.where(active,effective,0.)
        logs['component_capped']['offset_rate']=active&(raw>effective)
        running=j.where(active,logs['tick_reward']-reward_config['scale']*dt*effective,0.)
        logs['tick_reward']=running
        last=j.maximum(n-1,0)
        terminal=failure_reward(j.ceil(horizon_s/upper)-state.tick//4,env.config)
        # A previously terminal input contributes no additional terminal charge.
        logs['scored_tick_reward']=j.where((n>0)&end.physical.failed,
            j.zeros(4).at[last].set(terminal),running)
        logs['physical_tick']=j.where(active,state.tick+j.cumsum(active)-1,-1)
        return end,logs

    return step
