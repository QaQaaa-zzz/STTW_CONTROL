"""Paired nominal/event GPU validation with balanced declared event cases."""
from dataclasses import replace
import jax
import jax.numpy as jp
import numpy as np
from .env import RecoveryEnv


def make_validator(env,actor,scale,config):
    cases=config.validation_events
    seeds=config.validation_seeds
    keys=jp.stack([jax.random.PRNGKey(s) for s in seeds for _ in (cases or [None])])
    n=len(keys);dt=env.config.controller.dt
    if cases:
        for case in cases:
            if any(not np.isclose(case[k]/dt,round(case[k]/dt),rtol=0,atol=1e-7) for k in ('start','duration')):
                raise ValueError('validation events must align to control ticks')
        events=jp.asarray([[round(c['start']/dt),round((c['start']+c['duration'])/dt),c.get('steer_rate',0.),c.get('force',0.),float(c.get('waveform','constant')=='half_sine'),c.get('rear_torque',0.)] for _ in seeds for c in cases])
        max_end=max(c['start']+c['duration'] for c in cases)
    else:
        max_end=(env.config.random_events.start_max+env.config.random_events.duration_max if env.config.random_events else env.config.disturbance_start+env.config.disturbance_duration)
    alphas=None
    if getattr(env.config,'priority',None) is not None:
        choices=env.config.priority.validation_alphas
        alphas=jp.repeat(jp.asarray(choices),n)
        keys=jp.tile(keys,(len(choices),1))
        if cases:events=jp.tile(events,(len(choices),1))
        n=len(keys)
    ve=RecoveryEnv(replace(env.config,horizon_seconds=max(env.config.horizon_seconds,max_end+config.validation_post_seconds)),backend='mjx')
    reset=jax.vmap(ve.reset);step=jax.vmap(ve.step)
    needed=int(np.ceil(config.validation_hold_seconds/dt))

    @jax.jit
    def validate(params,zero=False):
        initial=reset(keys)
        if alphas is not None:initial=jax.vmap(ve.set_priority)(initial,alphas)
        if cases:initial=initial.replace(event=events)
        nominal=initial.replace(event=initial.event.at[:,2:4].set(0.).at[:,5].set(0.))
        state=jax.tree.map(lambda a,b:jp.concatenate([a,b],axis=0),initial,nominal)
        event_end=initial.event[:,1];present=jp.any(initial.event[:,2:4]!=0,axis=1)|(initial.event[:,5]!=0)
        def tick(carry,_):
            state,hold=carry
            action=actor.apply(params['actor'],state.obs/scale)
            active=~state.done
            nxt=step(state,jp.where(zero,jp.zeros_like(action),action))
            features=jax.vmap(ve.path_features)(nxt.pose)
            speed=jp.sum(nxt.data.qvel[:,:3]*nxt.data.xmat[:,ve.bundle.chassis,:,0],axis=-1)
            extra=features[:n,0]-features[n:,0]
            post=active[:n]&(state.tick[:n]>=event_end)
            valid=post&~nxt.terminated[:n]&~nxt.terminated[n:]&(jp.abs(extra)<=config.validation_extra_tolerance)&(jp.abs(features[:n,0])<=config.validation_path_tolerance)&(jp.abs(features[:n,1])<=config.validation_heading_tolerance)&(jp.abs(speed[:n]-ve.config.speed_reference)<=config.validation_speed_tolerance)
            hold=jp.where(valid,hold+1,0)
            row=(active,features[:,0]**2,(speed-ve.config.speed_reference)**2,
                 jp.where(active[:n]&(state.tick[:n]>=initial.event[:,0]),jp.abs(extra),0.),post)
            return (nxt,hold),row
        (last,hold),(active,radial,speed,extra,post)=jax.lax.scan(tick,(state,jp.zeros(n,dtype=jp.int32)),None,length=ve.horizon)
        count=jp.maximum(jp.sum(active,axis=0),1)
        complete=(hold>=needed)&~last.terminated[:n]&~last.terminated[n:]
        budget=jp.maximum(0,(ve.horizon-event_end)*dt)
        settling=jp.where(complete,(last.tick[:n]-hold+needed-event_end)*dt,budget)
        return {'radial_rmse':jp.sqrt(jp.sum(jp.where(active,radial,0),axis=0)/count)[:n],
                'speed_rmse':jp.sqrt(jp.sum(jp.where(active,speed,0),axis=0)/count)[:n],
                'nominal_radial_rmse':jp.sqrt(jp.sum(jp.where(active,radial,0),axis=0)/count)[n:],
                'failed':last.terminated[:n],'nominal_failed':last.terminated[n:],
                'event_present':present,'post_event_hold_complete':complete,
                'settling_seconds':settling,'post_event_peak':jp.max(extra,axis=0),
                'steps':count[:n],'nominal_steps':count[n:]}
    return validate
