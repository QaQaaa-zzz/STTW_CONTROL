"""Bounded held/live comparison and causal original-path diagnostic geometry."""
import numpy as np


def prepare_cases(source):
    selected=[r for r in source if r['panel']=='physical_deterministic' and r['force_N']
              and (r['alpha'] in (0.,.5) or r['role']=='zero_middle')]
    result=[]
    for row in selected:
        for tail in ('held','live_zero'):
            result.append(dict(row,source_case_id=row['case_id'],
                case_id=row['case_id']+'/'+tail,tail_rule=tail))
    if len(result)!=36 or len({r['case_id'] for r in result})!=36:
        raise ValueError('expected eighteen unique source cases, two tails each')
    return result


def geometric_task_trace(data,path,config,*,initial_xy):
    """Recompute task errors from real pose, without feeding held controls.

    Mirror the original local segment projection, including monotone progress,
    bounded forward search and floating-point tie tolerance. No missing yaw is
    inferred from XY velocity and no missing samples are interpolated.
    """
    active=np.asarray(data['active_tick'],bool)
    xy=np.asarray(data['actual_xy'],float);yaw=np.asarray(data['actual_yaw'],float)
    if not np.isfinite(xy[active]).all() or not np.isfinite(yaw[active]).all():
        raise ValueError('nonfinite active pose prevents task classification')
    s=np.asarray(path['s'],float);points=np.asarray(path['xy'],float)
    heading=np.asarray(path['heading'],float)
    if not np.all(np.diff(s)>0):raise ValueError('nonincreasing original path')
    c=config['path'];previous=0.;old=np.asarray(initial_xy,float)
    result={k:np.full(len(active),np.nan) for k in
            ['path_progress','path_cross_track','path_heading_error','goal_section_signed_distance']}
    goal=float(path['goal']);goal_xy=np.array([np.interp(goal,s,points[:,k]) for k in (0,1)])
    goal_h=np.interp(goal,s,heading)
    for tick in np.flatnonzero(active):
        distance=np.linalg.norm(xy[tick]-old)
        lo=max(0.,previous-c['projection_backward_m'])
        hi=min(s[-1],previous+max(c['projection_forward_floor_m'],
              c['projection_forward_motion_multiplier']*distance+c['projection_forward_padding_m']))
        idx=np.flatnonzero((s[1:]>=lo)&(s[:-1]<=hi))
        a=points[idx];vec=points[idx+1]-a;span=s[idx+1]-s[idx]
        u=((xy[tick]-a)*vec).sum(axis=1)/(vec*vec).sum(axis=1)
        u=np.clip(u,np.clip((lo-s[idx])/span,0,1),np.clip((hi-s[idx])/span,0,1))
        ss=s[idx]+u*span;projection=a+u[:,None]*vec
        d2=((xy[tick]-projection)**2).sum(axis=1)
        at=int(np.argmin(np.where(d2<=d2.min()+1e-10,np.abs(ss-previous),np.inf)))
        h=np.interp(ss[at],s,heading);delta=xy[tick]-projection[at]
        result['path_cross_track'][tick]=-np.sin(h)*delta[0]+np.cos(h)*delta[1]
        result['path_heading_error'][tick]=np.arctan2(np.sin(h-yaw[tick]),np.cos(h-yaw[tick]))
        previous=max(previous,ss[at]);result['path_progress'][tick]=previous
        result['goal_section_signed_distance'][tick]=(xy[tick]-goal_xy)@np.array([np.cos(goal_h),np.sin(goal_h)])
        old=xy[tick]
    return result


def make_comparison_step(env,actor,*,horizon_s,mode):
    """Original prefix; after domain exit select held or live-zero by data.

    Only deterministic mean and exact zero are supported. No Actor is queried
    after domain exit, and no future pulse data enters policy observations.
    """
    import jax
    import jax.numpy as j
    from .balance_probe import make_lower_tick
    from .physical_continuation import make_hold_last_tick
    from .live_path_continuation import make_live_path_step
    if mode not in ('mean','zero'):raise ValueError('deterministic comparison only')
    limit=round(horizon_s/env.cc.dt)
    live=make_live_path_step(env,horizon_s=horizon_s)
    held_tick=make_hold_last_tick(make_lower_tick(env,j.zeros(2)),dt=env.cc.dt)
    names=['actual_xy','actual_forward_speed','wheel_speed_proxy','phi','phi_dot',
           'actual_delta','peak_roll','contact_failure','nonfinite','lower_fault',
           'applied_force_impulse','time']

    def pack(logs,ticks,*,prefix,live_active):
        active=logs['active_tick']
        data={k:logs[k] for k in names}
        data.update(active_tick=active,physical_tick=j.where(active,ticks,-1),
            actual_yaw=logs['yaw_unwrapped'],reference=logs['governed'],
            domain_invalid=logs['domain_exit'],prefix_active=active&prefix,
            live_tracking_active=active&live_active,
            original_prefix_reward=j.where(active&prefix,logs['scored_tick_reward'],j.nan),
            original_running_reward=j.where(active&(prefix|live_active),logs['tick_reward'],j.nan),
            nominal_reference=j.where((prefix|live_active),logs['nominal'],j.nan),
            correction_offset=j.where((prefix|live_active),logs['offsets'],j.nan),
            filtered_offset=j.where((prefix|live_active),logs['filtered_offset'],j.nan))
        return data

    def normal(carry,params,path,speed,pulse):
        state=carry.state
        obs,_,fault=env.observation(state,path,speed,horizon_s)
        action=j.zeros(2) if mode=='zero' else actor.apply(params,obs)
        fault=fault|~j.all(j.isfinite(action))
        end,logs=env.policy_step(state.replace(fault=state.fault|fault),action,path,speed,horizon_s,pulse)
        handover=end.domain_exit&~carry.handed_over
        reference=j.where(handover,end.physical.governor.current_reference,carry.held_reference)
        return carry.replace(state=end,handed_over=carry.handed_over|handover,held_reference=reference),pack(
            logs,state.tick+j.arange(4),prefix=j.bool_(True),live_active=j.bool_(False))

    def live_tail(carry,_params,path,speed,pulse):
        start=carry.state.tick
        end,logs=live(carry.state,path,speed,pulse)
        return carry.replace(state=end),pack(logs,start+j.arange(4),prefix=j.bool_(False),live_active=j.bool_(True))

    def held_tail(carry,_params,_path,_speed,pulse):
        def tick(state,_):
            def advance(state):
                end,log=held_tick(state,pulse,state.tick*env.cc.dt,carry.held_reference)
                full=dict(env.zero_log)
                for key in names:full[key]=log[key]
                full.update(active_tick=log['active_tick'],yaw_unwrapped=end.physical.yaw_unwrapped,
                            governed=log['reference'],domain_exit=j.bool_(True),
                            scored_tick_reward=j.float32(0.))
                return end,full
            stopped=state.physical.failed|state.fault|(state.tick>=limit)
            blank=dict(env.zero_log,scored_tick_reward=j.float32(0.))
            return jax.lax.cond(stopped,lambda s:(s,blank),advance,state)
        end,logs=jax.lax.scan(tick,carry.state,None,length=4)
        return carry.replace(state=end),pack(logs,carry.state.tick+j.arange(4),prefix=j.bool_(False),live_active=j.bool_(False))

    # Use shape-compatible blank logs; scored_tick_reward is upper-derived.
    blank=jax.tree.map(lambda x:j.repeat(x[None],4,axis=0),env.zero_log)
    blank['scored_tick_reward']=j.zeros(4)
    blank=pack(blank,j.full(4,-1),prefix=j.bool_(False),live_active=j.bool_(False))

    def step(carry,params,path,speed,pulse,use_live):
        stopped=carry.state.physical.failed|carry.state.fault|(carry.state.tick>=limit)
        def active(carry):
            return jax.lax.cond(carry.state.domain_exit,
                lambda c:jax.lax.cond(use_live,live_tail,held_tail,c,params,path,speed,pulse),
                lambda c:normal(c,params,path,speed,pulse),carry)
        return jax.lax.cond(stopped,lambda c:(c,blank),active,carry)
    return step
