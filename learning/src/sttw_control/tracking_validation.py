"""Fixed-horizon geometric validation; nominal preservation precedes reward rank."""
import math
import numpy as np


def rank_tracking_candidate(candidate, baseline, *, speed_slack=.05, path_slack=.05):
    """Rank is diagnostic. Gate failures are never relabelled as task success."""
    keys = ('failed', 'nominal_failed', 'nominal_radial_rmse', 'nominal_speed_rmse',
            'final_tracking_hold', 'deadline_missed', 'speed_budget_fraction',
            'path_budget_fraction', 'episode_return')
    for result in (candidate, baseline):
        if any(k not in result for k in keys):
            return None, 'missing geometric validation evidence'
        if any(not np.isfinite(np.asarray(result[k], dtype=float)).all() for k in keys):
            return None, 'nonfinite geometric validation evidence'
    x = lambda r,k:np.asarray(r[k], dtype=float)
    regress = ((x(candidate,'nominal_radial_rmse') > x(baseline,'nominal_radial_rmse')+path_slack)
               | (x(candidate,'nominal_speed_rmse') > x(baseline,'nominal_speed_rmse')+speed_slack))
    score = (int(x(candidate,'failed').sum()+x(candidate,'nominal_failed').sum()),
             int(regress.sum()), int((1-x(candidate,'final_tracking_hold')).sum()),
             int(x(candidate,'deadline_missed').sum()),
             float(np.mean(x(candidate,'speed_budget_fraction')+x(candidate,'path_budget_fraction'))),
             -float(np.mean(x(candidate,'episode_return'))))
    return score, ('development gates passed; not a safety proof' if not any(score[:4])
                   else 'diagnostic candidate only: failed safety/nominal/hold/deadline gate')


def make_tracking_validator(env, actor, scale, config):
    import jax
    import jax.numpy as jp
    from .tracking_reward import tolerances
    c, rc = env.config, env.config.tracking_reward
    dt = c.controller.dt
    for a,b in ((config.validation_hold_seconds,rc.hold_seconds),
                (config.validation_path_tolerance,rc.return_lateral_tolerance),
                (config.validation_heading_tolerance,rc.return_heading_tolerance),
                (config.validation_speed_tolerance,rc.return_speed_tolerance)):
        if not math.isclose(a,b,rel_tol=0,abs_tol=1e-8):
            raise ValueError('training and validation tracking hold/tolerances must agree')
    cases = config.validation_events
    if not cases:
        raise ValueError('geometric validation requires explicit fixed event cases, including nominal')
    if any(case['start']+case['duration']+rc.return_deadline_seconds > c.horizon_seconds+1e-8 for case in cases):
        raise ValueError('fixed validation must contain the declared return observation window')
    if any(not math.isclose(case[k]/dt,round(case[k]/dt),abs_tol=1e-7)
           for case in cases for k in ('start','duration')):
        raise ValueError('validation events must align to control ticks')
    choices, seeds = c.priority.validation_alphas, config.validation_seeds
    keys = jp.stack([jax.random.PRNGKey(s) for a in choices for s in seeds for case in cases])
    alphas = jp.asarray([a for a in choices for s in seeds for case in cases])
    events = jp.asarray([[round(case['start']/dt), round((case['start']+case['duration'])/dt),
        case.get('steer_rate',0.), case.get('force',0.), float(case.get('waveform','constant')=='half_sine'),
        case.get('rear_torque',0.)] for a in choices for s in seeds for case in cases])
    n = len(keys)
    needed = max(1,int(math.ceil(rc.hold_seconds/dt-1e-9)))
    reset, step = jax.vmap(env.reset), jax.vmap(env.step)

    @jax.jit
    def validate(params, zero=False):
        initial = jax.vmap(env.set_priority)(reset(keys),alphas).replace(event=events)
        nominal = initial.replace(event=events.at[:,2:4].set(0.).at[:,5].set(0.))
        state = jax.tree.map(lambda a,b:jp.concatenate((a,b),axis=0),initial,nominal)
        event_end = jp.concatenate((events[:,1],events[:,1]))
        bv, by = tolerances(jp.concatenate((alphas,alphas)),rc,xp=jp)
        def tick(carry,_):
            s, hold = carry
            active = ~s.done
            action = actor.apply(params['actor'],s.obs/scale)
            new = step(s,jp.where(zero,jp.zeros_like(action),action))
            path = jax.vmap(env.path_features)(new.pose)
            speed = jp.sum(new.data.qvel[:,:3]*new.data.xmat[:,env.bundle.chassis,:,0],axis=-1)-c.speed_reference
            valid = (active & ~new.terminated & (s.tick>=event_end)
                     & (jp.abs(path[:,0])<=rc.return_lateral_tolerance)
                     & (jp.abs(path[:,1])<=rc.return_heading_tolerance)
                     & (jp.abs(speed)<=rc.return_speed_tolerance)
                     & (jp.abs(new.measurement[:,0])<=rc.roll_working_limit)
                     & (jp.abs(new.measurement[:,1])<=rc.return_roll_rate_tolerance))
            hold = jp.where(valid,hold+1,0)
            extra = jp.abs(path[:n,0]-path[n:,0])
            return (new,hold),(active,path[:,0]**2,path[:,1]**2,speed**2,new.reward,
                jp.abs(speed)>bv,jp.abs(path[:,0])>by,extra,jp.abs(new.measurement[:,0]))
        (last,hold),rows = jax.lax.scan(tick,(state,jp.zeros(2*n,dtype=jp.int32)),None,length=env.horizon)
        active,y,h,v,rewards,vex,yex,extra,roll = rows
        count = jp.maximum(jp.sum(active,axis=0),1)
        avg = lambda z:jp.sum(jp.where(active,z,0),axis=0)/count
        complete = (hold>=needed)&~last.terminated&(last.tick>=env.horizon)
        present = jp.any(events[:,2:4]!=0,axis=1)|(events[:,5]!=0)
        budget = jp.maximum(0,(env.horizon-events[:,1])*dt)
        settling = jp.where(complete[:n],(last.tick[:n]-hold[:n]+needed-events[:,1])*dt,budget)
        return dict(radial_rmse=jp.sqrt(avg(y))[:n], heading_rmse=jp.sqrt(avg(h))[:n],
            speed_rmse=jp.sqrt(avg(v))[:n], nominal_radial_rmse=jp.sqrt(avg(y))[n:],
            nominal_speed_rmse=jp.sqrt(avg(v))[n:], failed=last.terminated[:n],
            nominal_failed=last.terminated[n:], event_present=present,
            post_event_hold_complete=complete[:n], final_tracking_hold=complete[:n],
            nominal_final_tracking_hold=complete[n:],
            task_recovered=last.tracking.ever_departed[:n]&complete[:n]&~last.tracking.deadline_missed[:n],
            ever_departed=last.tracking.ever_departed[:n], deadline_missed=last.tracking.deadline_missed[:n],
            settling_seconds=settling, post_event_peak=jp.max(jp.where(active[:,:n],extra,0),axis=0),
            steps=count[:n], nominal_steps=count[n:], speed_budget_fraction=avg(vex)[:n],
            path_budget_fraction=avg(yex)[:n], roll_peak=jp.max(jp.where(active,roll,0),axis=0)[:n],
            episode_return=jp.sum(jp.where(active,rewards,0),axis=0)[:n],
            nominal_episode_return=jp.sum(jp.where(active,rewards,0),axis=0)[n:])
    return validate
