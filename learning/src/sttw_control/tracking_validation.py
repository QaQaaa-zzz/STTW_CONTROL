"""Fixed-horizon paired geometric validation using the training return state.

No horizon extension, post-failure zero padding, or yaw-rate-as-path criterion.
"""
import jax
import jax.numpy as jp
import numpy as np
from .tracking_reward import tolerances


def make_tracking_validator(env, actor, scale, config):
    c=env.config; tc=c.tracking; dt=c.controller.dt
    cases=config.validation_events or ({'start':0.,'duration':dt},)
    for case in cases:
        if any(not np.isclose(case[k]/dt,round(case[k]/dt),rtol=0,atol=1e-7)
               for k in ('start','duration')):
            raise ValueError('validation events must align to control ticks')
        present=any(case.get(k,0.)!=0 for k in ('steer_rate','force','rear_torque'))
        if present and case['start']+case['duration']+tc.return_seconds>c.horizon_seconds+1e-7:
            raise ValueError('declared return window does not fit the fixed episode')
    choices=c.priority.validation_alphas
    keys=jp.stack([jax.random.PRNGKey(seed) for alpha in choices
                   for seed in config.validation_seeds for case in cases])
    alphas=jp.asarray([alpha for alpha in choices for seed in config.validation_seeds for case in cases])
    event_rows=np.asarray([[round(case['start']/dt),round((case['start']+case['duration'])/dt),
                       case.get('steer_rate',0.),case.get('force',0.),
                       float(case.get('waveform','constant')=='half_sine'),case.get('rear_torque',0.)]
                      for alpha in choices for seed in config.validation_seeds for case in cases],dtype=np.float32)
    # Assemble the paired panel explicitly before tracing; no scatter updates
    # to closed-over event constants and no concatenation of aliased states.
    nominal_rows=event_rows.copy();nominal_rows[:,[2,3,5]]=0.
    all_events=jp.asarray(np.concatenate((event_rows,nominal_rows),axis=0))
    all_keys=jp.concatenate((keys,keys),axis=0)
    all_alphas=jp.concatenate((alphas,alphas),axis=0)
    present=jp.asarray(np.any(event_rows[:,[2,3,5]]!=0.,axis=1))
    n=len(keys); reset=jax.vmap(env.reset); step=jax.vmap(env.step)

    @jax.jit
    def validate(params, zero=False):
        state=jax.vmap(env.set_priority)(reset(all_keys),all_alphas).replace(event=all_events)
        def tick(state,_):
            active=~state.done
            action=actor.apply(params['actor'],state.obs/scale)
            nxt=step(state,jp.where(zero,jp.zeros_like(action),action))
            features=jax.vmap(env.path_features)(nxt.pose)
            speed=jp.sum(nxt.data.qvel[:,:3]*nxt.data.xmat[:,env.bundle.chassis,:,0],axis=-1)
            ev=speed-jax.vmap(env.speed_command)(state.tick)
            bv,by=tolerances(state.priority_alpha,tc)
            mature=active & (state.tick*dt>=tc.start_seconds)
            row=(active,features[:,0]**2,ev**2,features[:,1]**2,nxt.reward,
                 jp.where(active,jp.abs(nxt.measurement[:,0]),0.),
                 mature,mature & (jp.abs(ev)>bv),mature & (jp.abs(features[:,0])>by),
                 jp.where(active[:n],jp.abs(features[:n,0]-features[n:,0]),0.))
            return nxt,row
        last,rows=jax.lax.scan(tick,state,None,length=env.horizon)
        active,lateral,speed,heading,reward,roll,mature,speed_out,path_out,extra=rows
        count=jp.maximum(jp.sum(active,axis=0),1)
        mature_count=jp.maximum(jp.sum(mature,axis=0),1)
        def rmse(error):return jp.sqrt(jp.sum(jp.where(active,error,0.),axis=0)/count)
        rs=last.tracking_state
        complete=((last.tick>=env.horizon)&~last.terminated&~rs.pending
                  &(rs.hold+1e-6>=tc.hold_seconds))
        qualified=complete&~rs.deadline_missed
        return dict(return_ever_left=rs.ever_left[:n],return_credited=rs.credited[:n],
                    return_pending=rs.pending[:n],return_hold_seconds=rs.hold[:n],
                    radial_rmse=rmse(lateral)[:n],speed_rmse=rmse(speed)[:n],
                    heading_rmse=rmse(heading)[:n],nominal_radial_rmse=rmse(lateral)[n:],
                    nominal_speed_rmse=rmse(speed)[n:],nominal_heading_rmse=rmse(heading)[n:],
                    failed=last.terminated[:n],nominal_failed=last.terminated[n:],
                    episode_return=jp.sum(jp.where(active,reward,0.),axis=0)[:n],
                    nominal_episode_return=jp.sum(jp.where(active,reward,0.),axis=0)[n:],
                    terminal_tracking_hold=complete[:n],nominal_terminal_tracking_hold=complete[n:],
                    post_event_hold_complete=qualified[:n],event_present=present,
                    return_deadline_missed=rs.deadline_missed[:n],
                    nominal_return_deadline_missed=rs.deadline_missed[n:],
                    recovered_after_excursion=qualified[:n]&rs.ever_left[:n]&rs.credited[:n],
                    maintained_without_excursion=qualified[:n]&~rs.ever_left[:n],
                    roll_peak=jp.max(roll,axis=0)[:n],nominal_roll_peak=jp.max(roll,axis=0)[n:],
                    speed_tolerance_exceed_fraction=(jp.sum(speed_out,axis=0)/mature_count)[:n],
                    path_tolerance_exceed_fraction=(jp.sum(path_out,axis=0)/mature_count)[:n],
                    post_event_peak=jp.max(extra,axis=0),steps=count[:n],nominal_steps=count[n:])
    return validate
