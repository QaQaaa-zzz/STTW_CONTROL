"""Priority V2 scalar-compatible costs and causal, single-maneuver return contract.

Only currently published reference inputs enter the state machine. Repeated or
 overlapping maneuvers are outside this contract and never reset existing debt.
"""
from __future__ import annotations

import jax.numpy as jp
import numpy as np
from flax import struct


def huber(z, *, xp=jp):
    z = xp.abs(z)
    return xp.where(z <= 1., z*z, 2.*z-1.)


def reward_terms(*, alpha, speed_error, lateral_error, heading_error, roll=0.,
                 roll_rate=0., action=(0., 0.), previous_action=(0., 0.),
                 relaxation=1., dt=.005, failed=False, missed_deadline_now=False,
                 terminal_incomplete_now=False, task_penalty_already_paid=False, xp=jp):
    """Exact finite-input reference arithmetic; invalid runtime data fail closed.

    NumPy rejects invalid alpha/actions/timing. Traced JAX values cannot raise a
    Python exception: invalid values instead produce the failure replacement.
    """
    if isinstance(alpha, (int, float, np.number)) and alpha not in (0., .5, 1.):
        raise ValueError('alpha must be exactly 0, 0.5, or 1')
    a, prev = xp.asarray(action), xp.asarray(previous_action)
    if a.shape != (2,) or prev.shape != (2,):
        raise ValueError('actions must contain steer and rear channels')
    values = xp.asarray([speed_error,lateral_error,heading_error,roll,roll_rate,relaxation,dt])
    valid = (xp.all(xp.isfinite(values)) & xp.all(xp.isfinite(a)) & xp.all(xp.isfinite(prev))
             & ((alpha == 0.) | (alpha == .5) | (alpha == 1.))
             & (relaxation >= 0.) & (relaxation <= 1.) & (dt > 0.)
             & xp.all(xp.abs(a) <= 1.) & xp.all(xp.abs(prev) <= 1.))
    if xp is np and not valid:
        raise ValueError('invalid or nonfinite reward input')
    ev,ey,ep,r,rr,q,step = xp.nan_to_num(values,nan=0.,posinf=0.,neginf=0.)
    a=xp.nan_to_num(a,nan=0.,posinf=0.,neginf=0.)
    prev=xp.nan_to_num(prev,nan=0.,posinf=0.,neginf=0.)
    wp=xp.where(alpha==0.,8.,xp.where(alpha==.5,4.,.1))
    wv=xp.where(alpha==0.,.1,xp.where(alpha==.5,4.,8.))
    bu=.05+.45*(1-alpha)*q
    by=.10+.30*alpha*q
    bp=.15+.20*alpha*q
    h=lambda z:huber(z,xp=xp)
    raw={'path':wp*(h(ey/.10)+.3*h(ep/.15)), 'speed':wv*h(ev/.05),
         'under_budget':2*h(xp.maximum(xp.maximum(-ev,0)-bu,0)/.05),
         'over_budget':2*h(xp.maximum(xp.maximum(ev,0)-.05,0)/.05),
         'path_budget':2*h(xp.maximum(xp.abs(ey)-by,0)/.10),
         'heading_budget':.6*h(xp.maximum(xp.abs(ep)-bp,0)/.15),
         'roll_excess':20*h(xp.maximum(xp.abs(r)-.30,0)/.10),
         'roll_rate':.2*h(rr), 'action':.01*xp.sum(a*a),
         'action_delta':.02*xp.sum((a-prev)**2)}
    failure=xp.asarray(failed) | ~valid
    parts={k:xp.where(failure,0.,-.1*step*v) for k,v in raw.items()}
    miss=(xp.asarray(missed_deadline_now)|xp.asarray(terminal_incomplete_now)) & ~xp.asarray(task_penalty_already_paid)
    parts['task_incomplete']=xp.where(failure,0.,xp.where(miss,-20.,0.))
    parts['failure']=xp.where(failure,-200.,0.)
    return {'raw_costs':raw,'reward_parts':parts,'reward':sum(parts.values()),
            'bands':{'underspeed':bu,'overspeed':xp.asarray(.05),'lateral':by,'heading':bp}}


@struct.dataclass
class ReturnState:
    q: object
    exit_valid: object
    exit_progress: object
    exit_time: object
    deadline: object
    return_start: object
    hold: object
    deadline_missed: object
    penalty_paid: object
    task_complete: object
    late_but_finally_recovered: object
    insufficient_return_window: object
    seen_maneuver: object
    stable_hold: object
    stable_start_progress: object
    physical_failed: object
    deadline_checked: object


def initial_state(*, xp=jp):
    z=lambda:xp.asarray(0.,dtype=xp.float32)
    f=lambda:xp.asarray(False)
    return ReturnState(z(),f(),z(),z(),z(),xp.asarray(-1.,dtype=xp.float32),z(),
                       f(),f(),f(),f(),f(),f(),z(),z(),f(),f())


def advance(state, *, t, dt, episode_end, reference_speed, reference_yaw,
            previous_reference_speed, reference_progress, actual_progress,
            speed_error, lateral_error, heading_error, roll, roll_rate,
            failed=False, published_yaw_request=None, xp=jp):
    """Advance once per actual control transition; t is the post-step time.

    Deadline is inclusive: a completed hold at D is on time. One task miss is
    charged at D if incomplete, or at the fixed horizon, and never refunded.
    task_complete describes the *current* final hold, not historical success.
    """
    values=xp.asarray([t,dt,episode_end,reference_speed,reference_yaw,
                       previous_reference_speed,reference_progress,actual_progress,
                       speed_error,lateral_error,heading_error,roll,roll_rate])
    finite=xp.all(xp.isfinite(values)) & (dt>0.)
    physical=state.physical_failed | xp.asarray(failed) | ~finite
    curvature=xp.abs(reference_yaw)/xp.maximum(xp.abs(reference_speed),1e-6)
    published_yaw=reference_yaw if published_yaw_request is None else published_yaw_request
    physical=physical | ~xp.isfinite(published_yaw)
    turning=(xp.abs(published_yaw)>0.) | (curvature>.05)
    seen=state.seen_maneuver | turning
    stable=(curvature<=.05) & (xp.abs(reference_speed-previous_reference_speed)/xp.maximum(dt,1e-8)<=.10) & finite
    tracking_exit=seen & ~state.exit_valid & stable
    stable_hold=xp.where(tracking_exit,state.stable_hold+dt,0.)
    start_progress=xp.where(tracking_exit & (state.stable_hold<=0.),reference_progress,state.stable_start_progress)
    confirmed=tracking_exit & (stable_hold>=.25-1e-6)
    exit_valid=state.exit_valid | confirmed
    exit_time=xp.where(confirmed,t,state.exit_time)
    exit_progress=xp.where(confirmed,start_progress,state.exit_progress)
    deadline=xp.where(confirmed,xp.minimum(episode_end,t+3.),state.deadline)
    start_now=exit_valid & (state.return_start<0.) & ((actual_progress>=exit_progress) | (t>=exit_time+1.-1e-6))
    return_start=xp.where(start_now,xp.minimum(t,exit_time+1.),state.return_start)
    denominator=deadline-.5-return_start
    insufficient=state.insufficient_return_window | (exit_valid & (return_start>=0.) & (denominator<=0.))
    fraction=xp.clip((deadline-.5-t)/xp.maximum(denominator,1e-8),0.,1.)
    q=xp.where(~seen,0.,xp.where(~exit_valid | (return_start<0.),1.,xp.where(denominator<=0.,0.,fraction)))
    in_band=(xp.abs(speed_error)<=.05) & (xp.abs(lateral_error)<=.10) & (xp.abs(heading_error)<=.15) & (xp.abs(roll)<=.30) & (xp.abs(roll_rate)<=.30) & ~physical
    hold=xp.where(in_band,state.hold+dt,0.)
    completed_segment=~seen | (exit_valid & (actual_progress>=exit_progress+1.))
    complete=completed_segment & (hold>=.5-1e-6) & ~physical
    deadline_reached=exit_valid & (t>=deadline-1e-6)
    due=deadline_reached & ~state.deadline_checked & ~complete
    checked=state.deadline_checked | deadline_reached
    missed=state.deadline_missed | due
    terminal_incomplete=(t>=episode_end-1e-6) & ~complete
    new_miss=(due | terminal_incomplete) & ~state.penalty_paid
    late=state.late_but_finally_recovered | (missed & complete)
    new=ReturnState(q,exit_valid,exit_progress,exit_time,deadline,return_start,hold,
                    missed,state.penalty_paid|new_miss,complete,late,insufficient,
                    seen,stable_hold,start_progress,physical,checked)
    events={'new_task_miss':new_miss,'missed_deadline_now':due,
            'terminal_incomplete_now':terminal_incomplete,'physical_failed':physical}
    return new,events


def observation_context(state, *, t, episode_end, actual_progress, xp=jp):
    return xp.asarray([state.q,xp.where(state.exit_valid,xp.maximum(state.deadline-t,0.),0.),
                       state.exit_valid,xp.where(state.exit_valid,actual_progress-state.exit_progress,0.),
                       xp.maximum(episode_end-t,0.)],dtype=xp.float32)
