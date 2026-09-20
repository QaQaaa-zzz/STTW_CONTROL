"""STTW three-mode soft-budget reward proposal, version 1.

NOT a repository patch, NOT a vehicle simulation, NOT a safety certificate.
Reference read: STTW_CONTROL commit 68f6a727, alpha in {0, .5, 1}.
Keeps signed real-speed error, committed geometric path and existing recovery
trigger semantics. All coefficients are declared design values, not fitted
from the user's six-row summary.

Dependencies: NumPy. Optional JAX backend: pass xp=jax.numpy; use vmap for batches.
The caller supplies the same pre-action alpha and post-action errors as timed_env.
The caller must stop stepping after true physical termination; timeout is separate.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import NamedTuple
import math
import numpy as np

@dataclass(frozen=True)
class Config:
    dt: float = .005
    gamma: float = .9995
    horizon_s: float = 10.
    primary_rate: float = 8.
    speed_scale: float = .05
    lateral_scale: float = .10
    heading_scale: float = .15
    heading_weight: float = .30
    budget_rate: float = 2.
    relaxed_under: float = .50
    final_speed: float = .05
    relaxed_path: float = .40
    final_path: float = .10
    relaxed_heading: float = .35
    final_heading: float = .15
    over_band: float = .05
    grace_s: float = 1.
    return_s: float = 3.
    hold_s: float = .5
    roll_working: float = .30
    roll_failure: float = .70
    final_roll_rate: float = .30
    roll_rate_weight: float = 1.
    roll_excess_weight: float = 100.
    action_weight: float = .01
    action_delta_weight: float = .02
    overdue_rate: float = 2.
    deadline_penalty: float = 5.
    failure_penalty: float = 200.
    reward_scale: float = .1
    bounded_cost: float = 100.
    alive_rate: float = 1.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not self.grace_s < self.return_s - self.hold_s:
            raise ValueError('need time to shrink bands before final hold')
        if not 0 < self.gamma <= 1:
            raise ValueError('gamma must be in (0,1]')
        if not self.roll_working < self.roll_failure:
            raise ValueError('working and physical roll bounds are distinct')
        if self.relaxed_under < self.final_speed or self.relaxed_path < self.final_path or self.relaxed_heading < self.final_heading:
            raise ValueError('relaxed bands must contain final bands')
        for x in (self.horizon_s,self.grace_s,self.return_s,self.hold_s):
            if not math.isclose(x/self.dt,round(x/self.dt),abs_tol=1e-8):
                raise ValueError('times must align to control ticks')

class State(NamedTuple):
    pending: object
    elapsed_ticks: object
    hold_ticks: object
    credited: object  # diagnostic on-time recovery ONLY; no positive reward
    ever_left: object
    deadline_missed: object
    previous_action: object


def initial_state(*, xp=np):
    return State(xp.asarray(False),xp.asarray(0,dtype=xp.int32),xp.asarray(0,dtype=xp.int32),
                 xp.asarray(False),xp.asarray(False),xp.asarray(False),xp.zeros(2))


def huber(z, *, xp=np):
    z=xp.abs(xp.asarray(z))
    return xp.minimum(z,1.)**2+2.*xp.maximum(z-1.,0.)


def smooth_bound(cost, limit=100., *, xp=np):
    """Stable implementation of limit*cost/(limit+cost), for nonnegative cost.

    Multiplying limit*cost first could overflow. This form stays finite for inf.
    An elementwise monotone transform need NOT preserve cumulative-cost ordering.
    """
    return limit*(1.-limit/(limit+cost))


def bands(alpha, elapsed_s, config=Config(), *, xp=np):
    c=config
    f=xp.clip((elapsed_s-c.grace_s)/(c.return_s-c.hold_s-c.grace_s),0.,1.)
    under=c.final_speed+(c.relaxed_under-c.final_speed)*(1.-alpha)*(1.-f)
    path=c.final_path+(c.relaxed_path-c.final_path)*alpha*(1.-f)
    heading=c.final_heading+(c.relaxed_heading-c.final_heading)*alpha*(1.-f)
    return under,c.over_band,path,heading,f


def within_final(roll,roll_rate,speed_error,lateral_error,heading_error,c,*,xp=np):
    return ((xp.abs(roll)<=c.roll_working)&(xp.abs(roll_rate)<=c.final_roll_rate)
            &(speed_error>=-c.final_speed)&(speed_error<=c.over_band)
            &(xp.abs(lateral_error)<=c.final_path)&(xp.abs(heading_error)<=c.final_heading))


def instantaneous_costs(*, speed_error,lateral_error,heading_error,roll,roll_rate,
                        action,previous_action,alpha,elapsed_s,pending=False,
                        config=Config(),xp=np):
    """Raw, unscaled reward-rate costs. Do NOT add the old precision costs again.

    alpha0: primary path, no continuous underspeed cost inside its allowance.
    alpha1: primary speed, no continuous path cost inside its allowance.
    alpha.5: equal coefficients on the FIXED normalized primary costs.
    Overspeed is never discounted by alpha; secondary objectives incur band costs.
    """
    c=config;u=xp.maximum(-speed_error,0.);o=xp.maximum(speed_error,0.)
    bu,bo,by,bh,f=bands(alpha,elapsed_s,c,xp=xp)
    a=xp.asarray(action);prev=xp.asarray(previous_action)
    costs={
        'underspeed_primary': c.primary_rate*alpha*huber(u/c.speed_scale,xp=xp),
        'overspeed_primary': c.primary_rate*huber(o/c.speed_scale,xp=xp),
        'path_primary': c.primary_rate*(1.-alpha)*(huber(lateral_error/c.lateral_scale,xp=xp)
                            +c.heading_weight*huber(heading_error/c.heading_scale,xp=xp)),
        'underspeed_budget': c.budget_rate*huber(xp.maximum(u-bu,0.)/c.speed_scale,xp=xp),
        'overspeed_budget': c.budget_rate*huber(xp.maximum(o-bo,0.)/c.speed_scale,xp=xp),
        'path_budget': c.budget_rate*huber(xp.maximum(xp.abs(lateral_error)-by,0.)/c.lateral_scale,xp=xp),
        'heading_budget': c.budget_rate*c.heading_weight*huber(xp.maximum(xp.abs(heading_error)-bh,0.)/c.heading_scale,xp=xp),
        'roll_excess': c.roll_excess_weight*xp.maximum(xp.abs(roll)-c.roll_working,0.)**2,
        'roll_rate': c.roll_rate_weight*roll_rate**2,
        'action': c.action_weight*xp.sum(a*a),
        'action_delta': c.action_delta_weight*xp.sum((a-prev)**2),
        'return_overdue': c.overdue_rate*xp.asarray(pending)*(elapsed_s>c.return_s+1e-9),
    }
    return costs,{'under_band':bu,'over_band':bo,'path_band':by,'heading_band':bh,'shrink_fraction':f}


def transition(state,*,speed_error,lateral_error,heading_error,roll,roll_rate,
               action,alpha,enabled,physical_failed=False,recovery_trigger=False,
               clock_from_departure=True,config=Config(),xp=np):
    """Pure NumPy/JAX transition; finite arrays and discrete alpha are required.

    Existing task phase semantics: scenario1 uses published recovery_trigger;
    other scenarios use observed final-band departure after initialization.
    Never reset this clock just because alpha changes or another command arrives.
    Caller must keep geometric projection, command timing and physical limits.
    Invalid observations/actions are a true failure, not silently free tracking.
    The reward calculation does not itself generate any physical trajectory.
    """
    c=config
    vals=[xp.asarray(v) for v in (speed_error,lateral_error,heading_error,roll,roll_rate,alpha)]
    a=xp.asarray(action)
    finite=xp.all(xp.stack([xp.all(xp.isfinite(x)) for x in (*vals,a)]))
    alpha_valid=(vals[-1]==0.)|(vals[-1]==.5)|(vals[-1]==1.)
    failed=xp.asarray(physical_failed)|~finite|~alpha_valid|(xp.abs(vals[3])>c.roll_failure)
    ev,ey,ep,phi,rate,alpha=[xp.nan_to_num(x,nan=0.,posinf=0.,neginf=0.) for x in vals]
    a=xp.nan_to_num(a,nan=0.,posinf=0.,neginf=0.)
    final=within_final(phi,rate,ev,ey,ep,c,xp=xp)&~failed
    on=xp.asarray(enabled)&~failed
    left=on&~final
    armed=(state.pending|xp.asarray(recovery_trigger)|(left&xp.asarray(clock_from_departure)))&~failed
    elapsed=xp.where(armed,state.elapsed_ticks+on.astype(xp.int32),0)
    hold=xp.where(on&final,state.hold_ticks+1,0)
    hold_required=round(c.hold_s/c.dt);deadline=round(c.return_s/c.dt)
    missed=state.deadline_missed|(armed&on&(elapsed>deadline))
    completed=armed&on&final&(hold>=hold_required)
    pending=armed&~completed
    new=State(pending,xp.where(pending,elapsed,0),xp.minimum(hold,hold_required),
              (state.credited|(completed&~missed))&~failed,state.ever_left|left,
              missed,a)
    # Cost uses the elapsed duration of this transition, BEFORE completion resets.
    raw,band_info=instantaneous_costs(speed_error=ev,lateral_error=ey,heading_error=ep,
        roll=phi,roll_rate=rate,action=a,previous_action=state.previous_action,
        alpha=alpha,elapsed_s=elapsed*c.dt,pending=pending,config=c,xp=xp)
    cost=sum(raw.values())
    arithmetic_bad=~xp.isfinite(cost)
    failed=failed|arithmetic_bad
    new=new._replace(pending=new.pending&~failed,credited=new.credited&~failed)
    bounded=smooth_bound(cost,c.bounded_cost,xp=xp)
    factor=c.bounded_cost/(c.bounded_cost+cost)
    # For finite physical values the factor partitions G(sum costs) exactly.
    parts={name:xp.where(failed,0.,-c.dt*c.reward_scale*value*factor) for name,value in raw.items()}
    parts['alive']=xp.where(failed,0.,c.dt*c.reward_scale*c.alive_rate)
    parts['deadline']=xp.where(failed,0.,-c.deadline_penalty*(missed&~state.deadline_missed))
    parts['failure']=xp.where(failed,-c.failure_penalty,0.)
    reward=xp.where(failed,-c.failure_penalty,c.dt*c.reward_scale*(c.alive_rate-bounded)+parts['deadline'])
    diagnostics={**band_info,'raw_cost':cost,'bounded_cost':bounded,
                 'bound_slope':factor**2,'final_now':final,
                 'on_time_recovered':new.credited&~new.pending&~new.deadline_missed&~failed,
                 'invalid_input':~finite|~alpha_valid|arithmetic_bad,'physical_failed':failed}
    return new,reward,parts,raw,diagnostics


def validate_inputs(alpha, action):
    """Host-side shape/enum check before JIT or direct standalone calls."""
    if alpha not in (0.,.5,1.):raise ValueError('alpha must be exactly 0, .5 or 1')
    a=np.asarray(action)
    if a.shape!=(2,) or not np.isfinite(a).all() or np.any(np.abs(a)>1.+1e-7):
        raise ValueError('need two finite normalized residual actions in [-1,1]')
