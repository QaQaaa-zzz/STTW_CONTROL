"""Independent v/yaw-rate requests; schedules are state, not disturbance labels."""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jp

@dataclass(frozen=True)
class MotionCommands:
    residual_gate_min: float | None=None  # None preserves legacy constant authority
    segments: int=12
    duration_min: float=1.
    duration_max: float=3.
    speed_min: float=1.2
    speed_max: float=2.8
    yaw_max: float=1.
    dynamic_alpha: bool=True
    fixed: tuple | None=None  # rows: seconds, requested m/s, requested world yaw rad/s, alpha
    speed_scale: float=.2
    yaw_scale: float=.2
    roll_working_limit: float=.3
    attitude_weight: float=100.
    rate_weight: float=1.
    tracking_scale: float=.3
    tracking_priority_ratio: float=100.  # endpoint speed:yaw ratio; legacy default

    tolerance_penalty_rate: float=0.  # optional bounded extra cost [reward/s]
    speed_tolerance: float=.5  # m/s, on the relaxed speed objective
    yaw_tolerance: float=.1  # rad/s, on the relaxed yaw-rate objective
    speed_excess_scale: float=.1  # excess giving half the maximum extra cost
    yaw_excess_scale: float=.05
    yaw_tracking_reward_rate: float=0.  # optional positive reward/s, independent of alpha
    yaw_tracking_reward_scale: float=.1  # rad/s; reward falls to exp(-1) here

    def __post_init__(self):
        if self.residual_gate_min is not None and (not math.isfinite(self.residual_gate_min) or not 0<=self.residual_gate_min<=1):
            raise ValueError('residual_gate_min must be in [0,1] or None')
        if type(self.segments) is not int or self.segments<2:raise ValueError('segments >=2 required')
        if not 0<self.duration_min<=self.duration_max or not 0<self.speed_min<=self.speed_max:raise ValueError('invalid command ranges')
        for x in (self.duration_min,self.duration_max,self.speed_min,self.speed_max,self.yaw_max,self.speed_scale,self.yaw_scale,self.roll_working_limit,self.attitude_weight,self.rate_weight,self.tracking_scale):
            if not math.isfinite(x) or x<=0:raise ValueError('invalid command scalar')
        if not math.isfinite(self.tracking_priority_ratio) or self.tracking_priority_ratio<1.:
            raise ValueError('tracking_priority_ratio must be finite and >= 1')
        if not math.isfinite(self.tolerance_penalty_rate) or self.tolerance_penalty_rate<0:
            raise ValueError('invalid tolerance penalty rate')
        for value in (self.speed_tolerance,self.yaw_tolerance,self.speed_excess_scale,self.yaw_excess_scale):
            if not math.isfinite(value) or value<=0:raise ValueError('invalid tolerance threshold/scale')
        if not math.isfinite(self.yaw_tracking_reward_rate) or self.yaw_tracking_reward_rate<0:
            raise ValueError('invalid yaw tracking reward rate')
        if not math.isfinite(self.yaw_tracking_reward_scale) or self.yaw_tracking_reward_scale<=0:
            raise ValueError('invalid yaw tracking reward scale')
        if self.fixed is not None:
            if not self.fixed or self.fixed[0][0]!=0:raise ValueError('fixed commands must start at zero')
            last=-1.
            for row in self.fixed:
                if len(row)!=4 or not all(math.isfinite(x) for x in row) or row[0]<=last or row[1]<=0 or not 0<=row[3]<=1:raise ValueError('invalid command row')
                last=row[0]

def gated_action(action,alpha,c,*,xp=jp):
    """Scale raw normalized action once, using the alpha BEFORE the transition."""
    if c.residual_gate_min is None:return action
    gain=c.residual_gate_min+(1-c.residual_gate_min)*xp.abs(2*xp.asarray(alpha)-1)
    return xp.asarray(action)*gain[...,None]


def sample_schedule(key,c,initial_speed):
    if c.fixed is not None:return jp.asarray(c.fixed)
    kt,kv,kr,ka=jax.random.split(key,4)
    t=jp.concatenate([jp.zeros(1),jp.cumsum(jax.random.uniform(kt,(c.segments-1,),minval=c.duration_min,maxval=c.duration_max))])
    v=jax.random.uniform(kv,(c.segments,),minval=c.speed_min,maxval=c.speed_max).at[0].set(initial_speed)
    r=jax.random.uniform(kr,(c.segments,),minval=-c.yaw_max,maxval=c.yaw_max).at[0].set(0.)
    a=jax.random.uniform(ka,(c.segments,))
    return jp.stack([t,v,r,a],axis=1)

def reference_at(seconds,schedule):
    i=jp.maximum(jp.sum(seconds>=schedule[:,0])-1,0)
    return schedule[i,1:]

def tracking_weights(alpha,c):
    """Centered ratio**(alpha-.5) weights, before normalized squared errors.

    The geometric center stays at tracking_scale for either task. Equal
    weights do not imply equal penalties when the normalized errors differ.
    Using sqrt(ratio) keeps the legacy ratio=100 operation exactly 10**(2a-1).
    """
    base=math.sqrt(c.tracking_priority_ratio)
    return c.tracking_scale*base**(2*alpha-1),c.tracking_scale*base**(1-2*alpha)


def reward_terms(roll,rate,speed_error,yaw_error,action,alpha,c,*,xp=jp):
    speed_weight,yaw_weight=tracking_weights(alpha,c)
    costs=dict(attitude=c.attitude_weight*xp.maximum(xp.abs(roll)-c.roll_working_limit,0.)**2,
                roll_rate=c.rate_weight*rate**2,
                speed=speed_weight*(speed_error/c.speed_scale)**2,
                yaw=yaw_weight*(yaw_error/c.yaw_scale)**2,
                action=.01*xp.sum(action**2,axis=-1))
    if c.tolerance_penalty_rate>0:
        # No new cost inside tolerance; finite upper cost for abrupt requests.
        # 1-1/(1+z^2) avoids inf/inf for extreme finite simulation errors.
        def excess_cost(error,tolerance,scale):
            z=xp.maximum(xp.abs(error)-tolerance,0.)/scale
            return c.tolerance_penalty_rate*(1.-1./(1.+z*z))
        costs['speed_tolerance']=(1-alpha)*excess_cost(speed_error,c.speed_tolerance,c.speed_excess_scale)
        costs['yaw_tolerance']=alpha*excess_cost(yaw_error,c.yaw_tolerance,c.yaw_excess_scale)
    return costs


def yaw_tracking_bonus(yaw_error,c,*,xp=jp):
    """Positive accuracy bonus; original penalties remain unchanged."""
    return c.yaw_tracking_reward_rate*xp.exp(-(yaw_error/c.yaw_tracking_reward_scale)**2)


def signed_reward_components(roll,rate,speed_error,yaw_error,action,alpha,c,dt,alive_rate,failure_penalty,failed,*,xp=jp):
    """Signed per-transition terms, including terminal replacement, for logging."""
    costs=reward_terms(roll,rate,speed_error,yaw_error,action,alpha,c,xp=xp)
    parts={k:xp.where(failed,0.,-dt*v) for k,v in costs.items()}
    if c.yaw_tracking_reward_rate>0:
        parts['yaw_tracking']=xp.where(failed,0.,dt*yaw_tracking_bonus(yaw_error,c,xp=xp))
    parts['alive']=xp.where(failed,0.,dt*alive_rate)
    parts['failure']=xp.where(failed,-failure_penalty,0.)
    return parts
