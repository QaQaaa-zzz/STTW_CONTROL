"""Independent v/yaw-rate requests; schedules are state, not disturbance labels."""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jp

@dataclass(frozen=True)
class MotionCommands:
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

    def __post_init__(self):
        if type(self.segments) is not int or self.segments<2:raise ValueError('segments >=2 required')
        if not 0<self.duration_min<=self.duration_max or not 0<self.speed_min<=self.speed_max:raise ValueError('invalid command ranges')
        for x in (self.duration_min,self.duration_max,self.speed_min,self.speed_max,self.yaw_max,self.speed_scale,self.yaw_scale,self.roll_working_limit,self.attitude_weight,self.rate_weight,self.tracking_scale):
            if not math.isfinite(x) or x<=0:raise ValueError('invalid command scalar')
        if self.fixed is not None:
            if not self.fixed or self.fixed[0][0]!=0:raise ValueError('fixed commands must start at zero')
            last=-1.
            for row in self.fixed:
                if len(row)!=4 or not all(math.isfinite(x) for x in row) or row[0]<=last or row[1]<=0 or not 0<=row[3]<=1:raise ValueError('invalid command row')
                last=row[0]

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

def reward_terms(roll,rate,speed_error,yaw_error,action,alpha,c):
    return dict(attitude=c.attitude_weight*jp.maximum(jp.abs(roll)-c.roll_working_limit,0.)**2,
                roll_rate=c.rate_weight*rate**2,
                speed=c.tracking_scale*10.**(2*alpha-1)*(speed_error/c.speed_scale)**2,
                yaw=c.tracking_scale*10.**(1-2*alpha)*(yaw_error/c.yaw_scale)**2,
                action=.01*jp.sum(action**2))
