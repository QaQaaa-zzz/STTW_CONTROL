"""Reset-sampled event schedules; hidden disturbances are not Actor inputs."""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jp

@dataclass(frozen=True)
class RandomEvents:
    start_min: float=4.
    start_max: float=7.
    duration_min: float=.2
    duration_max: float=.6
    steer_min: float=.2
    steer_max: float=.8
    force_min: float=2.
    force_max: float=5.
    nominal_probability: float=.2
    sine_probability: float=.5

    def __post_init__(self):
        for low,high in [(self.start_min,self.start_max),(self.duration_min,self.duration_max),(self.steer_min,self.steer_max),(self.force_min,self.force_max)]:
            if not math.isfinite(low+high) or low<0 or high<low:raise ValueError('invalid random event range')
        if self.duration_min<=0 or not 0<=self.nominal_probability<1 or not 0<=self.sine_probability<=1:raise ValueError('invalid event duration/probability')


def sample_event(key,c,dt):
    u=jax.random.uniform(key,(7,))
    start=jp.round((c.start_min+(c.start_max-c.start_min)*u[0])/dt)
    duration=jp.maximum(1,jp.round((c.duration_min+(c.duration_max-c.duration_min)*u[1])/dt))
    sign=jp.where(u[2]<.5,-1.,1.)
    active=u[3]>=c.nominal_probability
    steer=active&(u[3]<(1+c.nominal_probability)/2)
    force=active&~steer
    return jp.array([start,start+duration,
                     jp.where(steer,sign*(c.steer_min+u[4]*(c.steer_max-c.steer_min)),0.),
                     jp.where(force,sign*(c.force_min+u[5]*(c.force_max-c.force_min)),0.),
                     (u[6]<c.sine_probability).astype(jp.float32)])


def profile(tick,event,xp=jp):
    phase=xp.clip((tick-event[0])/xp.maximum(1,event[1]-event[0]),0,1)
    amplitude=xp.where(event[4]>.5,xp.sin(xp.pi*phase),1.)
    return xp.where((tick>=event[0])&(tick<event[1]),amplitude,0.)
