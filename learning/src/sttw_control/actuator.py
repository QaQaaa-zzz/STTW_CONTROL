"""Residual command composition, delay and shared final command limits.

MuJoCo still owns the velocity servo and torque limits. Acceleration parameters
limit command changes, not mechanical acceleration. Defaults add no ramp/delay.
"""
from dataclasses import dataclass
from flax import struct
import jax.numpy as jp


@dataclass(frozen=True)
class ActuatorConfig:
    dt: float=.005
    steer_limit: float=.8
    steer_rate_limit: float=3.
    rear_rate_limit: float=60.
    steer_residual_scale: float=1.
    rear_residual_scale: float=5.
    strength: float=1.
    steer_acceleration: float | None=None
    rear_acceleration: float | None=None
    delay_steps: int=0

    def __post_init__(self):
        import math
        positive=(self.dt,self.steer_limit,self.steer_rate_limit,self.rear_rate_limit)
        rates=(self.steer_acceleration,self.rear_acceleration)
        if any(not math.isfinite(x) or x<=0 for x in positive) or any(x is not None and (not math.isfinite(x) or x<=0) for x in rates) or self.delay_steps<0 or not isinstance(self.delay_steps,int):
            raise ValueError('invalid actuator timing or limits')
        if not 0<=self.strength<=1 or any(not math.isfinite(x) or x<0 for x in (self.steer_residual_scale,self.rear_residual_scale)):
            raise ValueError('invalid residual scale')


@struct.dataclass
class ActuatorState:
    pending: object
    previous: object


def initial_actuator(config=ActuatorConfig(),rear_command=0.):
    command=jp.array([0.,rear_command])
    return ActuatorState(jp.tile(command,(config.delay_steps+1,1)),command)


def apply_residual(state,base,action,steer,config=ActuatorConfig()):
    c=config
    # A malformed network output disables the entire residual for this tick.
    action=jp.where(jp.all(jp.isfinite(action)),jp.clip(action,-1,1),jp.zeros(2))
    base=jp.where(jp.all(jp.isfinite(base)),base,jp.zeros(2))
    target=base+c.strength*jp.array([c.steer_residual_scale,c.rear_residual_scale])*action
    limits=jp.array([c.steer_rate_limit,c.rear_rate_limit])
    target=jp.clip(target,-limits,limits)
    pending=jp.concatenate([state.pending[1:],target[None]],axis=0)
    command=pending[0]
    changes=c.dt*jp.array([float('inf') if c.steer_acceleration is None else c.steer_acceleration,
                          float('inf') if c.rear_acceleration is None else c.rear_acceleration])
    command=state.previous+jp.clip(command-state.previous,-changes,changes)
    # Apply the current joint-position constraint AFTER delay; stale queued
    # commands must not request travel outward beyond the measured end stop.
    lower=jp.maximum(-c.steer_rate_limit,(-c.steer_limit-steer)/c.dt)
    upper=jp.minimum(c.steer_rate_limit,(c.steer_limit-steer)/c.dt)
    lower=jp.minimum(lower,c.steer_rate_limit)
    upper=jp.maximum(upper,-c.steer_rate_limit)
    command=command.at[0].set(jp.clip(command[0],lower,upper))
    return ActuatorState(pending,command),command
