"""Finite-time hold criteria. Stable motion before an event is not recovery."""
from dataclasses import dataclass
from flax import struct
import jax.numpy as jp


@dataclass(frozen=True)
class RecoveryConfig:
    dt: float=.005
    hold_seconds: float=.5
    roll_tolerance: float=.05
    roll_rate_tolerance: float=.15
    speed_tolerance: float=.2
    steer_tolerance: float=.05

    def __post_init__(self):
        import math
        if any(not math.isfinite(x) or x<=0 for x in (self.dt,self.hold_seconds,self.roll_tolerance,self.roll_rate_tolerance,self.speed_tolerance,self.steer_tolerance)):
            raise ValueError('recovery intervals and tolerances must be positive and finite')


@struct.dataclass
class RecoveryState:
    balance_ticks: object
    task_ticks: object
    balance_recovered: object
    task_recovered: object


def initial_recovery():
    return RecoveryState(jp.int32(0),jp.int32(0),jp.bool_(False),jp.bool_(False))


def update_recovery(state,roll_error,roll_rate,speed_error,steer_error,event_finished,failed,config=RecoveryConfig()):
    c=config
    balance=event_finished & ~jp.asarray(failed) & (jp.abs(roll_error)<=c.roll_tolerance) & (jp.abs(roll_rate)<=c.roll_rate_tolerance)
    task=balance & (jp.abs(speed_error)<=c.speed_tolerance) & (jp.abs(steer_error)<=c.steer_tolerance)
    bt=jp.where(balance,state.balance_ticks+1,0)
    tt=jp.where(task,state.task_ticks+1,0)
    import math
    needed=int(math.ceil(c.hold_seconds/c.dt))
    return RecoveryState(bt,tt,state.balance_recovered|(bt>=needed),state.task_recovered|(tt>=needed))
