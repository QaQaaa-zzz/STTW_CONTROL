"""Deployable attitude observations based on the supplied HRRL figure."""
from dataclasses import dataclass
from flax import struct
import jax.numpy as jp

FIELDS=('roll_error','roll','roll_rate','speed_estimate','steer','steer_rate',
        'yaw_rate_body','rear_rate','front_rate','steer_reference','speed_reference',
        'base_steer_rate','previous_steer_command','previous_rear_command','estimated_disturbance')


PATH_FIELDS=("radial_error","heading_error","path_curvature")

def observation_fields(config):
    return FIELDS + (PATH_FIELDS if config.include_path else ())

@dataclass(frozen=True)
class ObservationConfig:
    history_steps: int=1
    include_path: bool=False
    def __post_init__(self):
        if not isinstance(self.history_steps,int) or self.history_steps<1:
            raise ValueError('history_steps must be positive integer')


@struct.dataclass
class History:
    frames: object
    mask: object


def make_frame(measurement,command,reference_roll,base_steer,previous,disturbance):
    roll,roll_rate,steer,steer_rate,yaw_rate,rear_rate,front_rate=measurement
    return jp.array([roll-reference_roll,roll,roll_rate,rear_rate*.1,steer,steer_rate,
                     yaw_rate,rear_rate,front_rate,command[0],command[1],base_steer,
                     previous[0],previous[1],disturbance])


def initial_history(config=ObservationConfig()):
    return History(jp.zeros((config.history_steps,len(observation_fields(config)))),jp.zeros(config.history_steps))


def advance_history(state,frame,config=ObservationConfig()):
    frames=jp.concatenate([state.frames[1:],frame[None]],axis=0)
    mask=jp.concatenate([state.mask[1:],jp.ones(1)])
    return History(frames,mask),jp.concatenate([frames.flatten(),mask])
