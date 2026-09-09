"""Finite-horizon local residual allocation using observable wheel-speed proxy.

The first-order servo parameters are declared design assumptions, NOT identified
vehicle dynamics. The mapping is a policy coordinate change, not a safety proof.
"""
from dataclasses import dataclass
import math
import jax.numpy as jp

ACTION_FIELDS=['lateral_acceleration_proxy_residual','speed_proxy_residual']


@dataclass(frozen=True)
class MappingConfig:
    horizon: float=.1
    steer_time_constant: float=.02
    rear_time_constant: float=.1
    wheel_radius_proxy: float=.1
    minimum_speed_proxy: float=.5
    lateral_proxy_scale: float=1.
    speed_proxy_scale: float=.2
    coupled: bool=True

    def __post_init__(self):
        values=(self.horizon,self.steer_time_constant,self.rear_time_constant,
                self.wheel_radius_proxy,self.minimum_speed_proxy,self.lateral_proxy_scale,self.speed_proxy_scale)
        if any(not math.isfinite(x) or x<=0 for x in values) or not isinstance(self.coupled,bool):
            raise ValueError('invalid action mapping parameters')


def response_matrix(speed,steer,config,actuator,controller):
    """Map command residual [rad/s, rad/s] to [m/s² proxy, m/s proxy].

    Curvature proxy = cos(caster)*tan(steer)/wheelbase. Response coefficients
    predict increments versus the same-state zero-residual continuation, assuming
    a held command and first-order servos, before shared nonlinear limits.
    """
    c=config
    h=jp.maximum(c.horizon-actuator.delay_steps*actuator.dt,0.)
    angle_response=h-c.steer_time_constant*(-jp.expm1(-h/c.steer_time_constant))
    speed_response=c.wheel_radius_proxy*(-jp.expm1(-h/c.rear_time_constant))
    projection=jp.cos(controller.caster)/controller.wheelbase
    curvature=projection*jp.tan(steer)
    dk=projection/jp.cos(steer)**2
    cross=2*speed*curvature*speed_response if c.coupled else jp.asarray(0.)
    return jp.array([[speed**2*dk*angle_response,cross],[0.,speed_response]])


def map_action(action,speed,steer,config,actuator,controller):
    """Return bounded normalized motor residuals for the existing shared limiter."""
    valid=jp.all(jp.isfinite(action))&jp.isfinite(speed)&jp.isfinite(steer)&(speed>=config.minimum_speed_proxy)
    safe_speed=jp.where(jp.isfinite(speed),jp.maximum(speed,config.minimum_speed_proxy),config.minimum_speed_proxy)
    safe_steer=jp.clip(jp.where(jp.isfinite(steer),steer,0.),-actuator.steer_limit,actuator.steer_limit)
    request=jp.where(jp.isfinite(action),jp.clip(action,-1,1),0.)*jp.array([config.lateral_proxy_scale,config.speed_proxy_scale])
    J=response_matrix(safe_speed,safe_steer,config,actuator,controller)
    rear=jp.clip(request[1]/jp.maximum(J[1,1],1e-8),-actuator.rear_residual_scale,actuator.rear_residual_scale)
    turn=(request[0]-J[0,1]*rear)/jp.maximum(J[0,0],1e-8)
    scales=jp.array([actuator.steer_residual_scale,actuator.rear_residual_scale])
    normalized=jp.where(scales>0,jp.array([turn,rear])/jp.maximum(scales,1e-8),0.)
    return jp.where(valid,jp.clip(normalized,-1,1),jp.zeros(2))
