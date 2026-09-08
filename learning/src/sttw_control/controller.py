"""Pure JAX ECBC with explicit state; internal roll is left-positive.

Input: speed m/s, steer rad, steer rate rad/s, roll rad, roll rate rad/s,
steer reference rad. getOutput-before-update follows the C++ implementation.
"""
from dataclasses import dataclass
from flax import struct
import jax.numpy as jp


@dataclass(frozen=True)
class ControllerConfig:
    dt: float = .005
    wo: float = 40.
    wc: float = 5.
    disturbance_filter: float = 8.
    minimum_speed: float = .5
    max_steer_rate: float = 4.
    mass: float = 5.4
    cg_forward: float = .164
    cg_height: float = .2
    wheelbase: float = .408
    trail: float = .024
    caster: float = 25.*3.141592653589793/180.
    gravity: float = 9.8

    def __post_init__(self):
        import math
        values=(self.dt,self.wo,self.wc,self.disturbance_filter,self.minimum_speed,self.max_steer_rate,
                self.mass,self.cg_forward,self.cg_height,self.wheelbase,self.trail,self.gravity)
        if any(not math.isfinite(x) or x<=0 for x in values) or not math.isfinite(self.caster):
            raise ValueError('invalid controller parameters')
        if self.dt*self.disturbance_filter>1:
            raise ValueError('disturbance Euler filter requires dt * frequency <= 1')


@struct.dataclass
class ControllerState:
    eso: object
    disturbance: object
    gains: object


@struct.dataclass
class ControllerOutput:
    steer_rate: object
    reference_roll: object
    disturbance: object
    equilibrium_shift: object


def _floor_magnitude(x, floor):
    return jp.where(x < 0, -1., 1.) * jp.maximum(jp.abs(x), floor)


def _system(speed, previous_gains, c):
    v=jp.maximum(c.minimum_speed,speed)
    inertia=c.mass*c.cg_height**2+.043929*c.mass/5.434
    m1=-c.mass*c.cg_forward*c.cg_height*v*jp.cos(c.caster)/c.wheelbase
    m2=_floor_magnitude(-(c.mass*v*v*c.cg_height-c.mass*c.cg_forward*c.trail*c.gravity)*jp.cos(c.caster)/c.wheelbase,.1)
    m4=-c.mass*c.gravity*c.cg_height
    a1,a2,a4=m1/inertia,m2/inertia,-m4/inertia
    matrix=jp.array([[-a2*a2,a1*a2,-a1*a1],[-a2*a4,a1*a4,-a2],[a1*a4,-a2,a1]])
    det=_floor_magnitude(a4*a1*a1-a2*a2,2.)
    gains=jp.where(jp.abs(det)>5.,matrix@jp.array([3*c.wc,3*c.wc**2+a4,c.wc**3])/det,previous_gains)
    return a1,a2,a4,m4/m2,gains


def initial_controller(config=ControllerConfig()):
    *_,gains=_system(1.,jp.zeros(3),config)
    return ControllerState(jp.zeros(4),jp.asarray(0.),gains)


def controller_step(state, measurement, enable_eso, config=ControllerConfig()):
    c=config
    speed,steer,steer_rate,roll,roll_rate,reference=measurement
    a1,a2,a4,ratio,gains=_system(speed,state.gains,c)
    shift=jp.where(enable_eso,-state.disturbance/a4,0.)
    reference_roll=reference/ratio
    u=jp.clip(gains@jp.array([reference-steer,reference_roll-roll+shift,-roll_rate]),-c.max_steer_rate,c.max_steer_rate)
    e0,e1=steer-state.eso[0],roll-state.eso[1]
    derivative=jp.array([steer_rate+30*e0,
                         state.eso[2]+3*c.wo*e1,
                         a2*state.eso[0]+a4*state.eso[1]+state.eso[3]+a1*steer_rate+(3*c.wo**2+a4)*e1,
                         c.wo**3*e1])
    new=ControllerState(state.eso+c.dt*derivative,
                        state.disturbance+c.dt*c.disturbance_filter*(state.eso[3]-state.disturbance),gains)
    return new,ControllerOutput(u,reference_roll,state.disturbance,-state.disturbance/a4)
