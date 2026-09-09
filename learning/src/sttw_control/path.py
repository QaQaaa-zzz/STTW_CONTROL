"""Circle reference and geometric lookahead feedback for the existing ECBC.

Tracks chassis-root XY using simulated localization. No learned path controller.
"""
from dataclasses import dataclass
import math
import numpy as np
import jax.numpy as jp


@dataclass(frozen=True)
class CircleConfig:
    radius: float=3.
    lookahead: float=1.2
    center_x: float=0.
    center_y: float=3.
    direction: int=1
    max_steer: float=.35

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in (self.radius,self.lookahead,self.max_steer)):
            raise ValueError('circle radius, lookahead and steer bound must be positive finite')
        if not all(math.isfinite(x) for x in (self.center_x,self.center_y)) or self.direction not in (-1,1):
            raise ValueError('invalid circle center or direction')


def circle_reference(config,count=361):
    phase=np.linspace(-np.pi/2,-np.pi/2+config.direction*2*np.pi,count)
    return np.column_stack([config.center_x+config.radius*np.cos(phase),config.center_y+config.radius*np.sin(phase)])


def circle_command(pose,config,wheelbase,caster):
    x,y,yaw=pose
    phase=jp.arctan2(y-config.center_y,x-config.center_x)
    ahead=phase+config.direction*config.lookahead/config.radius
    dx=config.center_x+config.radius*jp.cos(ahead)-x
    dy=config.center_y+config.radius*jp.sin(ahead)-y
    lateral=-jp.sin(yaw)*dx+jp.cos(yaw)*dy
    curvature=2*lateral/jp.maximum(dx*dx+dy*dy,1e-8)
    # Fork-axis steering projects onto the ground by cos(caster).
    steer=jp.arctan(wheelbase*curvature/jp.cos(caster))
    return jp.clip(steer,-config.max_steer,config.max_steer)


def tracking_metrics(xy,config):
    relative=np.asarray(xy)-np.array([config.center_x,config.center_y])
    radial=np.linalg.norm(relative,axis=1)-config.radius
    angle=np.unwrap(np.arctan2(relative[:,1],relative[:,0]))
    return {'radial_rmse_m':float(np.sqrt(np.mean(radial**2))),
            'radial_max_abs_m':float(np.max(np.abs(radial))),
            'radial_final_m':float(radial[-1]),
            'completed_turns':float(config.direction*(angle[-1]-angle[0])/(2*np.pi))}


def lateral_space_metrics(xy,config):
    """Root-point lateral occupancy relative to circle, positive left of travel.

    This excludes body width/lean and is not a swept-body collision envelope.
    Caller selects and declares the observation window (including censored runs).
    """
    xy=np.asarray(xy)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)==0 or not np.all(np.isfinite(xy)):
        raise ValueError('space metrics require nonempty finite XY samples')
    radial=np.linalg.norm(xy-np.array([config.center_x,config.center_y]),axis=1)-config.radius
    left=-config.direction*radial
    l=float(max(0.,np.max(left)));r=float(max(0.,-np.min(left)))
    return {'left_extent_m':l,'right_extent_m':r,'total_corridor_width_m':l+r,
            'scope':'chassis_root_relative_to_reference_path_excludes_body_envelope'}
