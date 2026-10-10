"""Fixed world route, left-positive cross track and bounded local projection."""
import numpy as np
import jax.numpy as j
from flax import struct

@struct.dataclass
class Path:
    s: object
    xy: object
    heading: object
    curvature: object
    turn_end: object
    goal: object

@struct.dataclass
class Projection:
    progress: object
    unclamped: object
    point: object
    cross_track: object
    heading: object


def build_path(name, pose, config, lateral_offset=0.):
    c=config['path']; ds=c['spacing_m']; pose=np.asarray(pose,float)
    if name=='straight':
        end=0.; length=80.
    elif name in ('left90_R2','right90_R2'):
        ramp=c['curvature_transition_length_m']; k=1/c['turn_radius_m']
        plateau=abs(c['turn_angle_rad'])/k-ramp
        if plateau<0:raise ValueError('turn too short for curvature ramps')
        end=c['straight_prefix_m']+2*ramp+plateau; length=end+c['straight_tail_m']
    else:raise ValueError(name)
    s=np.linspace(0,length,int(np.ceil(length/ds))+1)
    def curvature(x):
        if name=='straight':return np.zeros_like(x)
        u=(x-c['straight_prefix_m'])/ramp; w=(end-x)/ramp
        h=lambda a:np.clip(a,0,1)**2*(3-2*np.clip(a,0,1))
        return (1 if name=='left90_R2' else -1)*k*h(u)*h(w)
    # Integrate on a fine fixed grid, independently of wall clock/vehicle motion.
    fine=np.linspace(0,length,int(np.ceil(length/.001))+1);df=np.diff(fine)
    kk=curvature(fine);heading=np.r_[0,np.cumsum((kk[:-1]+kk[1:])*.5*df)]
    x=np.r_[0,np.cumsum((np.cos(heading[:-1])+np.cos(heading[1:]))*.5*df)]
    y=np.r_[0,np.cumsum((np.sin(heading[:-1])+np.sin(heading[1:]))*.5*df)]
    local=np.column_stack([np.interp(s,fine,x),np.interp(s,fine,y)])
    rot=np.array([[np.cos(pose[2]),-np.sin(pose[2])],[np.sin(pose[2]),np.cos(pose[2])]])
    xy=(local+np.array([0,lateral_offset]))@rot.T+pose[:2]
    return Path(j.asarray(s),j.asarray(xy),j.asarray(np.interp(s,fine,heading)+pose[2]),j.asarray(curvature(s)),j.asarray(end),j.asarray(end+c['goal_section_offset_after_turn_m']))


def at(path,s):
    return j.stack([j.interp(s,path.s,path.xy[:,0]),j.interp(s,path.s,path.xy[:,1])]),j.interp(s,path.s,path.heading),j.interp(s,path.s,path.curvature)


def project(path,xy,previous,distance,config):
    c=config['path'];lo=j.maximum(0.,previous-c['projection_backward_m'])
    hi=j.minimum(path.s[-1],previous+j.maximum(c['projection_forward_floor_m'],c['projection_forward_motion_multiplier']*distance+c['projection_forward_padding_m']))
    a=path.xy[:-1];v=path.xy[1:]-a;span=path.s[1:]-path.s[:-1]
    u=j.sum((xy-a)*v,axis=-1)/j.sum(v*v,axis=-1)
    u=j.clip(u,j.clip((lo-path.s[:-1])/span,0,1),j.clip((hi-path.s[:-1])/span,0,1))
    ss=path.s[:-1]+u*span;points=a+u[:,None]*v
    valid=(path.s[1:]>=lo)&(path.s[:-1]<=hi)
    d2=j.where(valid,j.sum((xy-points)**2,axis=-1),j.inf);minimum=j.min(d2)
    # Tie tolerance only for floating-point identical projections.
    idx=j.argmin(j.where(d2<=minimum+1e-10,j.abs(ss-previous),j.inf))
    h=j.interp(ss[idx],path.s,path.heading);delta=xy-points[idx]
    ey=-j.sin(h)*delta[0]+j.cos(h)*delta[1]
    return Projection(j.maximum(previous,ss[idx]),ss[idx],points[idx],ey,h)


def wrap(angle):return j.arctan2(j.sin(angle),j.cos(angle))
