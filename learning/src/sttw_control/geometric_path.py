"""Fixed world route, left-positive cross track and bounded local projection."""
import numpy as np
import jax
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
    turn_starts: object = None
    turn_ends: object = None
    turn_directions: object = None
    turn_mask: object = None

@struct.dataclass
class Projection:
    progress: object
    unclamped: object
    point: object
    cross_track: object
    heading: object


def build_path(name, pose, config, lateral_offset=0.):
    if name in ('left180_R1p8','right180_R1p8','compact_LR','compact_RL'):
        return _build_conflict_path(name,pose,lateral_offset)
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
    path=Path(j.asarray(s),j.asarray(xy),j.asarray(np.interp(s,fine,heading)+pose[2]),j.asarray(curvature(s)),j.asarray(end),j.asarray(end+c['goal_section_offset_after_turn_m']))
    if config.get('training_sampler',{}).get('version')=='path_conflict_exposure_v1':
        turning=name!='straight';direction=1. if name=='left90_R2' else -1.
        path=path.replace(turn_starts=j.array([c['straight_prefix_m'] if turning else 0.,0.,0.,0.]),
            turn_ends=j.array([end,0.,0.,0.]),turn_directions=j.array([direction if turning else 0.,0.,0.,0.]),
            turn_mask=j.array([turning,False,False,False]))
    return path


def _grid_index(path,s):
    """All generated routes use a uniform grid; use actual float32 knots."""
    last=path.s.shape[0]-1
    spacing=(path.s[-1]-path.s[0])/last
    idx=j.clip(j.floor((s-path.s[0])/spacing).astype(j.int32),0,last-1)
    # Grid rounding can put an exact knot on either side of the estimate.
    idx=j.where(s<path.s[idx],j.maximum(idx-1,0),idx)
    return j.where(s>=path.s[idx+1],j.minimum(idx+1,last-1),idx)


def at(path,s):
    s=j.clip(s,path.s[0],path.s[-1]);idx=_grid_index(path,s)
    fraction=(s-path.s[idx])/(path.s[idx+1]-path.s[idx])
    interp=lambda values:values[idx]+fraction*(values[idx+1]-values[idx])
    return interp(path.xy),interp(path.heading),interp(path.curvature)


def project(path,xy,previous,distance,config):
    c=config['path'];lo=j.maximum(0.,previous-c['projection_backward_m'])
    hi=j.minimum(path.s[-1],previous+j.maximum(c['projection_forward_floor_m'],c['projection_forward_motion_multiplier']*distance+c['projection_forward_padding_m']))
    count=path.s.shape[0]-1;width=min(64,count)
    start=_grid_index(path,lo)
    start=j.where(path.s[start]>=lo,j.maximum(start-1,0),start)
    def segment_data(base):
        ids=base+j.arange(width);index=j.minimum(ids,count-1)
        knots=path.s[index];span=path.s[index+1]-knots
        a=path.xy[index];v=path.xy[index+1]-a
        u=j.sum((xy-a)*v,axis=-1)/j.sum(v*v,axis=-1)
        u=j.clip(u,j.clip((lo-knots)/span,0,1),j.clip((hi-knots)/span,0,1))
        ss=knots+u*span;points=a+u[:,None]*v
        valid=(ids<count)&(path.s[index+1]>=lo)&(knots<=hi)
        d2=j.where(valid,j.sum((xy-points)**2,axis=-1),j.inf)
        return ss,points,d2
    def has_segments(carry):
        base=carry[0]
        return (base<count)&(path.s[j.minimum(base,count)]<=hi)
    def minimum_step(carry):
        base,minimum=carry;_,_,d2=segment_data(base)
        return base+width,j.minimum(minimum,j.min(d2))
    _,minimum=jax.lax.while_loop(has_segments,minimum_step,(start,j.asarray(j.inf,xy.dtype)))
    # Second local pass preserves the original global min+1e-10 tie rule,
    # even when an unusually wide motion window spans multiple blocks.
    def nearest_step(carry):
        base,best_distance,best_s,best_point=carry
        ss,points,d2=segment_data(base)
        delta=j.where(d2<=minimum+1e-10,j.abs(ss-previous),j.inf)
        idx=j.argmin(delta);choose=delta[idx]<best_distance
        return base+width,j.minimum(best_distance,delta[idx]),j.where(choose,ss[idx],best_s),j.where(choose,points[idx],best_point)
    _,_,selected,point=jax.lax.while_loop(has_segments,nearest_step,
        (start,j.asarray(j.inf,xy.dtype),path.s[0],path.xy[0]))
    _,h,_=at(path,selected);delta=xy-point
    ey=-j.sin(h)*delta[0]+j.cos(h)*delta[1]
    return Projection(j.maximum(previous,selected),selected,point,ey,h)


def wrap(angle):return j.arctan2(j.sin(angle),j.cos(angle))


def _build_conflict_path(name,pose,lateral_offset=0.):
    """Fixed original geometry; independent fine integration avoids grid angle bias."""
    pose=np.asarray(pose,float);radius=1.8;ramp=.5;prefix=4.;gap=.8
    angles=[np.pi] if '180' in name else [np.pi/2,np.pi/2]
    sign=-1 if name in ('right180_R1p8','compact_RL') else 1
    directions=np.array([sign*(-1)**i for i in range(len(angles))])
    starts=[];ends=[];cursor=prefix
    for angle in angles:
        starts.append(cursor);ends.append(cursor+angle*radius+ramp);cursor=ends[-1]+gap
    s=np.arange(5001)*.02;fine=np.arange(100001)*.001
    def curvature(x):
        k=np.zeros_like(x);smooth=lambda u:np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
        for start,end,direction in zip(starts,ends,directions):
            k+=direction*smooth((x-start)/ramp)*smooth((end-x)/ramp)/radius
        return k
    kk=curvature(fine);h=np.r_[0,np.cumsum((kk[:-1]+kk[1:])*.0005)]
    x=np.r_[0,np.cumsum((np.cos(h[:-1])+np.cos(h[1:]))*.0005)]
    y=np.r_[0,np.cumsum((np.sin(h[:-1])+np.sin(h[1:]))*.0005)]
    local=np.column_stack([np.interp(s,fine,x),np.interp(s,fine,y)])
    rot=np.array([[np.cos(pose[2]),-np.sin(pose[2])],[np.sin(pose[2]),np.cos(pose[2])]])
    xy=(local+np.array([0,lateral_offset]))@rot.T+pose[:2]
    pad=lambda values:j.asarray(np.pad(values,(0,4-len(values))))
    return Path(j.asarray(s),j.asarray(xy),j.asarray(np.interp(s,fine,h)+pose[2]),
        j.asarray(curvature(s)),j.asarray(ends[-1]),j.asarray(ends[-1]+4.),
        pad(starts),pad(ends),pad(directions),j.arange(4)<len(angles))
