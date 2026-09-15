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


@dataclass(frozen=True)
class FigureEightConfig:
    length: float=12.
    width: float=6.
    lookahead: float=2.5
    max_steer: float=.35

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in (self.length,self.width,self.lookahead,self.max_steer)):
            raise ValueError('figure eight dimensions and limits must be positive finite')


def _eight_vectors(phase,c):
    # Gerono curve rotated so initial crossing tangent is +world X.
    angle=-jp.arctan2(2*c.width,c.length)
    rot=jp.array([[jp.cos(angle),-jp.sin(angle)],[jp.sin(angle),jp.cos(angle)]])
    p=jp.array([c.length*jp.sin(phase),c.width*jp.sin(2*phase)])
    v=jp.array([c.length*jp.cos(phase),2*c.width*jp.cos(2*phase)])
    a=jp.array([-c.length*jp.sin(phase),-4*c.width*jp.sin(2*phase)])
    return rot@p,rot@v,rot@a


def eight_geometry(phase,c):
    p,v,a=_eight_vectors(phase,c)
    return p,jp.arctan2(v[1],v[0]),(v[0]*a[1]-v[1]*a[0])/jp.maximum(jp.linalg.norm(v)**3,1e-8)


def eight_phase(pose,c):
    import jax
    grid=jp.arange(512)*(2*jp.pi/512)
    points,headings,_=jax.vmap(lambda t:eight_geometry(t,c))(grid)
    score=jp.sum((points-pose[:2])**2,axis=1)+.25*(1-jp.cos(headings-pose[2]))
    phase=grid[jp.argmin(score)]
    # Refine projection on the selected branch; fixed-size, JAX-compatible.
    for _ in range(4):
        p,v,a=_eight_vectors(phase,c);d=p-pose[:2]
        denom=jp.dot(v,v)+jp.dot(d,a)
        phase-=jp.clip(jp.dot(d,v)/jp.maximum(denom,1e-6),-.1,.1)
    return phase


def eight_features(pose,c):
    p,tangent,curvature=eight_geometry(eight_phase(pose,c),c)
    delta=pose[:2]-p
    right=jp.sin(tangent)*delta[0]-jp.cos(tangent)*delta[1]
    heading=jp.arctan2(jp.sin(pose[2]-tangent),jp.cos(pose[2]-tangent))
    return jp.array([right,heading,curvature])


def eight_command(pose,c,wheelbase,caster):
    phase=eight_phase(pose,c)
    for _ in range(12):
        _,v,_=_eight_vectors(phase,c)
        phase+=(c.lookahead/12)/jp.linalg.norm(v)
    target,_,_=eight_geometry(phase,c);d=target-pose[:2]
    lateral=-jp.sin(pose[2])*d[0]+jp.cos(pose[2])*d[1]
    steer=jp.arctan(wheelbase*2*lateral/jp.maximum(jp.dot(d,d),1e-8)/jp.cos(caster))
    return jp.clip(steer,-c.max_steer,c.max_steer)


def eight_reference(c,count=513):
    import jax
    return np.asarray(jax.vmap(lambda t:eight_geometry(t,c)[0])(jp.linspace(0,2*jp.pi,count)))


def eight_trace_features(pose,c):
    import jax
    return np.asarray(jax.jit(jax.vmap(lambda p:eight_features(p,c)))(jp.asarray(pose)))


@dataclass(frozen=True)
class BendConfig:
    straight: float=10.
    bend_length: float=20.
    exit_length: float=50.
    peak_curvature: float=.12
    lookahead: float=2.5
    max_steer: float=.35
    direction: int=1

    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in (self.straight,self.bend_length,self.exit_length,self.peak_curvature,self.lookahead,self.max_steer)) or self.direction not in (-1,1):
            raise ValueError('invalid smooth bend')
        if self.peak_curvature*self.bend_length/2>=1.5:raise ValueError('bend heading must remain below 1.5 radians')


def bend_table(c):
    s=np.linspace(0,c.straight+c.bend_length+c.exit_length,801)
    u=np.clip((s-c.straight)/c.bend_length,0,1)
    k=c.direction*c.peak_curvature*np.sin(np.pi*u)**2
    yaw=c.direction*c.peak_curvature*c.bend_length*(u/2-np.sin(2*np.pi*u)/(4*np.pi))
    ds=np.diff(s);x=np.r_[0,np.cumsum(ds*(np.cos(yaw[1:])+np.cos(yaw[:-1]))/2)]
    y=np.r_[0,np.cumsum(ds*(np.sin(yaw[1:])+np.sin(yaw[:-1]))/2)]
    return np.column_stack((s,x,y,yaw,k))


def bend_features(pose,table,progress=None,window=None):
    xy=table[:,1:3];v=xy[1:]-xy[:-1]
    u=jp.clip(jp.sum((pose[:2]-xy[:-1])*v,axis=1)/jp.sum(v*v,axis=1),0,1)
    valid=jp.ones_like(u,dtype=bool)
    if progress is not None:
        lo=jp.maximum(table[0,0],progress-window);hi=jp.minimum(table[-1,0],progress+window)
        ds=table[1:,0]-table[:-1,0]
        valid=(table[1:,0]>=lo)&(table[:-1,0]<=hi)
        u=jp.clip(u,jp.clip((lo-table[:-1,0])/ds,0,1),jp.clip((hi-table[:-1,0])/ds,0,1))
    foot=xy[:-1]+u[:,None]*v;i=jp.argmin(jp.where(valid,jp.sum((foot-pose[:2])**2,axis=1),jp.inf))
    yaw=table[i,3]+u[i]*(table[i+1,3]-table[i,3]);d=pose[:2]-foot[i]
    right=jp.sin(yaw)*d[0]-jp.cos(yaw)*d[1]
    heading=jp.arctan2(jp.sin(pose[2]-yaw),jp.cos(pose[2]-yaw))
    curvature=table[i,4]+u[i]*(table[i+1,4]-table[i,4])
    progress=table[i,0]+u[i]*(table[i+1,0]-table[i,0])
    return jp.array([right,heading,curvature]),progress


def bend_command(pose,c,table,wheelbase,caster,progress=None):
    s=bend_features(pose,table)[1] if progress is None else progress
    ahead=s+c.lookahead
    dx=jp.interp(ahead,table[:,0],table[:,1])-pose[0];dy=jp.interp(ahead,table[:,0],table[:,2])-pose[1]
    lateral=-jp.sin(pose[2])*dx+jp.cos(pose[2])*dy
    return jp.clip(jp.arctan(wheelbase*2*lateral/jp.maximum(dx*dx+dy*dy,1e-8)/jp.cos(caster)),-c.max_steer,c.max_steer)


def bend_trace_features(pose,c):
    import jax
    return np.asarray(jax.jit(jax.vmap(lambda p:bend_features(p,jp.asarray(bend_table(c)))[0]))(jp.asarray(pose)))


@dataclass(frozen=True)
class ReferencePaths:
    """Fixed global paths obtained by integrating declared (time, speed, yaw-rate)."""
    cases: tuple
    selected: int | None = None  # evaluation only; training samples uniformly on reset
    continuous_projection: bool = False  # false preserves frozen historical tasks
    projection_margin: float = .05  # m, added to twice the actual XY displacement

    def __post_init__(self):
        if not math.isfinite(self.projection_margin) or self.projection_margin<=0:raise ValueError('projection margin must be positive')
        if not self.cases or len({c['name'] for c in self.cases})!=len(self.cases):
            raise ValueError('reference cases must be nonempty and uniquely named')
        if self.selected is not None and (type(self.selected) is not int or not 0<=self.selected<len(self.cases)):
            raise ValueError('invalid selected reference')
        for case in self.cases:
            if not case['name'] or any(x in case['name'] for x in ('/','\\')) or case['name'] in ('.','..'):raise ValueError('invalid reference name')
            last=-1.
            for row in case['commands']:
                if len(row)!=3 or not all(math.isfinite(x) for x in row) or row[0]<=last or row[1]<=0:
                    raise ValueError('reference rows require increasing time, positive speed, finite yaw rate')
                last=row[0]
            if not case['commands'] or case['commands'][0][0]!=0:raise ValueError('reference must start at zero')


def integrated_reference(commands,horizon,dt=.005,count=801):
    """Exact constant-twist integration, resampled by arc length; no feasibility claim."""
    rows=np.asarray(commands,float)
    times=np.unique(np.r_[np.arange(0,horizon+5+dt/2,dt),rows[:,0]])
    xy=[np.zeros(2)];yaw=[0.];arc=[0.]
    for t,end in zip(times[:-1],times[1:]):
        _,v,w=rows[max(0,np.searchsorted(rows[:,0],t+1e-9,side='right')-1)]
        h=end-t;theta=yaw[-1];n=theta+w*h
        d=v*h*np.array([np.cos(theta),np.sin(theta)]) if abs(w)<1e-12 else v/w*np.array([np.sin(n)-np.sin(theta),np.cos(theta)-np.cos(n)])
        xy.append(xy[-1]+d);yaw.append(n);arc.append(arc[-1]+v*h)
    s=np.linspace(0,arc[-1],count);xy=np.asarray(xy);t=np.interp(s,arc,times)
    idx=np.maximum(np.searchsorted(rows[:,0],t,side='right')-1,0)
    return np.column_stack((s,np.interp(s,arc,xy[:,0]),np.interp(s,arc,xy[:,1]),np.interp(s,arc,yaw),rows[idx,2]/rows[idx,1]))


def reference_table(config):
    """Plot/evaluation reference from a frozen task dictionary."""
    bank=config.get('reference_paths')
    if bank is not None:
        i=bank.get('selected')
        if i is None:raise ValueError('select a reference before rendering a mixed bank')
        return integrated_reference(bank['cases'][i]['commands'],config['horizon_seconds'],config.get('controller',{}).get('dt',.005))
    return bend_table(BendConfig(**config['bend']))


def features_at_progress(pose,table,progress):
    """Measure errors against the one stored projection used by the controller."""
    x=jp.interp(progress,table[:,0],table[:,1]);y=jp.interp(progress,table[:,0],table[:,2])
    yaw=jp.interp(progress,table[:,0],table[:,3]);k=jp.interp(progress,table[:,0],table[:,4])
    right=jp.sin(yaw)*(pose[0]-x)-jp.cos(yaw)*(pose[1]-y)
    heading=jp.arctan2(jp.sin(pose[2]-yaw),jp.cos(pose[2]-yaw))
    return jp.array([right,heading,k])
