"""Pure geometric recovery reference rules and immutable-prefix episode joining.

No learned Actor, force estimate, future disturbance, state reset or path rewrite.
Forward geometric points do not certify physical recoverability.
"""
import copy
import numpy as np
import jax
import jax.numpy as j
from .geometric_path import at,wrap


def geometric_target(path,progress,pose,speed,nominal,mode,config):
    """mode0=zero,1=fixed6m,2=closest sampled forward point; common turn fallback."""
    c=config['follower']
    original=j.clip(c['lookahead_time_s']*j.maximum(speed,c['lookahead_min_speed_m_s']),c['lookahead_min_m'],c['lookahead_max_m'])
    lengths=j.array([original,2.,4.,6.,10.,16.,24.])
    points=jax.vmap(lambda L:at(path,progress+L)[0])(lengths)
    delta=points-pose[:2];cs,sn=j.cos(pose[2]),j.sin(pose[2])
    body=j.stack([cs*delta[:,0]+sn*delta[:,1],-sn*delta[:,0]+cs*delta[:,1]],axis=-1)
    forward=body[:,0]>c['preview_body_x_min_m']
    closest=j.argmin(j.where(forward,lengths,j.inf))
    index=j.where(mode==1,3,j.where(mode==2,closest,0))
    available=forward[index]
    kappa=2*body[index,1]/j.maximum(j.sum(body[index]**2),1e-12)
    point_steer=j.arctan(c['wheelbase_m_expected']*kappa/j.cos(j.deg2rad(c['caster_deg_expected'])))
    _,heading,_=at(path,progress)
    fallback=.18*j.sign(wrap(heading-pose[2]))
    desired=j.clip(j.where(available,point_steer,fallback),-c['steer_reference_max_rad'],c['steer_reference_max_rad'])
    target=j.where(mode==0,j.zeros(2),j.stack([j.float32(0),desired-nominal[1]]))
    fault=~j.all(j.isfinite(j.concatenate([pose,nominal,j.atleast_1d(progress),j.atleast_1d(speed),target])))|((mode<0)|(mode>2))
    return j.where(fault,j.zeros(2),target),dict(override=mode>0,point_available=available,
        heading_fallback=(mode>0)&~available,lookahead_m=j.where(available,lengths[index],j.nan),
        body_point=body[index],world_point=points[index],desired_steer_rad=j.where(mode==0,nominal[1],desired),fault=fault)


def make_geometry_step(env,*,horizon_s=22.):
    from .anticipatory_reference import floor_config
    from .full_task_reference_sequence import make_full_sequence_step
    def step(carry,mode,path,speed,pulse,lower_bound):
        local=copy.copy(env);local.config=floor_config(env.config,lower_bound)
        s=carry.state;_,pose,actual_speed=local.physics.observe(s.physical.data)
        target,record=geometric_target(path,s.projection.progress,pose,actual_speed,s.physical.raw,mode,local.config)
        stopped=s.physical.failed|s.fault|(s.tick>=round(horizon_s/local.cc.dt))
        start=carry.replace(state=s.replace(fault=s.fault|(~stopped&record['fault'])))
        # All5 values equal: original sequence mapper publishes the current
        # target through the SAME original live/filter/slew/plant kernel once.
        advance=make_full_sequence_step(local,horizon_s=horizon_s)
        end,data=advance(start,dict(targets=j.tile(target,(5,1))),path,speed,pulse,j.int32(0),lower_bound,j.bool_(False))
        active=data['active_tick']
        data.update(zero_fallback=active&(mode==0),geometry_override=active&record['override'],geometry_point_available=active&record['point_available'],
            geometry_heading_fallback=active&record['heading_fallback'],
            geometry_lookahead_m=j.repeat(record['lookahead_m'],4),geometry_body_point=j.tile(record['body_point'],(4,1)),
            geometry_world_point=j.tile(record['world_point'],(4,1)),geometry_desired_steer_rad=j.repeat(record['desired_steer_rad'],4))
        return end,data
    return step


def join_prefix_suffix(prefix,suffix):
    """Only common logged fields; complete suffix diagnostics stay separately saved."""
    pa=np.asarray(prefix['active_tick'],bool);sa=np.asarray(suffix['active_tick'],bool)
    pt=np.asarray(prefix['physical_tick'])[pa];st=np.asarray(suffix['physical_tick'])[sa]
    if len(pt)!=2000 or not pa.all() or not np.array_equal(pt,np.arange(2000)):
        raise ValueError('unchanged complete2000tick prefix required')
    if len(st) and not np.array_equal(st,np.arange(2000,2000+len(st))):
        raise ValueError('suffix clock must continue source2000 without reset/gaps')
    if np.any(np.diff(sa.astype(np.int8))>0):raise ValueError('stopped suffix cannot resume')
    shared=prefix.keys()&suffix.keys()
    joined={k:np.concatenate([prefix[k],suffix[k]],axis=0) for k in shared}
    joined['suffix_active']=np.r_[np.zeros(len(pa),bool),sa]
    return joined
