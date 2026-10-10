"""50 Hz Pure Pursuit target and 200 Hz nominal reference publication."""
import jax.numpy as j
from flax import struct
from .geometric_path import at,wrap

@struct.dataclass
class Pursuit:
    command: object
    preview_body: object
    preview_world: object
    lookahead: object
    kappa: object
    invalid: object


def pursuit(path,progress,pose,speed,v_user,config):
    c=config['follower']
    look=j.clip(c['lookahead_time_s']*j.maximum(speed,c['lookahead_min_speed_m_s']),c['lookahead_min_m'],c['lookahead_max_m'])
    point,_,_=at(path,progress+look);_,heading,_=at(path,progress)
    delta=point-pose[:2];cs=j.cos(pose[2]);sn=j.sin(pose[2])
    body=j.stack([cs*delta[0]+sn*delta[1],-sn*delta[0]+cs*delta[1]])
    denominator=j.sum(body**2);k=2*body[1]/j.maximum(denominator,1e-12)
    steer=j.arctan(c['wheelbase_m_expected']*k/j.cos(j.deg2rad(c['caster_deg_expected'])))
    cmd=j.stack([v_user,j.clip(steer,-c['steer_reference_max_rad'],c['steer_reference_max_rad'])])
    invalid=(body[0]<=c['preview_body_x_min_m'])|(j.abs(wrap(heading-pose[2]))>=j.pi/2)|~j.all(j.isfinite(cmd))|~j.all(j.isfinite(body))
    return Pursuit(cmd,body,point,look,k,invalid)


def publish(previous,target,config):
    c=config['follower'];dt=config['timing']['lower_dt_s']
    limits=j.array([c['nominal_speed_slew_m_s2'],c['nominal_steer_slew_rad_s']])*dt
    issued=previous+j.clip(target-previous,-limits,limits)
    return issued,(issued-previous)/dt
