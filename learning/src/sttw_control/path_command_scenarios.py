"""Reset-only autonomous route sampling; never reads teleop future commands.

Resolved subfamilies: ordinary half straight / half30deg R4 gentle;
S consists of two opposite60..100deg bends separated by3m straight.
Every route has a100m static grid; only its reset anchor depends on the pose.
"""
import jax
import jax.numpy as j
from flax import struct
from .geometric_path import Path

@struct.dataclass
class Route:
    path: object
    speed: object
    family: object
    radius: object
    angle: object
    lateral_offset: object
    heading_offset: object


def sample_route(key,pose,config):
    c=config['training_future_phase_C'];p=config['path'];u=jax.random.uniform(key,(10,))
    family=j.where(u[0]<.3,0,j.where(u[0]<.8,1,2)).astype(j.int32)
    gentle=u[1]>=.5;direction=j.where(u[2]<.5,-1.,1.)
    radius=c['radius_range_m'][0]+u[3]*(c['radius_range_m'][1]-c['radius_range_m'][0])
    radius=j.where(family==0,4.,radius)
    angle=j.deg2rad(j.where(family==0,j.where(gentle,30.,0.),60.+40.*u[4]))
    speed=c['speed_range_m_s'][0]+u[5]*(c['speed_range_m_s'][1]-c['speed_range_m_s'][0])
    perturbed=u[6]<c['initial_route_pose_offset_fraction']
    lat=j.where(perturbed,c['initial_lateral_offset_range_m'][0]+u[7]*(c['initial_lateral_offset_range_m'][1]-c['initial_lateral_offset_range_m'][0]),0.)
    heading_offset=j.where(perturbed,c['initial_heading_offset_range_rad'][0]+u[8]*(c['initial_heading_offset_range_rad'][1]-c['initial_heading_offset_range_rad'][0]),0.)
    spacing=p['spacing_m'];s=j.arange(5001)*spacing;ramp=p['curvature_transition_length_m']
    length=j.where(angle>0,angle*radius+ramp,0.);start=p['straight_prefix_m'];end=start+length
    smooth=lambda x:j.clip(x,0,1)**2*(3-2*j.clip(x,0,1))
    def bend(start,end):return j.where(angle>0,smooth((s-start)/ramp)*smooth((end-s)/ramp)/radius,0.)
    second_start=end+3.;second_end=second_start+length
    k=direction*(bend(start,end)-j.where(family==2,bend(second_start,second_end),0.))
    theta=j.concatenate([j.zeros(1),j.cumsum((k[:-1]+k[1:])*.5*spacing)])
    x=j.concatenate([j.zeros(1),j.cumsum((j.cos(theta[:-1])+j.cos(theta[1:]))*.5*spacing)])
    y=j.concatenate([j.zeros(1),j.cumsum((j.sin(theta[:-1])+j.sin(theta[1:]))*.5*spacing)])
    anchor=pose[:2]+lat*j.array([-j.sin(pose[2]),j.cos(pose[2])]);yaw=pose[2]+heading_offset
    xy=j.stack([anchor[0]+j.cos(yaw)*x-j.sin(yaw)*y,anchor[1]+j.sin(yaw)*x+j.cos(yaw)*y],axis=-1)
    turn_end=j.where(family==2,second_end,j.where(angle>0,end,0.))
    path=Path(s,xy,theta+yaw,k,turn_end,turn_end+p['goal_section_offset_after_turn_m'])
    return Route(path,speed,family,radius,angle*direction,lat,heading_offset)


def episode_key(seed,env_id,episode):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed),env_id),episode)
