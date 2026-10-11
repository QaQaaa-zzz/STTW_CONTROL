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
    sample_group: object = 0


def sample_route(key,pose,config):
    return _integrate_route(pose,config['path'],*_legacy_parameters(key,config))


def _legacy_parameters(key,config):
    c=config['training_future_phase_C'];u=jax.random.uniform(key,(10,))
    family=j.where(u[0]<.3,0,j.where(u[0]<.8,1,2)).astype(j.int32)
    gentle=u[1]>=.5;direction=j.where(u[2]<.5,-1.,1.)
    radius=c['radius_range_m'][0]+u[3]*(c['radius_range_m'][1]-c['radius_range_m'][0])
    radius=j.where(family==0,4.,radius)
    angle=j.deg2rad(j.where(family==0,j.where(gentle,30.,0.),60.+40.*u[4]))
    speed=c['speed_range_m_s'][0]+u[5]*(c['speed_range_m_s'][1]-c['speed_range_m_s'][0])
    perturbed=u[6]<c['initial_route_pose_offset_fraction']
    lat=j.where(perturbed,c['initial_lateral_offset_range_m'][0]+u[7]*(c['initial_lateral_offset_range_m'][1]-c['initial_lateral_offset_range_m'][0]),0.)
    heading_offset=j.where(perturbed,c['initial_heading_offset_range_rad'][0]+u[8]*(c['initial_heading_offset_range_rad'][1]-c['initial_heading_offset_range_rad'][0]),0.)
    return family,radius,angle,speed,direction,lat,heading_offset


def _integrate_route(pose,p,family,radius,angle,speed,direction,lat,heading_offset,sample_group=0):
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
    return Route(path,speed,family,radius,angle*direction,lat,heading_offset,j.asarray(sample_group,j.int32))


def episode_key(seed,env_id,episode):
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed),env_id),episode)


_V2_GROUPS = [
    {'ids_inclusive':[0,101], 'count':102, 'mode':'legacy_sample_route'},
    {'ids_inclusive':[102,305], 'count':204, 'mode':'strong_single'},
    {'ids_inclusive':[306,511], 'count':206, 'mode':'strong_s'},
]
_V2_STRONG = dict(speed_m_s=[2.4,2.6], radius_m=[2.0,2.2], angle_deg=[90.,100.],
    left_right_pairing=True, prefix_m=3., curvature_ramp_m=.5, s_inter_bend_m=3.,
    goal_offset_m=4., points=5001, grid_ds_m=.02, initial_offset_fraction=.2,
    lateral_offset_m=[-.1,.1], heading_offset_rad=[-.05,.05])


def validate_training_sampler(config):
    """Validate the frozen v2 reset contract; absent sampler preserves legacy."""
    sampler=config.get('training_sampler')
    if sampler is None:return
    if not isinstance(sampler,dict) or sampler.get('version')!='demand_exposure_v2':
        raise ValueError('unknown training_sampler version')
    for name,expected in [('route_seed',910201),('groups',_V2_GROUPS),('strong',_V2_STRONG)]:
        if sampler.get(name)!=expected:raise ValueError('training_sampler '+name+' differs from demand_exposure_v2')
    c=config['training_future_phase_C'];p=config['path']
    expected_c=dict(num_envs=512,episode_s=20.,speed_range_m_s=[1.8,2.6],radius_range_m=[1.8,4.],
        family_fraction={'straight_or_gentle':.3,'rounded_left_or_right_60_to_100_degrees':.5,'smooth_S_path':.2},
        initial_route_pose_offset_fraction=.2,initial_lateral_offset_range_m=[-.1,.1],
        initial_heading_offset_range_rad=[-.05,.05],freeze_physical_prepared_snapshot=True,
        randomize_path_relative_to_actual_pose_once_at_reset=True)
    expected_p=dict(spacing_m=.02,straight_prefix_m=3.,curvature_transition_length_m=.5,
        goal_section_offset_after_turn_m=4.,curvature_ramp='3*u^2-2*u^3',
        path_is_fixed_before_rollout=True,reanchor_during_rollout=False,allow_online_teleop_future=False)
    for label,actual,expected in [('training',c,expected_c),('geometry',p,expected_p)]:
        for name,value in expected.items():
            if actual.get(name)!=value:raise ValueError('demand_exposure_v2 '+label+' contract changed: '+name)


def sample_training_route(seed,env_id,episode,pose,config):
    """Reset-only version dispatch; strong pairs reflect routes, never vehicle state.

    Pair members share random draws only at matching episode numbers. Independent
    episode counters remain independent after failures. The route seed deliberately
    ignores the endpoint learner seed in v2.
    """
    sampler=config.get('training_sampler')
    if sampler is None:return sample_route(episode_key(seed,env_id,episode),pose,config)
    if sampler.get('version')!='demand_exposure_v2':raise ValueError('unknown training_sampler version')
    group=j.where(env_id<102,0,j.where(env_id<306,1,2)).astype(j.int32)
    def legacy(_):return _legacy_parameters(episode_key(sampler['route_seed'],env_id,episode),config)
    def strong(_):
        pair_id=(env_id-102)//2
        # Disjoint from the legacy group and shared by the two members of a pair.
        key=episode_key(sampler['route_seed'],102+pair_id,episode)
        u=jax.random.uniform(key,(10,));c=sampler['strong']
        uniform=lambda name,idx:c[name][0]+u[idx]*(c[name][1]-c[name][0])
        direction=j.where((env_id-102)%2==0,1.,-1.)
        radius=uniform('radius_m',3);angle=j.deg2rad(uniform('angle_deg',4))
        speed=uniform('speed_m_s',5);perturbed=u[6]<c['initial_offset_fraction']
        lat=direction*j.where(perturbed,uniform('lateral_offset_m',7),0.)
        heading=direction*j.where(perturbed,uniform('heading_offset_rad',8),0.)
        return group,radius,angle,speed,direction,lat,heading
    parameters=jax.lax.cond(group==0,legacy,strong,operand=None)
    return _integrate_route(pose,config['path'],*parameters,sample_group=group)
