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
    sampling_stage: object = None
    speed_values: object = None
    speed_switch_times: object = None


def sample_route(key,pose,config):
    return _integrate_route(pose,config['path'],*_legacy_parameters(key,config))


def _legacy_parameters(key,config):
    c=config['training_future_phase_C'];u=jax.random.uniform(key,(10,))
    family=j.where(u[0]<.3,0,j.where(u[0]<.8,1,2)).astype(j.int32)
    gentle=u[1]>=.5;direction=j.where(u[2]<.5,-1.,1.)
    radius=c['radius_range_m'][0]+u[3]*(c['radius_range_m'][1]-c['radius_range_m'][0])
    radius=j.where(family==0,c.get('gentle_radius_m',4.),radius)
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
    """Keep v2 frozen; v3 exposes speed/radius ranges in the run config."""
    sampler=config.get('training_sampler')
    if sampler is None:return
    if isinstance(sampler,dict) and sampler.get('version')=='path_conflict_exposure_v1':
        if sampler.get('route_seed')!=910501 or sampler.get('speed_event_seed')!=911503:
            raise ValueError('conflict sampler independent seed identity differs')
        conflict_groups=[dict(_V2_GROUPS[0]),dict(_V2_GROUPS[1],mode='double_u'),dict(_V2_GROUPS[2],mode='alternating_four')]
        if sampler.get('groups') not in (_V2_GROUPS,conflict_groups):
            raise ValueError('conflict sampler groups differ from declared 102/204/206 lanes')
        legacy=dict(config,training_sampler=dict(sampler,version='demand_exposure_v2',route_seed=910201,groups=_V2_GROUPS))
        return validate_training_sampler(legacy)
    if not isinstance(sampler,dict) or sampler.get('version') not in ('demand_exposure_v2','demand_exposure_v3'):
        raise ValueError('unknown training_sampler version')
    configured=sampler['version']=='demand_exposure_v3'
    strong=dict(_V2_STRONG)
    if configured:
        for key in ('speed_m_s','radius_m'):
            lo,hi=sampler['strong'][key]
            if not 0<lo<=hi:raise ValueError('positive ordered strong ranges required')
            strong[key]=[lo,hi]
        strong['prefix_m']=config['path']['straight_prefix_m']
    for name,expected in [('route_seed',910201),('groups',_V2_GROUPS),('strong',strong)]:
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
    if configured:
        for key in ('speed_range_m_s','radius_range_m'):
            lo,hi=c[key]
            if not 0<lo<=hi:raise ValueError('positive ordered training ranges required')
            expected_c[key]=[lo,hi]
        if c.get('gentle_radius_m',4.)<=0:raise ValueError('positive gentle radius required')
        expected_p['straight_prefix_m']=p['straight_prefix_m']
    for label,actual,expected in [('training',c,expected_c),('geometry',p,expected_p)]:
        for name,value in expected.items():
            if actual.get(name)!=value:raise ValueError('demand_exposure_v2 '+label+' contract changed: '+name)


def sample_training_route(seed,env_id,episode,pose,config,absolute_update=1):
    """Reset-only version dispatch; strong pairs reflect routes, never vehicle state.

    Pair members share random draws only at matching episode numbers. Independent
    episode counters remain independent after failures. The route seed deliberately
    ignores the endpoint learner seed in v2.
    """
    sampler=config.get('training_sampler')
    if sampler is None:return sample_route(episode_key(seed,env_id,episode),pose,config)
    if sampler.get('version')=='path_conflict_exposure_v1':
        return _sample_conflict_route(env_id,episode,pose,config,absolute_update)
    if sampler.get('version') not in ('demand_exposure_v2','demand_exposure_v3'):raise ValueError('unknown training_sampler version')
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


def speed_request(values,times,time):
    """Resolve only the request already published at this control boundary."""
    # Events are quantized to20ms. Compare tick indices, not float32 products
    # such as40*.005 vs a separately encoded .2 switch boundary.
    now=j.floor(j.asarray(time)/.02+1e-4)
    switches=j.rint(j.asarray(times)/.02)
    return values[j.sum(now>=switches).astype(j.int32)]


def _legacy_turn_metadata(route,config):
    p=config['path'];angle=j.abs(route.angle);length=j.where(angle>0,angle*route.radius+p['curvature_transition_length_m'],0.)
    start=p['straight_prefix_m'];end=start+length
    starts=j.array([start,end+3.,0.,0.]);ends=j.array([end,end+3.+length,0.,0.])
    mask=j.array([angle>0,(angle>0)&(route.family==2),False,False])
    directions=j.sign(route.angle)*j.array([1.,-1.,0.,0.])
    return route.path.replace(turn_starts=j.where(mask,starts,0.),turn_ends=j.where(mask,ends,0.),
        turn_directions=j.where(mask,directions,0.),turn_mask=mask)


def _persistent_route(pose,parameters,group,prefix,gap,angle):
    family,radius,_,speed,direction,lat,heading_offset=parameters
    s=j.arange(5001)*.02;ramp=.5;nturn=j.where(group==1,2,4)
    length=angle*radius+ramp;starts=prefix+j.arange(4)*(length+gap);ends=starts+length
    mask=j.arange(4)<nturn;directions=direction*j.array([1.,-1.,1.,-1.])
    smooth=lambda u:j.clip(u,0,1)**2*(3-2*j.clip(u,0,1))
    shapes=smooth((s[None,:]-starts[:,None])/ramp)*smooth((ends[:,None]-s[None,:])/ramp)
    # Each discrete bend integrates to its declared angle exactly, including non-grid boundaries.
    areas=j.sum((shapes[:,:-1]+shapes[:,1:])*.01,axis=1)
    k=j.sum(shapes*(j.where(mask,directions*angle/areas,0.))[:,None],axis=0)
    theta=j.concatenate([j.zeros(1),j.cumsum((k[:-1]+k[1:])*.01)])
    x=j.concatenate([j.zeros(1),j.cumsum((j.cos(theta[:-1])+j.cos(theta[1:]))*.01)])
    y=j.concatenate([j.zeros(1),j.cumsum((j.sin(theta[:-1])+j.sin(theta[1:]))*.01)])
    anchor=pose[:2]+lat*j.array([-j.sin(pose[2]),j.cos(pose[2])]);yaw=pose[2]+heading_offset
    xy=j.stack([anchor[0]+j.cos(yaw)*x-j.sin(yaw)*y,anchor[1]+j.sin(yaw)*x+j.cos(yaw)*y],axis=-1)
    turn_end=ends[nturn-1]
    path=Path(s,xy,theta+yaw,k,turn_end,turn_end+4.,j.where(mask,starts,0.),
        j.where(mask,ends,0.),j.where(mask,directions,0.),mask)
    return Route(path,speed,group,radius,direction*angle,lat,heading_offset,group)


def _sample_conflict_route(env_id,episode,pose,config,absolute_update):
    sampler=config['training_sampler'];stage=j.where(absolute_update<=40,1,j.where(absolute_update<=80,2,3)).astype(j.int32)
    # Dispatch through the unmodified V2 arithmetic for phase one and ordinary lanes.
    legacy_config=dict(config,training_sampler=dict(sampler,version='demand_exposure_v2'))
    old=sample_training_route(0,env_id,episode,pose,legacy_config)
    old=old.replace(path=_legacy_turn_metadata(old,config))
    group=old.sample_group;pair_id=102+(env_id-102)//2
    u=jax.random.uniform(episode_key(sampler['route_seed'],pair_id,episode),(10,))
    direction=j.where((env_id-102)%2==0,1.,-1.)
    radius=1.8+.2*u[3];angle=j.deg2rad(j.where(group==1,150.+30.*u[4],80.+20.*u[4]))
    speed=j.where(stage==2,2.4+.3*u[5],2.7+.2*u[5])
    lat=direction*j.where(u[6]<.2,-.1+.2*u[7],0.);heading=direction*j.where(u[6]<.2,-.05+.1*u[8],0.)
    parameters=(group,radius,angle,speed,direction,lat,heading)
    route=jax.lax.cond((stage>1)&(group>0),
        lambda _: _persistent_route(pose,parameters,group,2.+3.*u[1],.5+.5*u[2],angle),lambda _:old,None)
    events=jax.random.uniform(episode_key(sampler['speed_event_seed'],env_id,episode),(5,))
    low=2.+.3*events[0];high=2.6+.3*events[1];first=j.where(events[2]<.5,low,high);second=j.where(events[2]<.5,high,low)
    variable=(stage>1)&(group==0)
    values=j.where(variable,j.array([first,second,first]),j.repeat(route.speed,3))
    times=j.where(variable,j.round(j.array([1.+2.*events[3],5.+3.*events[4]])/.02)*.02,j.array([1.e30,1.e30]))
    return route.replace(speed=values[0],sampling_stage=stage,speed_values=values,speed_switch_times=times)


def explicit_case_declarations(config):
    """Serializable geometry and command identities; legacy tuple consumers remain separate."""
    e=config['stage_C_evaluation'];p=config['path'];rows=[]
    for route in e['route_ids']:
        for speed in [e['low_speed_m_s'],e['conflict_speed_m_s']]:
            sign=-1 if route.startswith('right') else 1
            geometry=dict(angles_rad=[] if route=='straight' else [sign*p['turn_angle_rad']],
                radius_m=None if route=='straight' else p['turn_radius_m'],prefix_m=p['straight_prefix_m'],
                gap_m=0.,curvature_transition_length_m=p['curvature_transition_length_m'],
                goal_offset_m=p['goal_section_offset_after_turn_m'])
            rows.append(dict(name=f'{route}_{speed:.1f}',route_id=route,speed_values=[speed]*3,
                speed_switch_times=[1.e30,1.e30],evaluation_role='legacy_regression',geometry=geometry))
    if config.get('training_sampler',{}).get('version')!='path_conflict_exposure_v1':return rows
    for route in ['left180_R1p8','right180_R1p8','compact_LR','compact_RL']:
        sign=-1 if route in ('right180_R1p8','compact_RL') else 1
        angles=[sign*float(j.pi)] if '180' in route else [sign*float(j.pi)/2,-sign*float(j.pi)/2]
        geometry=dict(angles_rad=angles,radius_m=1.8,prefix_m=4.,gap_m=.8 if len(angles)>1 else 0.,curvature_transition_length_m=.5,goal_offset_m=4.)
        rows.append(dict(name=route+'_2.8',route_id=route,speed_values=[2.8]*3,
            speed_switch_times=[1.e30,1.e30],evaluation_role='constant_pressure',geometry=geometry))
    for route in ['compact_LR','compact_RL']:
        geometry=next(r['geometry'] for r in rows if r['route_id']==route)
        rows.append(dict(name=route+'_variable',route_id=route,speed_values=[2.3,2.8,2.3],
            speed_switch_times=[1.,5.],evaluation_role='variable_speed_response',geometry=geometry))
    return rows
