"""Small device statistics from active 5 ms samples; no physics or policy changes.

Use merge_statistics/reduce_statistics: window_peak_roll is a maximum, while
all other leaves are additive. Window masks overlap: strong is a subset of bend, and
exit is a spatial subset of ordinary. Route fractions are sample-weighted,
not reset-count fractions. Missing logged clipping flags remain unavailable.
"""
import jax.numpy as j
import numpy as np

from .path_command_reward import BASE_COST_KEYS as COST_KEYS,configured_cost_keys

WINDOWS=('ordinary','bend','strong_demand','exit_straight',
         'geometric_turn','demand_positive','local_left','local_right')
COMMANDS=('target_offset','filtered_offset','offsets','governed_minus_nominal')


def merge_statistics(left,right):
    """Merge whole trees or sparse reset events, without transferring devices."""
    result=dict(left)
    for key,value in right.items():
        if key not in result:
            result[key]=value
        elif key=='window_peak_roll':
            xp=np if isinstance(result[key],np.ndarray) and isinstance(value,np.ndarray) else j
            result[key]=xp.maximum(result[key],value)
        else:
            result[key]=result[key]+value
    return result


def reduce_statistics(statistics,axis=0):
    """Reduce batch/scan axes; NumPy trees stay on host, JAX trees on device."""
    result={}
    for key,value in statistics.items():
        xp=np if isinstance(value,np.ndarray) else j
        result[key]=(xp.max if key=='window_peak_roll' else xp.sum)(value,axis=axis)
    return result


def reset_statistics(route,reset=True):
    """One actual route installation (including initial reset), never padding."""
    count=j.asarray(reset,j.float32)
    return dict(reset_count=count,reset_family_count=(j.arange(3)==route.family)*count,
        reset_sample_group_count=(j.arange(3)==getattr(route,'sample_group',0))*count)


def step_statistics(logs,route,cfg):
    cost_keys=configured_cost_keys(cfg)
    active=j.asarray(logs['active_tick'],bool)
    count=j.sum(active,dtype=j.float32)
    chi=logs['chi'];progress=logs['path_progress']
    curvature=logs['path_curvature']
    masks=j.stack((chi<=0,chi>0,chi>=.5,
        (route.path.turn_end>0)&(progress>route.path.turn_end)&(chi<=0),
        j.abs(curvature)>1e-4,chi>0,curvature>0,curvature<0),axis=-1)&active[:,None]
    errors=j.stack((logs['actual_forward_speed']-logs['v_user'],
                    logs['path_cross_track'],logs['path_heading_error']),axis=-1)
    masked=lambda values:j.where(active.reshape(active.shape+(1,)*(values.ndim-1)),values,0.)
    # where avoids contamination by padding; active NaN is deliberately preserved.
    window_sums=j.sum(j.where(masks[:,:,None],errors[:,None,:]**2,0.),axis=0)
    family=j.arange(3)==route.family
    signs=j.stack((route.angle<0,route.angle==0,route.angle>0))
    commands=j.stack((logs['target_offset'],logs['filtered_offset'],logs['offsets'],
                       logs['governed']-logs['nominal']),axis=1)
    raw=j.stack([logs['raw_components'][key] for key in cost_keys],axis=-1)
    effective=j.stack([logs['effective_components'][key] for key in cost_keys],axis=-1)
    capped=j.stack([logs['component_capped'][key] for key in cost_keys],axis=-1)
    def window_sum(values):
        shape=masks.shape+(1,)*(values.ndim-1)
        return j.sum(j.where(masks.reshape(shape),values[:,None,...],0.),axis=0)
    roll=logs['peak_roll']
    lower_error=logs['lower_error']
    def flag_stats(key,channels):
        if key not in logs:return j.zeros(channels),j.array(0.)
        flags=j.asarray(logs[key]).reshape((active.shape[0],channels))
        return j.sum(masked(flags),axis=0),count
    slew,slew_observed=flag_stats('correction_rate_clipped',1)
    nominal_slew,nominal_observed=flag_stats('nominal_slew_channels',2)
    clip,clip_observed=flag_stats('reference_clip_channels',2)
    c=cfg['training_future_phase_C']
    def histogram(value,bounds):
        # Five uniform bins over declared distribution, with endpoints retained.
        index=j.clip(j.floor((value-bounds[0])/(bounds[1]-bounds[0])*5),0,4)
        return (j.arange(5)==index)*count
    return dict(count=count,window_count=j.sum(masks,axis=0,dtype=j.float32),
        window_error_sum=window_sum(errors),window_error_abs_sum=window_sum(j.abs(errors)),
        window_peak_roll=j.max(j.where(masks,roll[:,None],0.),axis=0),
        window_working_roll_exceedance_count=window_sum(roll>cfg['stage_B_baseline']['working_roll_rad']),
        window_command_sum=window_sum(commands),
        window_lower_error_sum=window_sum(lower_error),window_lower_error_sumsq=window_sum(lower_error**2),
        window_cost_raw_sum=window_sum(raw),window_cost_effective_sum=window_sum(effective),
        window_cost_cap_count=window_sum(capped),
        route_sample_group_count=(j.arange(3)==getattr(route,'sample_group',0))*count,
        **reset_statistics(route,False),
        window_error_sumsq=window_sums,route_family_count=family*count,
        route_sign_count=signs*count,route_speed_moments=j.array([route.speed,route.speed**2])*count,
        route_radius_moments=j.array([route.radius,route.radius**2])*count,
        route_speed_histogram=histogram(route.speed,c['speed_range_m_s']),
        route_radius_histogram=histogram(route.radius,c['radius_range_m']),
        command_sum=j.sum(masked(commands),axis=0),command_sumsq=j.sum(masked(commands**2),axis=0),
        command_positive_count=j.sum(masked(commands>0),axis=0),
        command_negative_count=j.sum(masked(commands<0),axis=0),
        correction_slew_count=slew,correction_slew_observed=slew_observed,
        nominal_slew_count=nominal_slew,nominal_slew_observed=nominal_observed,
        reference_clip_count=clip,reference_clip_observed=clip_observed,
        cost_raw_sum=j.sum(masked(raw),axis=0),cost_effective_sum=j.sum(masked(effective),axis=0),
        cost_cap_count=j.sum(masked(capped),axis=0))


def summarize_statistics(sumstats,cfg):
    """JSON-safe summary of properly reduced trees, including saved v1 trees."""
    cost_keys=configured_cost_keys(cfg)
    s={k:np.asarray(v,dtype=float) for k,v in sumstats.items()}
    if not all(np.isfinite(v).all() for v in s.values()):
        raise ValueError('nonfinite active training diagnostics')
    n=float(s['count'])
    def ratio(value,denom=n):
        return (np.asarray(value)/denom).tolist() if denom>0 else None
    windows={}
    for i,name in enumerate(WINDOWS[:len(s['window_count'])]):
        samples=float(s['window_count'][i]);rms=ratio(np.sqrt(s['window_error_sumsq'][i]*samples),samples)
        windows[name]=dict(samples=int(samples),sample_fraction=ratio(samples),
            speed_rmse_m_s=None if rms is None else rms[0],
            lateral_rmse_m=None if rms is None else rms[1],
            heading_rmse_rad=None if rms is None else rms[2])
        if 'window_peak_roll' in s:
            mean=ratio(s['window_error_sum'][i],samples)
            mae=ratio(s['window_error_abs_sum'][i],samples)
            windows[name].update(
                speed_signed_mean_m_s=None if mean is None else mean[0],
                lateral_signed_mean_m=None if mean is None else mean[1],
                heading_signed_mean_rad=None if mean is None else mean[2],
                error_mae=mae,error_channels=['speed_m_s','lateral_m','heading_rad'],
                peak_roll_rad=float(s['window_peak_roll'][i]) if samples else None,
                working_roll_bound_rad=cfg['stage_B_baseline']['working_roll_rad'],
                working_roll_exceedance_duration_s=float(s['window_working_roll_exceedance_count'][i])*cfg['timing']['lower_dt_s'],
                working_roll_exceedance_fraction=ratio(s['window_working_roll_exceedance_count'][i],samples),
                commands={key:dict(channels=['delta_v_m_s','delta_delta_rad'],
                    signed_mean=ratio(s['window_command_sum'][i,k],samples)) for k,key in enumerate(COMMANDS)},
                lower_actual_minus_governed=dict(channels=['speed_m_s','steer_rad'],
                    signed_mean=ratio(s['window_lower_error_sum'][i],samples),
                    rms=ratio(np.sqrt(s['window_lower_error_sumsq'][i]*samples),samples)),
                costs={key:dict(raw_mean=ratio(s['window_cost_raw_sum'][i,k],samples),
                    effective_mean=ratio(s['window_cost_effective_sum'][i,k],samples),
                    cap_fraction=ratio(s['window_cost_cap_count'][i,k],samples)) for k,key in enumerate(cost_keys)})
    commands={}
    for i,name in enumerate(COMMANDS):
        commands[name]=dict(channels=['delta_v_m_s','delta_delta_rad'],
            signed_mean=ratio(s['command_sum'][i]),
            rms=ratio(np.sqrt(s['command_sumsq'][i]*n)),
            positive_fraction=ratio(s['command_positive_count'][i]),
            negative_fraction=ratio(s['command_negative_count'][i]))
    distribution={}
    if 'route_sample_group_count' in s:
        distribution.update(sample_group_names=['legacy','strong_single','strong_s'],
            sample_group_fraction=ratio(s['route_sample_group_count']))
    if 'reset_count' in s:
        resets=float(s['reset_count'])
        distribution.update(reset_count=int(resets),reset_family_fraction=ratio(s['reset_family_count'],resets),
            reset_sample_group_fraction=ratio(s['reset_sample_group_count'],resets))
    for name,bounds_key,units in [('speed','speed_range_m_s','m/s'),('radius','radius_range_m','m')]:
        moments=s['route_'+name+'_moments'];bounds=cfg['training_future_phase_C'][bounds_key]
        distribution[name]=dict(units=units,mean=ratio(moments[0]),
            std=float(np.sqrt(max(moments[1]/n-(moments[0]/n)**2,0))) if n else None,
            histogram_edges=np.linspace(*bounds,6).tolist(),
            histogram_fraction=ratio(s['route_'+name+'_histogram']))
    return dict(schema='path_training_statistics_v1',active_lower_samples=int(n),
        weighting='active lower samples; overlapping spatial/chi windows; route fractions are sample weighted; exceedance durations sum active environment time',
        windows=windows,route_distribution=dict(family_names=['ordinary_gentle','single_bend','s_bend'],
            family_fraction=ratio(s['route_family_count']),sign_names=['right','straight','left'],
            initial_bend_sign_fraction=ratio(s['route_sign_count']),**distribution),
        commands=commands,reference_limits=dict(
            correction_slew_any_fraction=ratio(s['correction_slew_count'],float(s['correction_slew_observed'])),
            nominal_slew_channel_fraction=ratio(s['nominal_slew_count'],float(s['nominal_slew_observed'])),
            amplitude_clip_channel_fraction=ratio(s['reference_clip_count'],float(s['reference_clip_observed'])),
            missing_flag_semantics='null means flag not recorded; never treated as zero clipping'),
        costs={key:dict(raw_mean=ratio(s['cost_raw_sum'][i]),effective_mean=ratio(s['cost_effective_sum'][i]),
                         cap_fraction=ratio(s['cost_cap_count'][i])) for i,key in enumerate(cost_keys)})
