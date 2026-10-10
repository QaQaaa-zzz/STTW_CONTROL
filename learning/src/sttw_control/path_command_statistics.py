"""Small additive statistics from active 5 ms samples; no physics or policy changes.

All leaves returned by step_statistics can be summed over environment, policy
step, and update axes. Window masks overlap: strong is a subset of bend, and
exit is a spatial subset of ordinary. Route fractions are sample-weighted,
not reset-count fractions. Missing logged clipping flags remain unavailable.
"""
import jax.numpy as j
import numpy as np

COST_KEYS=('speed','path','primary','roll','working_roll','roll_rate',
           'overspeed','low_speed','offset_magnitude','offset_rate')
WINDOWS=('ordinary','bend','strong_demand','exit_straight')
COMMANDS=('target_offset','filtered_offset','offsets','governed_minus_nominal')


def step_statistics(logs,route,cfg):
    active=j.asarray(logs['active_tick'],bool)
    count=j.sum(active,dtype=j.float32)
    chi=logs['chi'];progress=logs['path_progress']
    masks=j.stack((chi<=0,chi>0,chi>=.5,
        (route.path.turn_end>0)&(progress>route.path.turn_end)&(chi<=0)),axis=-1)&active[:,None]
    errors=j.stack((logs['actual_forward_speed']-logs['v_user'],
                    logs['path_cross_track'],logs['path_heading_error']),axis=-1)
    masked=lambda values:j.where(active.reshape(active.shape+(1,)*(values.ndim-1)),values,0.)
    # where avoids contamination by padding; active NaN is deliberately preserved.
    window_sums=j.sum(j.where(masks[:,:,None],errors[:,None,:]**2,0.),axis=0)
    family=j.arange(3)==route.family
    signs=j.stack((route.angle<0,route.angle==0,route.angle>0))
    commands=j.stack((logs['target_offset'],logs['filtered_offset'],logs['offsets'],
                       logs['governed']-logs['nominal']),axis=1)
    raw=j.stack([logs['raw_components'][key] for key in COST_KEYS],axis=-1)
    effective=j.stack([logs['effective_components'][key] for key in COST_KEYS],axis=-1)
    capped=j.stack([logs['component_capped'][key] for key in COST_KEYS],axis=-1)
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
    """JSON-safe summary after the caller sums every returned leaf."""
    s={k:np.asarray(v,dtype=float) for k,v in sumstats.items()}
    if not all(np.isfinite(v).all() for v in s.values()):
        raise ValueError('nonfinite active training diagnostics')
    n=float(s['count'])
    def ratio(value,denom=n):
        return (np.asarray(value)/denom).tolist() if denom>0 else None
    windows={}
    for i,name in enumerate(WINDOWS):
        samples=float(s['window_count'][i]);rms=ratio(np.sqrt(s['window_error_sumsq'][i]*samples),samples)
        windows[name]=dict(samples=int(samples),sample_fraction=ratio(samples),
            speed_rmse_m_s=None if rms is None else rms[0],
            lateral_rmse_m=None if rms is None else rms[1],
            heading_rmse_rad=None if rms is None else rms[2])
    commands={}
    for i,name in enumerate(COMMANDS):
        commands[name]=dict(channels=['delta_v_m_s','delta_delta_rad'],
            signed_mean=ratio(s['command_sum'][i]),
            rms=ratio(np.sqrt(s['command_sumsq'][i]*n)),
            positive_fraction=ratio(s['command_positive_count'][i]),
            negative_fraction=ratio(s['command_negative_count'][i]))
    distribution={}
    for name,bounds_key,units in [('speed','speed_range_m_s','m/s'),('radius','radius_range_m','m')]:
        moments=s['route_'+name+'_moments'];bounds=cfg['training_future_phase_C'][bounds_key]
        distribution[name]=dict(units=units,mean=ratio(moments[0]),
            std=float(np.sqrt(max(moments[1]/n-(moments[0]/n)**2,0))) if n else None,
            histogram_edges=np.linspace(*bounds,6).tolist(),
            histogram_fraction=ratio(s['route_'+name+'_histogram']))
    return dict(schema='path_training_statistics_v1',active_lower_samples=int(n),
        weighting='active lower samples; overlapping spatial/chi windows; route fractions are sample weighted',
        windows=windows,route_distribution=dict(family_names=['ordinary_gentle','single_bend','s_bend'],
            family_fraction=ratio(s['route_family_count']),sign_names=['right','straight','left'],
            initial_bend_sign_fraction=ratio(s['route_sign_count']),**distribution),
        commands=commands,reference_limits=dict(
            correction_slew_any_fraction=ratio(s['correction_slew_count'],float(s['correction_slew_observed'])),
            nominal_slew_channel_fraction=ratio(s['nominal_slew_count'],float(s['nominal_slew_observed'])),
            amplitude_clip_channel_fraction=ratio(s['reference_clip_count'],float(s['reference_clip_observed'])),
            missing_flag_semantics='null means flag not recorded; never treated as zero clipping'),
        costs={key:dict(raw_mean=ratio(s['cost_raw_sum'][i]),effective_mean=ratio(s['cost_effective_sum'][i]),
                         cap_fraction=ratio(s['cost_cap_count'][i])) for i,key in enumerate(COST_KEYS)})
