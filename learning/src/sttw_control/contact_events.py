"""Observed contact transitions; no inferred task success or invented recovery labels."""
import numpy as np


def extract_contact_events(time, wheel_position, wheel_force, valid, far_edge_x,
                           far_bank_half_width, recovery_mask=None, recovery_hold_s=1.,
                           terminal_failure=False, force_threshold_n=1.):
    time=np.asarray(time,dtype=float);position=np.asarray(wheel_position);force=np.asarray(wheel_force)
    valid=np.asarray(valid,dtype=bool)
    if (time.ndim!=1 or len(time)<2 or np.any(np.diff(time)<=0)
        or position.shape!=(len(time),2,3) or force.shape!=position.shape
        or valid.shape!=time.shape or not np.isfinite(time).all()
        or not np.isfinite(position[valid]).all() or not np.isfinite(force[valid]).all()
        or far_bank_half_width<=0 or recovery_hold_s<=0):
        raise ValueError('Expected increasing timestamps and finite valid [time,front/rear,xyz] samples')
    if recovery_mask is not None and np.asarray(recovery_mask).shape!=time.shape:
        raise ValueError('Recovery predicate must align with samples')
    contact=force[:,:,2]>force_threshold_n
    modes=contact[:,0].astype(int)+2*contact[:,1].astype(int)
    events=[];previous=None;flight=False;landing=None;stable_start=None;recovered=None
    for i in range(len(time)):
        if not valid[i]:
            previous=None;stable_start=None
            continue
        mode=int(modes[i])
        if previous is None or mode!=previous:
            events.append(dict(kind='contact_mode',index=i,time_s=float(time[i]),mode=mode))
        if previous is not None and previous!=0 and mode==0:
            flight=True;events.append(dict(kind='support_lost',index=i,time_s=float(time[i])))
        far=(position[i,:,0]>far_edge_x)&(np.abs(position[i,:,1])<far_bank_half_width)&contact[i]
        if flight and landing is None and far.any():
            landing=i;events.append(dict(kind='first_far_bank_contact_proxy',index=i,time_s=float(time[i]),wheels=np.flatnonzero(far).tolist()))
        if landing is not None and recovery_mask is not None and recovered is None:
            supported=far.all() and bool(recovery_mask[i])
            stable_start=i if supported and stable_start is None else stable_start if supported else None
            if stable_start is not None and time[i]-time[stable_start]>=recovery_hold_s:
                recovered=i
                events.append(dict(kind='recovered',index=i,time_s=float(time[i]),hold_start_s=float(time[stable_start])))
        previous=mode
    observed=np.flatnonzero(valid)
    if recovery_mask is None:status='unlabelled_missing_frozen_recovery_predicate'
    elif recovered is not None:status='observed'
    elif terminal_failure:status='failed_before_recovery'
    elif landing is None:status='no_far_bank_contact_observed'
    else:status='right_censored_at_observation_end'
    return dict(events=events,mode_names={'0':'no_wheel_support','1':'front_only','2':'rear_only','3':'both'},
                sample_dt_min_s=float(np.diff(time).min()),sample_dt_max_s=float(np.diff(time).max()),
                recovery_status=status,recovery_distance_m=None if recovered is None else float(position[recovered,:,0].mean()-position[landing,:,0].mean()),
                last_observed_time_s=None if not len(observed) else float(time[observed[-1]]),
                limitations='Far-bank contact is a spatial net-force proxy, not collider identity. No task success inferred. Recovery distance is longitudinal displacement; curved-route arclength must be supplied by task geometry.')
