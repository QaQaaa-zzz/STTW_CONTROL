"""Physical outcomes and preference utility from factual endpoint trajectories.

Rescoring changes the user's utility weights on one saved path, not the actual
policy conditioning, motion, physical outcome, or number of simulated cases.
"""
import math
import numpy as np
from .path_command_reward import configured_cost_keys
from .survival_credit import resolve_survival_credit


def rescore_v3(physical, *, user_alpha, config):
    """Replace only the three V3 preference-dependent cost channels.

    Factual terminal penalties and all non-preference costs are preserved.
    Invalid task/held/padded rows stay NaN and contribute no invented utility.
    NumPy vectorization avoids compiling a new reward kernel per trace length.
    """
    c=config['path_reward']
    if c.get('reward_version') != 'path_persistent_preference_v3':
        raise ValueError('authenticated V3 reward required for trajectory rescoring')
    if not math.isfinite(user_alpha) or not 0 <= user_alpha <= 1:
        raise ValueError('legal finite user preference required')
    valid=np.asarray(physical['active_tick'],bool) & np.asarray(physical['task_reward_valid'],bool)
    huber=lambda x:np.where(np.abs(x)<=1,x*x,2*np.abs(x)-1)
    read=lambda k:np.asarray(physical[k],dtype=np.float64)[valid]
    chi=read('chi'); ev=read('actual_forward_speed')-read('requested_speed')
    ey=read('path_cross_track'); eh=read('path_heading_error')
    caps=c['component_caps']; bounded=lambda x,k:np.minimum(x,caps[k])
    speed0=bounded((10-9*chi)*huber(ev/c['speed_scale_m_s']),'speed')
    speed1=bounded(10*huber(ev/c['speed_scale_m_s']),'speed')
    path=bounded(10*(huber(ey/c['cross_track_scale_m'])+
                    c['path_heading_factor']*huber(eh/c['path_heading_scale_rad'])),'path')
    primary0=bounded(c['primary_weight']*chi*huber(np.maximum(np.abs(ey)-c['primary_path_band_m'],0)/c['primary_path_scale_m']),'primary')
    primary1=bounded(c['primary_weight']*chi*huber(np.maximum(np.abs(ev)-c['primary_speed_band_m_s'],0)/c['primary_speed_scale_m_s']),'primary')
    target=(1-user_alpha)*speed0+user_alpha*speed1
    target+=(1-(1-c['path_secondary_ratio'])*user_alpha)*path
    target+=(1-user_alpha)*primary0+user_alpha*primary1
    keys=configured_cost_keys(config)
    source=np.asarray(physical['effective_costs'],np.float64)[valid]
    source=source[:,[keys.index(k) for k in ('speed','path','primary')]].sum(axis=1)
    factual=read('scored_tick_reward')
    score=factual+c['scale']*config['timing']['lower_dt_s']*(source-target)
    if not np.isfinite(score).all():raise ValueError('nonfinite factual task data cannot be rescored')
    per_tick=np.full(valid.shape,np.nan);per_tick[valid]=score
    return dict(user_alpha=float(user_alpha),task_valid_ticks=int(valid.sum()),
                task_scored_return=float(score.sum()),per_lower_tick_reward=per_tick)


def summarize_case(case, physical, decisions, *, horizon_s, dt_s, policy_id):
    """Keep physical survival, task termination and goal observation separate."""
    active=np.asarray(physical['active_tick'],bool)
    action=np.zeros_like(active);action[::4]=np.asarray(decisions['valid'],bool)
    fault=np.asarray(physical['nonfinite'],bool)|physical['lower_fault']|physical['policy_fault']
    result=resolve_survival_credit(active_tick=active,actor_action_valid=action,
        contact_failure=np.asarray(physical['contact_failure'],bool),engineering_fault=fault,
        horizon_reached=np.asarray(physical['horizon_reached'],bool),physical_tick=physical['physical_tick'],
        window_start_tick=0,window_ticks=round(horizon_s/dt_s),
        collection_policy_id=policy_id,learner_policy_id=policy_id)
    task=active & physical['task_reward_valid']
    values=lambda key,mask:np.asarray(physical[key],float)[mask]
    maximum=lambda key,mask:float(np.max(np.abs(values(key,mask)))) if mask.any() else None
    force=case.get('force_N',0.)
    impulse=np.asarray(physical['applied_force_impulse'])[active].sum(axis=0,dtype=np.float64)
    expected=force*case.get('pulse_duration_s',0.)*np.asarray(case.get('world_direction',[0.,0.,0.]))
    contact=np.flatnonzero(active & physical['contact_failure'])
    first_contact=float(physical['time'][contact[0]]) if contact.size else None
    return dict(case,simulation_reward_alpha=0. if case['alpha'] is None else case['alpha'],
        outcome=result['outcome'],physical_ticks=result['physical_ticks'],
        policy_decisions=int(np.asarray(decisions['valid']).sum()),task_valid_ticks=int(task.sum()),
        hold_ticks=int((active & physical['fallback_active']).sum()),
        domain_exit=bool((active & physical['domain_exit']).any()),
        first_contact_interval_start_s=first_contact,
        first_contact_interval_end_s=first_contact+dt_s if contact.size else None,
        task_scored_return=float(values('scored_tick_reward',task).sum()),
        mean_true_speed=float(values('actual_forward_speed',active).mean()) if active.any() else None,
        max_abs_roll=maximum('peak_roll',active),max_abs_task_path_error=maximum('path_cross_track',task),
        max_abs_task_heading_error=maximum('path_heading_error',task),
        goal_section_observed_passed=bool((values('goal_section_signed_distance',task)>=0).any()),
        impulse_Ns=impulse.tolist(),expected_impulse_Ns=expected.tolist(),
        nonzero_pulse_fully_delivered=bool(np.allclose(impulse,expected,atol=.025,rtol=.05)) if force else None)
