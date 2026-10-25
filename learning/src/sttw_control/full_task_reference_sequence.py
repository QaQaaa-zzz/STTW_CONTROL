"""Finite progress-conditioned reference processes, not an Actor or safety model."""
import numpy as np
import jax.numpy as j
from .reference_prefix import inverse_offset_target


def progress_knots(prefix,turn_end):
    return j.array([prefix-1.5,prefix,(prefix+turn_end)*.5,turn_end,turn_end+1.5])


def sequence_target(progress,positions,targets):
    interval=j.clip(j.searchsorted(positions,progress,side='right')-1,0,3)
    u=j.clip((progress-positions[interval])/(positions[interval+1]-positions[interval]),0.,1.)
    smooth=u*u*(3.-2.*u)
    return targets[interval]+smooth*(targets[interval+1]-targets[interval])


def sequence_action(progress,positions,targets,config,stopped,zero_mode):
    desired=sequence_target(progress,positions,targets)
    fault=~stopped&(~j.isfinite(progress)|~j.all(j.isfinite(desired)))
    safe=j.where(j.isfinite(desired),desired,0.)
    latent,realized,clipped=inverse_offset_target(safe,config)
    valid=~(stopped|zero_mode|fault)
    return j.where(valid,latent,j.zeros(2)),dict(reference_valid=valid,fault=fault,
        desired_target=j.where(valid,desired,j.zeros(2)),mapped_target=j.where(valid,realized,j.zeros(2)),
        target_clipped=valid&clipped)


def candidate_library(config,seed=20261025):
    """37 preregistered targets; no outcome-dependent rejection or resampling."""
    c=config['residual_interface'];rng=np.random.default_rng(seed)
    low=np.array([-c['speed_negative_scale_m_s'],-c['steer_scale_rad']])
    high=np.array([c['speed_positive_scale_m_s'],c['steer_scale_rad']])
    targets=np.zeros((37,5,2),np.float32)
    targets[:32,1:4]=rng.uniform(low,high,size=(32,3,2))
    constants=[[-.5,0.],[.24,0.],[0.,-.08],[0.,.08]]
    targets[33:,1:4]=np.asarray(constants,np.float32)[:,None,:]
    if not np.isfinite(targets).all() or np.any(targets<low) or np.any(targets>high):
        raise ValueError('candidate targets outside declared correction authority')
    return dict(targets=targets,names=[f'sequence_{i:02d}' for i in range(32)]+
        ['sequence_zero','constant_slow','constant_fast','constant_steer_negative','constant_steer_positive'],
        seed=seed,kind=['sequence']*32+['zero']+['constant']*4)


def make_full_sequence_step(env,*,horizon_s):
    """One existing live physical advance; no old Actor receives out-of-domain data.

    Domain exit remains cumulative telemetry. Geometric reference generation is
    allowed throughout the fixed horizon for both spline and constant controls.
    No physical, ESO, filter, history, or task state is cleared or synthesized.
    """
    from .live_path_continuation import make_live_reference_step
    from .path_command_reward import configured_cost_keys
    advance=make_live_reference_step(env,horizon_s=horizon_s)
    keys=tuple(configured_cost_keys(env.config))

    def step(carry,params,path,speed,pulse,_profile,_floor,zero_mode):
        s=carry.state
        stopped=s.physical.failed|s.fault|(s.tick>=round(horizon_s/env.cc.dt))
        positions=progress_knots(env.config['path']['straight_prefix_m'],path.turn_end)
        action,receipt=sequence_action(s.projection.progress,positions,params['targets'],
            env.config,stopped,zero_mode)
        end,logs=advance(s.replace(fault=s.fault|receipt['fault']),action,path,speed,pulse)
        active=logs['active_tick'];ticks=logs['physical_tick']
        names=['active_tick','actual_xy','actual_forward_speed','wheel_speed_proxy',
            'phi','phi_dot','actual_delta','peak_roll','contact_failure','physical_failure',
            'nonfinite','lower_fault','policy_fault','time','path_cross_track','path_heading_error',
            'path_progress','goal_section_signed_distance','applied_force_impulse',
            'tick_reward','scored_tick_reward','physical_tick','chi']
        data={k:logs[k] for k in names}
        data.update(actual_yaw=logs['yaw_unwrapped'],reference=logs['governed'],
            nominal_reference=logs['nominal'],filtered_offset=logs['filtered_offset'],
            correction_offset=logs['offsets'],target_offset=logs['target_offset'],
            domain_invalid=logs['domain_exit'],
            ever_domain_exit=active&(s.domain_exit|(j.cumsum(logs['domain_exit'])>0)),
            zero_fallback=active&zero_mode,actor_action_valid=j.zeros(4,bool),
            reference_action_valid=active&receipt['reference_valid'],
            task_reward_valid=active,requested_speed=j.repeat(speed,4),
            issued_latent=j.where(active[:,None],j.tile(action,(4,1)),j.nan),
            original_actor_latent=j.full((4,2),j.nan),
            sequence_desired_target=j.tile(receipt['desired_target'],(4,1)),
            sequence_mapped_target=j.tile(receipt['mapped_target'],(4,1)),
            sequence_target_clipped=active[:,None]&j.tile(receipt['target_clipped'],(4,1)),
            progress_envelope=j.zeros(4),gate_detected=j.zeros(4,bool),gate_override=j.zeros(4,bool),gate_signal=j.full(4,j.nan),
            raw_costs=j.stack([logs['raw_components'][k] for k in keys],axis=-1),
            effective_costs=j.stack([logs['effective_components'][k] for k in keys],axis=-1),
            horizon_reached=active&(ticks==round(horizon_s/env.cc.dt)-1))
        return carry.replace(state=end),data
    return step
