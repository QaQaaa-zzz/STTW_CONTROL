"""V3 per-control-tick cost and finite-task physical failure accounting."""
from __future__ import annotations

import jax.numpy as jp


COMPONENTS = ('speed', 'steer', 'heading', 'roll', 'roll_rate', 'overspeed',
              'low_speed', 'correction', 'command_change')


def huber(z):
    a = jp.abs(z)
    return jp.where(a <= 1., z*z, 2.*a-1.)


def interval_cost(*, alpha, chi, g, raw, actual_speed, actual_steer,
                  heading_error, roll, roll_rate, executed_offsets,
                  final_command, previous_final_command, spec):
    """Score post-physics state against this interval's original raw command.

    Returns raw/effective component dictionaries and a scalar 5 ms reward.
    Invalid values remain invalid; caller records physical/policy failure.
    """
    r = spec['reward']; dt = spec['plant']['control_dt_s']
    raw = jp.asarray(raw)
    ev = actual_speed - raw[..., 0]
    ed = actual_steer - raw[..., 1]
    priority_gap = r['priority_high']-r['priority_low']
    wv = r['priority_high'] - priority_gap*chi*(1.-alpha)
    wd_base = r['priority_high'] - priority_gap*chi*alpha
    wd = (1.-g)*wd_base + g*r['recovery_steer_weight']
    correction_normalizers = jp.asarray(r['correction_normalizers'])
    final_normalizers = jp.asarray(r['final_command_normalizers'])
    raw_components = dict(
        speed=wv*huber(ev/r['speed_error_scale_m_s']),
        steer=wd*huber(ed/r['steer_error_scale_rad']),
        heading=r['heading_weight']*g*huber(jp.maximum(jp.abs(heading_error)-r['heading_deadband_rad'], 0.)/r['heading_error_scale_rad']),
        roll=r['roll_weight']*huber(jp.maximum(jp.abs(roll)-r['roll_soft_start_rad'], 0.)/r['roll_error_scale_rad']),
        roll_rate=r['roll_rate_weight']*huber(jp.maximum(jp.abs(roll_rate)-r['roll_rate_soft_start_rad_s'], 0.)/r['roll_rate_error_scale_rad_s']),
        overspeed=r['overspeed_weight']*huber(jp.maximum(ev-r['overspeed_free_band_m_s'], 0.)/r['overspeed_scale_m_s']),
        low_speed=r['low_speed_weight']*huber(jp.maximum(r['low_speed_start_m_s']-actual_speed, 0.)/r['low_speed_scale_m_s']),
        correction=r['reference_correction_weight']*jp.sum((jp.asarray(executed_offsets)/correction_normalizers)**2, axis=-1),
        command_change=r['final_command_change_weight']*jp.sum(((jp.asarray(final_command)-jp.asarray(previous_final_command))/final_normalizers)**2, axis=-1))
    raw_cost = sum(raw_components.values())
    cap = r['cost_rate_cap']
    effective_cost = jp.minimum(raw_cost, cap)
    factor = jp.minimum(1., cap/jp.maximum(raw_cost, 1e-12))
    effective_components = {name: value*factor for name, value in raw_components.items()}
    cap_fraction = jp.where(raw_cost > cap, 1., 0.)
    return dict(raw_components=raw_components,
                effective_components=effective_components,
                raw_cost=raw_cost, effective_cost=effective_cost,
                cap_fraction=cap_fraction,
                reward=-r['scale']*dt*effective_cost,
                speed_weight=wv, steer_weight=wd)


def failure_reward(remaining_including_current, spec):
    """Replace the entire current 20 ms policy reward on first physical failure."""
    r = spec['reward']; p = spec['ppo']; gamma = p['gamma']
    n = jp.asarray(remaining_including_current)
    geometric = (1.-jp.power(gamma, n))/(1.-gamma)
    return -r['failure_extra_penalty'] - (
        r['scale']*spec['plant']['policy_dt_s']*r['cost_rate_cap']*geometric)
