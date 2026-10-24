"""Finite reference targets before the existing filter; no state or clock resets."""
import jax.numpy as j
import math


def prefix_schedule(delay_s, duration_s, lower_dt_s):
    """Validate an opt-in delay without moving the pulse or task clock."""
    if not all(math.isfinite(x) for x in (delay_s, duration_s, lower_dt_s)) or lower_dt_s <= 0:
        raise ValueError('finite schedule and positive timestep required')
    if delay_s < 0 or duration_s <= 0:
        raise ValueError('nonnegative delay and positive duration required')
    ticks = [round(x / lower_dt_s) for x in (delay_s, duration_s)]
    if any(t % 4 or not math.isclose(t * lower_dt_s, x, abs_tol=1e-9, rel_tol=0)
           for t, x in zip(ticks, (delay_s, duration_s))):
        raise ValueError('prefix timing must align with the existing four-tick policy period')
    return tuple(ticks)


def inverse_offset_target(target, config):
    """Return finite latent, realizable offset and per-channel clipping receipt."""
    c = config['residual_interface']
    scales = j.array([j.where(target[0] >= 0, c['speed_positive_scale_m_s'],
                             c['speed_negative_scale_m_s']), c['steer_scale_rad']])
    normalized = target / scales
    bounded = j.clip(normalized, -1. + 1e-6, 1. - 1e-6)
    return j.arctanh(bounded), bounded * scales, bounded != normalized


def prefix_action(original, fixed, elapsed_ticks, prefix_ticks, enabled, mask=True, start_ticks=0):
    """Choose only the action: callers retain their complete environment carry."""
    active = enabled & (elapsed_ticks >= start_ticks) & (elapsed_ticks < start_ticks + prefix_ticks)
    return j.where(active & mask, fixed, original), active
