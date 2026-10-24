"""Finite reference targets before the existing filter; no state or clock resets."""
import jax.numpy as j


def inverse_offset_target(target, config):
    """Return finite latent, realizable offset and per-channel clipping receipt."""
    c = config['residual_interface']
    scales = j.array([j.where(target[0] >= 0, c['speed_positive_scale_m_s'],
                             c['speed_negative_scale_m_s']), c['steer_scale_rad']])
    normalized = target / scales
    bounded = j.clip(normalized, -1. + 1e-6, 1. - 1e-6)
    return j.arctanh(bounded), bounded * scales, bounded != normalized


def prefix_action(original, fixed, elapsed_ticks, prefix_ticks, enabled, mask=True):
    """Choose only the action: callers retain their complete environment carry."""
    active = enabled & (elapsed_ticks < prefix_ticks)
    return j.where(active & mask, fixed, original), active
