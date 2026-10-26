"""Offline nominal preference diagnostics with explicit observation windows."""
import numpy as np


def phase_metrics(speed_error, path_error, tick_cost, count, dt):
    """Include exactly count completed physical transitions, no interpolation."""
    speed_error, path_error, tick_cost = map(np.asarray, (speed_error, path_error, tick_cost))
    if not (speed_error.ndim == path_error.ndim == tick_cost.ndim == 1
            and speed_error.shape == path_error.shape == tick_cost.shape
            and isinstance(count, (int, np.integer)) and 0 < count <= len(speed_error)
            and np.isfinite(dt) and dt > 0):
        raise ValueError('aligned finite prefix and positive dt required')
    if not all(np.isfinite(x[:count]).all() for x in [speed_error, path_error, tick_cost]):
        raise ValueError('nonfinite observed prefix')
    duration = count * dt
    return dict(ticks=int(count), duration_s=duration,
        speed_abs_error_integral_m=float(np.abs(speed_error[:count]).sum() * dt),
        path_abs_error_time_integral_m_s=float(np.abs(path_error[:count]).sum() * dt),
        mean_abs_speed_error_m_s=float(np.abs(speed_error[:count]).mean()),
        mean_abs_path_error_m=float(np.abs(path_error[:count]).mean()),
        original_alpha_cost=float(tick_cost[:count].sum()))


def first_hit_count(mask):
    hits = np.flatnonzero(np.asarray(mask, bool))
    return int(hits[0] + 1) if len(hits) else None


def endpoint_direction(a, b):
    """Describe alpha1-minus-alpha0; no success criterion or tolerance tuning."""
    speed = b['mean_abs_speed_error_m_s'] - a['mean_abs_speed_error_m_s']
    path = b['mean_abs_path_error_m'] - a['mean_abs_path_error_m']
    return dict(alpha1_minus_alpha0_mean_abs_speed_error_m_s=speed,
        alpha1_minus_alpha0_mean_abs_path_error_m=path,
        alpha1_better_speed=bool(speed < 0), alpha0_better_path=bool(path > 0),
        both_expected_directions=bool(speed < 0 and path > 0))
