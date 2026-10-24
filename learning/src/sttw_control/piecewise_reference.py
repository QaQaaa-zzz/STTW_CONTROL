"""Offline finite reference families; no physical state or policy reset."""
import numpy as np
import jax.numpy as j

from .reference_prefix import prefix_schedule


def validate_piecewise_schedule(delay_s, durations_s, lower_dt_s):
    if not durations_s:
        raise ValueError('at least one positive segment required')
    delay, _ = prefix_schedule(delay_s, durations_s[0], lower_dt_s)
    ends = []
    total = 0
    for duration in durations_s:
        _, ticks = prefix_schedule(0., duration, lower_dt_s)
        total += ticks
        ends.append(total)
    return delay, tuple(ends)


def piecewise_action(original, targets, segment_ends, elapsed_ticks,
                     delay_ticks, enabled, mask):
    """Targets are pre-inverted latents anchored by the caller once.

    Integer end boundaries are exclusive. The caller validates the schedule
    before JIT and retains all environment/filter/history carry unchanged.
    """
    if targets.ndim != 2 or targets.shape[1] != 2 or segment_ends.shape != (targets.shape[0],):
        raise ValueError('one endpoint for every two-channel target required')
    relative = elapsed_ticks - delay_ticks
    index = j.minimum(j.sum(relative >= segment_ends), targets.shape[0] - 1)
    active = enabled & j.any(mask) & (relative >= 0) & (relative < segment_ends[-1])
    return j.where(active & mask, targets[index], original), active


def matched_target_families(seed):
    """Equal 16-member families, with matching per-stage delta marginals.

    Temporal pairs are a randomized cyclic derangement, not CEM, optimization,
    or an exhaustive search. No outcomes enter generation.
    """
    speed = np.linspace(-.15, .15, 4)
    steer = np.linspace(-.04, .04, 4)
    grid = np.array([(v, d) for v in speed for d in steer])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(grid))
    shift = int(rng.integers(1, len(grid)))
    first = grid[order]
    second = grid[np.roll(order, shift)]
    return np.stack([first, first], axis=1), np.stack([first, second], axis=1)
