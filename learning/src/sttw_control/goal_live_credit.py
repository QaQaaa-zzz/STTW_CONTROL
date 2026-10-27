"""Bridge the existing live collector schema to opt-in G/C/S credit.

No collector, sampling permission, objective weights or optimizer is changed.
Caller authenticates the original user-alpha/path/config and full population.
"""
import math
import numpy as np
from .goal_task_credit import resolve_goal_task_credit


def resolve_live_goal_credit(decisions, physical, *, goal_xy, radius_m,
                             config, collection_id, learner_id, horizon_s):
    """Resolve one lane after collection using factual running, not scored reward.

    The existing collector repeats action permission over four lower ticks;
    an actual policy decision exists only at the first lower tick of its block.
    Running costs include fallback ticks; failure replacement rewards must not
    enter C. Stored effective components independently check their integral.
    New goal/cost value predictions are NOT fabricated from the old task head.
    """
    from .path_command_reward import configured_cost_keys
    valid = np.asarray(decisions['valid'])
    active = np.asarray(physical['active_tick'])
    if (valid.dtype != np.bool_ or valid.ndim != 1 or valid.size == 0 or
            active.dtype != np.bool_ or active.shape != (4 * valid.size,)):
        raise ValueError('one lane with four lower slots per Boolean upper decision required')
    dt = float(config['timing']['lower_dt_s'])
    upper = float(config['timing']['upper_dt_s'])
    scale = float(config['path_reward']['scale'])
    if (not all(math.isfinite(x) and x > 0 for x in (dt, upper, scale, horizon_s)) or
            not math.isclose(upper, 4 * dt, abs_tol=1e-10) or
            not math.isclose(round(horizon_s / dt) * dt, horizon_s, abs_tol=1e-8)):
        raise ValueError('positive aligned collector timing, scale and horizon required')
    clock = np.asarray(physical['physical_tick'])
    ticks = np.asarray(decisions['task_tick'])
    if (ticks.shape != valid.shape or not np.issubdtype(ticks.dtype, np.integer) or
            clock.shape != active.shape or
            not np.array_equal(ticks[valid], np.flatnonzero(valid) * 4)):
        raise ValueError('actual non-reset policy decision clock required')
    permission = np.asarray(physical['actor_action_valid'])
    if (permission.dtype != np.bool_ or permission.shape != active.shape or
            not np.array_equal(permission, np.repeat(valid, 4) & active) or
            np.any(valid & ~active[::4])):
        raise ValueError('collector policy permission and actual decisions disagree')
    cost_valid = np.asarray(physical['task_reward_valid'])
    if cost_valid.dtype != np.bool_ or cost_valid.shape != active.shape:
        raise ValueError('explicit aligned running-cost validity required')
    components = np.asarray(physical['effective_costs'], float)
    reward = np.asarray(physical['tick_reward'], float)
    keys = tuple(configured_cost_keys(config))
    if components.shape != (active.size, len(keys)) or reward.shape != active.shape:
        raise ValueError('configured effective cost channels and running reward required')
    rows = active & cost_valid
    if (not np.isfinite(components[rows]).all() or np.any(components[rows] < 0) or
            not np.isfinite(reward[rows]).all()):
        raise ValueError('finite nonnegative effective components required')
    integrated = np.full(active.shape, np.nan)
    integrated[rows] = scale * dt * components[rows].sum(axis=1)
    if not np.allclose(integrated[rows], -reward[rows], rtol=2e-5, atol=2e-6):
        raise ValueError('running reward does not match integrated effective costs')
    fault_fields = [np.asarray(physical[k]) for k in ('nonfinite', 'lower_fault', 'policy_fault')]
    if any(x.dtype != np.bool_ or x.shape != active.shape for x in fault_fields):
        raise ValueError('explicit aligned engineering fault fields required')
    action = np.zeros_like(active)
    action[np.flatnonzero(valid) * 4] = True
    result = resolve_goal_task_credit(actual_xy=physical['actual_xy'], goal_xy=goal_xy,
        radius_m=radius_m, running_tick_cost=integrated, task_cost_valid=cost_valid,
        active_tick=active, actor_action_valid=action,
        contact_failure=physical['contact_failure'], engineering_fault=np.logical_or.reduce(fault_fields),
        horizon_reached=physical['horizon_reached'], physical_tick=clock,
        window_start_tick=0, window_ticks=round(horizon_s / dt),
        collection_policy_id=collection_id, learner_policy_id=learner_id)
    return dict(credit=result, cost_keys=keys,
        goal_mask=result['goal_actor_mask'][::4], goal_targets=result['goal_targets'][::4],
        cost_mask=result['cost_actor_mask'][::4], cost_targets=result['cost_targets'][::4],
        survival_mask=result['physical']['actor_mask'][::4],
        survival_targets=result['physical']['targets'][::4])
