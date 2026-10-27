"""Opt-in complete-episode goal/cost credit, separate from full-window survival.

This is ordinary undiscounted Monte Carlo / PPO engineering, not a new policy
gradient, constrained optimizer, data collector or adopted production objective.
"""
import math
import numpy as np
from .survival_credit import resolve_survival_credit, survival_ppo_loss


def resolve_goal_task_credit(*, actual_xy, goal_xy, radius_m,
                             running_tick_cost, task_cost_valid, **episode):
    """Label original-goal entry and original-alpha pre-entry cost on legal actions.

    ``episode`` uses resolve_survival_credit's explicit physical clock, actions,
    terminal flags and frozen sampling/learner identities. Actions precede their
    physical tick. There is no discount or guessed action at a held tick.
    running_tick_cost is already integrated, nonnegative factual cost, excluding
    terminal replacements. The caller authenticates its original-alpha formula.
    Neither a domain exit nor lack of task cost erases observed physical XY.
    """
    active = np.asarray(episode['active_tick'])
    xy, goal = np.asarray(actual_xy, float), np.asarray(goal_xy, float)
    if (active.dtype != np.bool_ or active.ndim != 1 or xy.shape != (active.size, 2) or goal.shape != (2,)
            or not np.isfinite(goal).all() or not math.isfinite(radius_m) or radius_m <= 0):
        raise ValueError('aligned actual XY and finite original goal/radius required')
    valid = np.asarray(task_cost_valid)
    cost = np.asarray(running_tick_cost, float)
    if valid.dtype != np.bool_ or valid.shape != active.shape or cost.shape != active.shape:
        raise ValueError('explicit Boolean task-cost validity and aligned integrated costs required')
    # Invalid measured XY is an engineering unknown, never invented failure or
    # goal entry. The existing resolver validates all other event/clock fields.
    fault = np.asarray(episode['engineering_fault'])
    if fault.dtype != np.bool_ or fault.shape != active.shape:
        raise ValueError('Boolean aligned engineering fault required')
    fault = fault | (active & ~np.isfinite(xy).all(axis=1))
    physical = resolve_survival_credit(**dict(episode, engineering_fault=fault))
    observed = active & (np.arange(active.size) <= (
        physical['terminal_index'] if physical['terminal_index'] is not None else active.size - 1))
    contact = np.asarray(episode['contact_failure'])
    hits = np.flatnonzero(observed & ~contact & ~fault &
                         (np.linalg.norm(xy - goal, axis=1) <= radius_m))
    hit = int(hits[0]) if hits.size else None
    if hit is not None:
        outcome, target, end = 'arrived', 1., hit
    elif physical['outcome'] in ('contact', 'survived'):
        outcome, target, end = 'not_arrived', 0., physical['terminal_index']
    else:
        outcome, target, end = physical['outcome'], np.nan, None
    action = np.asarray(episode['actor_action_valid'])
    # A known early goal is not permission to update sampling parameters while
    # the same physical episode is still being collected for survival credit.
    update_ready = physical['terminal_index'] is not None
    goal_mask = (observed & action & (np.arange(active.size) <= end)
                 if end is not None and update_ready else np.zeros_like(active))
    targets = np.full(active.size, np.nan, dtype=np.float32)
    targets[goal_mask] = target
    cost_mask = np.zeros_like(active)
    cost_targets = np.full(active.size, np.nan, dtype=np.float64)
    cost_window_complete = False
    eligible = observed & valid & ~contact & ~fault
    if not np.isfinite(cost[eligible]).all() or np.any(cost[eligible] < 0):
        raise ValueError('valid task costs must be finite nonnegative integrated costs')
    if end is not None:
        # Count the goal-entry transition, exclude contact/fault transitions.
        cost_rows = observed & (np.arange(active.size) <= end) & ~contact & ~fault
        cost_window_complete = bool(np.all(valid[cost_rows]))
        for index in np.flatnonzero(goal_mask):
            suffix = cost_rows & (np.arange(active.size) >= index)
            if np.all(valid[suffix]):
                cost_targets[index] = cost[suffix].sum(dtype=np.float64)
                cost_mask[index] = True
    return dict(goal_outcome=outcome, goal_target=float(target),
                goal_entry_index=hit, goal_terminal_index=end,
                physical_episode_resolved=update_ready,
                goal_actor_mask=goal_mask, goal_targets=targets,
                cost_actor_mask=cost_mask, cost_targets=cost_targets,
                cost_window_complete=cost_window_complete,
                physical=physical, gamma_goal=1., gamma_cost=1.,
                collection_policy_id=episode['collection_policy_id'])


def goal_cost_population_complete(credits):
    """Check the ENTIRE collected population, not a selected complete-case subset.

    Caller authenticates that no collected episodes were dropped before this
    check. Known individual suffixes alone do not justify the original cost
    policy-gradient objective when missingness depends on future behaviour.
    """
    if not credits or len({c['collection_policy_id'] for c in credits}) != 1:
        raise ValueError('one nonempty unchanged-policy collection population required')
    return all(c['physical_episode_resolved'] and c['cost_window_complete'] for c in credits)


def goal_task_ppo_loss(*, new_log_prob, old_log_prob,
                       goal_values, old_goal_values, goal_targets, goal_mask,
                       cost_values, old_cost_values, cost_targets, cost_mask,
                       physical_values, old_physical_values, physical_targets, physical_mask,
                       objective_weights, fixed_denominator, cost_population_complete, clip=.2):
    """Three separate PPO terms, with explicit caller weights for engineering use.

    This weighted sum is NOT a lexicographic/chance-constrained solver and does
    not guarantee goal priority, alpha fidelity or a survival performance floor.
    Networks must be evaluated only on the union of legal resolved masks BEFORE
    calling this loss. Caller must freeze old values/log-prob for all PPO epochs.
    """
    import torch
    if (set(objective_weights) != {'goal', 'cost', 'survival'} or
            any(isinstance(w, bool) or not isinstance(w, (int, float)) or
                not math.isfinite(w) or w < 0 for w in objective_weights.values()) or
            not any(objective_weights.values())):
        raise ValueError('explicit nonnegative goal/cost/survival weights required')
    masks = [goal_mask, cost_mask, physical_mask]
    if (not isinstance(new_log_prob, torch.Tensor) or not new_log_prob.is_floating_point() or new_log_prob.ndim != 1 or
            not all(isinstance(m, torch.Tensor) and m.dtype == torch.bool and
                    m.shape == new_log_prob.shape and m.device == new_log_prob.device for m in masks)):
        raise ValueError('aligned Boolean objective masks required')
    if not bool((goal_mask | cost_mask | physical_mask).any()):
        raise ValueError('no resolved legal decisions; do not update Actor')
    if not isinstance(cost_population_complete, bool) or (
            objective_weights['cost'] > 0 and not cost_population_complete):
        raise ValueError('original cost Actor objective requires complete full-population cost coverage')
    count = int((goal_mask | cost_mask | physical_mask).sum())
    if (isinstance(fixed_denominator, bool) or not isinstance(fixed_denominator, (float, int)) or
            not math.isfinite(fixed_denominator) or fixed_denominator < count):
        raise ValueError('fixed denominator must cover all selected legal decisions')
    # Empty differentiable sum avoids touching NaNs on invalid/unresolved rows.
    zero = new_log_prob[:0].sum()
    terms, value_losses = {}, {}
    for name, values, old_values, targets, mask in [
        ('goal', goal_values, old_goal_values, goal_targets, goal_mask),
        ('survival', physical_values, old_physical_values, physical_targets, physical_mask)]:
        if bool(mask.any()):
            loss = survival_ppo_loss(new_log_prob, old_log_prob, values, targets, mask,
                old_values=old_values, clip=clip, fixed_denominator=fixed_denominator)
            terms[name], value_losses[name] = loss['actor_loss'], loss['value_loss']
        else:
            terms[name], value_losses[name] = zero, zero
    if bool(cost_mask.any()):
        # Validate before indexing/arithmetic. Cost heads are scalar values, not
        # probabilities, and may be negative predictions during early fitting.
        tensors = [new_log_prob, old_log_prob, cost_values, old_cost_values, cost_targets]
        if not all(isinstance(x, torch.Tensor) and x.is_floating_point() and
                   x.shape == cost_mask.shape and x.device == cost_mask.device for x in tensors):
            raise ValueError('aligned floating cost vectors required')
        new, old, values, baseline, targets = [x[cost_mask] for x in tensors]
        if not all(bool(torch.isfinite(x).all()) for x in [new, old, values, baseline, targets]):
            raise ValueError('selected costs must be known and finite')
        if bool((targets < 0).any()):
            raise ValueError('cost targets must be nonnegative')
        if (not math.isfinite(clip) or not 0 < clip < 1 or
                isinstance(fixed_denominator, bool) or not isinstance(fixed_denominator, (float, int)) or
                not math.isfinite(fixed_denominator) or fixed_denominator < int(cost_mask.sum())):
            raise ValueError('explicit fixed denominator and legal PPO clipping required')
        ratio = (new - old.detach()).exp()
        if not bool(torch.isfinite(ratio).all()):
            raise ValueError('nonfinite PPO ratio')
        advantage = baseline.detach() - targets.detach()  # Maximize negative cost.
        terms['cost'] = -torch.minimum(ratio * advantage,
            ratio.clamp(1 - clip, 1 + clip) * advantage).sum() / fixed_denominator
        value_losses['cost'] = (values - targets.detach()).square().mean()
    else:
        terms['cost'], value_losses['cost'] = zero, zero
    return dict(actor_loss=sum(objective_weights[k] * v for k, v in terms.items()),
                actor_terms=terms, value_losses=value_losses)


def goal_task_ppo_batch(evaluate_selected, observations, actions, *, masks,
                        collected, objective_weights, fixed_denominator,
                        cost_population_complete, clip=.2):
    """Select legal known rows before network forward; never evaluate held NaNs."""
    import torch
    if set(masks) != {'goal', 'cost', 'survival'}:
        raise ValueError('three explicit objective masks required')
    goal, cost, survival = (masks[k] for k in ('goal', 'cost', 'survival'))
    if (not isinstance(goal, torch.Tensor) or goal.ndim != 1 or
            not all(isinstance(m, torch.Tensor) and m.dtype == torch.bool and
                    m.shape == goal.shape and m.device == goal.device for m in masks.values())):
        raise ValueError('aligned Boolean masks required')
    expected = {'old_log_prob', 'old_goal_values', 'goal_targets', 'old_cost_values',
                'cost_targets', 'old_physical_values', 'physical_targets'}
    if set(collected) != expected or not all(
            isinstance(x, torch.Tensor) and x.is_floating_point() and
            x.shape == goal.shape and x.device == goal.device for x in collected.values()):
        raise ValueError('explicit aligned frozen collection vectors required')
    if not all(isinstance(x, torch.Tensor) and x.is_floating_point() and x.ndim == 2 and
               x.shape[0] == goal.numel() and x.device == goal.device for x in (observations, actions)):
        raise ValueError('aligned collected observation/action rows required')
    union = goal | cost | survival
    if not bool(union.any()):
        raise ValueError('no resolved legal decisions; skip network and optimizer')
    if not isinstance(cost_population_complete, bool) or (
            objective_weights.get('cost', 0) > 0 and not cost_population_complete):
        raise ValueError('original cost Actor objective requires complete full-population cost coverage')
    obs, action = observations[union].detach(), actions[union].detach()
    if not bool(torch.isfinite(obs).all() & torch.isfinite(action).all()):
        raise ValueError('selected pre-action data must be finite')
    new_log_prob, goal_values, cost_values, physical_values = evaluate_selected(obs, action)
    return goal_task_ppo_loss(new_log_prob=new_log_prob, goal_values=goal_values,
        cost_values=cost_values, physical_values=physical_values,
        goal_mask=goal[union], cost_mask=cost[union], physical_mask=survival[union],
        **{k: v[union].detach() for k, v in collected.items()},
        objective_weights=objective_weights, fixed_denominator=fixed_denominator,
        cost_population_complete=cost_population_complete, clip=clip)
