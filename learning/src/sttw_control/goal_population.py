"""Whole declared fresh population assembly for opt-in ordinary goal learning."""
import math
import numpy as np
import torch
from .goal_live_credit import resolve_live_goal_credit
from .goal_task_credit import goal_cost_population_complete


def assemble_goal_population(episodes, *, expected_episode_count, config,
                             collection_id, learner_id, horizon_s, radius_m):
    """Keep every declared episode, including no-action, unknown and failed rows.

    Caller authenticates inventory/source identity and unchanged collection
    parameters. Count/unique episode IDs catch accidental local omissions and
    duplicates, not deliberate replacement or false external declarations.
    No costs, old values or actions are recomputed from the changing learner.
    """
    if (isinstance(expected_episode_count, bool) or not isinstance(expected_episode_count, int) or
            expected_episode_count <= 0 or len(episodes) != expected_episode_count):
        raise ValueError('entire declared nonempty episode inventory required')
    ids = [e['episode_id'] for e in episodes]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('unique explicit episode IDs required')
    upper = float(config['timing']['upper_dt_s'])
    if (not math.isfinite(upper) or upper <= 0 or not math.isfinite(horizon_s) or horizon_s <= 0 or
            not math.isclose(round(horizon_s / upper) * upper, horizon_s, abs_tol=1e-8)):
        raise ValueError('positive complete-upper-aligned task horizon required')
    maximum = round(horizon_s / upper)
    sources = dict(observation='observation', critic_observation='critic_observation',
        context='goal_context', actions='raw_action')
    old = dict(old_log_prob='log_prob', old_goal_values='old_goal_value',
        old_cost_values='old_cost_value', old_physical_values='physical_value')
    arrays = {k: [] for k in (*sources, *old, 'goal_targets', 'cost_targets', 'physical_targets')}
    masks = {k: [] for k in ('goal', 'cost', 'survival')}
    credits, summaries = [], []
    for e in episodes:
        if e['collection_id'] != collection_id:
            raise ValueError('all episodes must come from the declared frozen policy snapshot')
        d, p = e['decisions'], e['physical']
        n = len(d['valid'])
        if n > maximum:
            raise ValueError('episode decisions exceed declared horizon')
        result = resolve_live_goal_credit(d, p, goal_xy=e['original_goal_xy'], radius_m=radius_m,
            config=config, collection_id=collection_id, learner_id=learner_id, horizon_s=horizon_s)
        credit = result['credit']; credits.append(credit)
        for key, source in sources.items():
            x = np.asarray(d[source])
            width = {'observation': 351, 'critic_observation': 352, 'context': 5, 'actions': 2}[key]
            if x.shape != (n, width):
                raise ValueError('aligned actual decision views/actions required')
            arrays[key].append(x)
        for key, source in old.items():
            x = np.asarray(d[source])
            if x.shape != (n,):
                raise ValueError('actual frozen scalar values and raw-action density required')
            arrays[key].append(x)
        for key in masks: masks[key].append(result[key + '_mask'])
        for key in ('goal_targets', 'cost_targets', 'physical_targets'):
            arrays[key].append(result['survival_targets' if key == 'physical_targets' else key])
        summaries.append(dict(episode_id=e['episode_id'], goal=credit['goal_outcome'],
            survival=credit['physical']['outcome'], cost_window_complete=credit['cost_window_complete'],
            physical_ticks=credit['physical']['physical_ticks'], actual_decisions=int(np.sum(d['valid']))))
    tensors = {k: torch.tensor(np.concatenate(v), dtype=torch.float32) for k, v in arrays.items()}
    # Invalid rows deliberately retain NaNs; the loss selects before NN forward.
    return dict(**{k: tensors[k] for k in sources},
        collected={k: tensors[k] for k in (*old, 'goal_targets', 'cost_targets', 'physical_targets')},
        masks={k: torch.tensor(np.concatenate(v), dtype=torch.bool) for k, v in masks.items()},
        fixed_denominator=expected_episode_count * maximum,
        cost_population_complete=goal_cost_population_complete(credits),
        summaries=summaries, collection_id=collection_id)
