"""Declared finite reference-correction interventions; never change alpha."""
import math
from .tolerance_comparison import chunk_schedule


def intervention_layout(declaration, rows):
    spec = declaration.get('recovery_intervention')
    if spec is None:
        return None
    start, end = spec['start_s'], spec['end_s']
    horizon = declaration['config']['training_future_phase_C']['episode_s']
    dt = declaration['config']['timing']['upper_dt_s']
    if not all(math.isfinite(t) and math.isclose(t/dt, round(t/dt), abs_tol=1e-8)
               for t in (start, end)) or not 0 < start < end < horizon:
        raise ValueError('intervention boundaries must align inside original horizon')
    if any(start < onset + declaration['duration_s'] - 1e-8 for onset in declaration['onsets_s']):
        raise ValueError('intervention must follow complete pulse')
    control = spec['control']
    policies = declaration['policies']
    masks = spec['target_channel_masks']
    if set(masks) != set(policies) or masks.get(control) != [1, 1]:
        raise ValueError('matched control and every arm require explicit masks')
    for name, record in policies.items():
        if any(record[k] != policies[control][k] for k in ('actor_sha256', 'checkpoint_sha256')):
            raise ValueError('intervention arms require same frozen actor and checkpoint')
        if len(masks[name]) != 2 or any(v not in (0, 1) for v in masks[name]):
            raise ValueError('only keep/zero latent-target channel masks supported')
    key = lambda r: tuple(r[k] for k in ('route', 'speed', 'alpha', 'force_N'))
    sources = {key(r): i for i, r in enumerate(rows) if r['policy'] == control}
    if len(sources) != sum(r['policy'] == control for r in rows):
        raise ValueError('duplicate control state')
    return dict(start_step=round(start/dt), end_step=round(end/dt),
                start_tick=round(start/declaration['config']['timing']['lower_dt_s']),
                end_tick=round(end/declaration['config']['timing']['lower_dt_s']),
                source_indices=[sources[key(r)] for r in rows],
                masks=[masks[r['policy']] for r in rows])


def intervention_schedule(onset, horizon, layout):
    schedule = chunk_schedule(onset, duration=horizon)
    if layout is None:
        return schedule
    result = []
    for start, length, branch in schedule:
        cuts = [start] + sorted(t for t in (layout['start_step'], layout['end_step'])
                               if start < t < start + length) + [start + length]
        result.extend((a, b-a, branch and a == start) for a, b in zip(cuts, cuts[1:]))
    return result


def masked_target(z, tick, mask, start_tick, end_tick):
    import jax.numpy as j
    return j.where((tick >= start_tick) & (tick < end_tick), z * mask, z)
