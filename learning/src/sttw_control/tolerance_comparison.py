"""Matched evaluation layout and bookkeeping without simulator dependencies."""
from itertools import product
import math


def chunk_schedule(onset, duration=20., dt=.02, max_chunk=50):
    """Split scans at the pulse onset, retaining every control interval."""
    if dt <= 0 or max_chunk < 1 or not math.isfinite(onset):
        raise ValueError("invalid timing")
    total = round(duration / dt)
    pulse = round(onset / dt)
    if not math.isclose(total * dt, duration, abs_tol=1e-9) or not math.isclose(pulse * dt, onset, abs_tol=1e-9) or not 0 <= pulse < total:
        raise ValueError("onset and duration must align with control ticks")
    result = []
    start = 0
    while start < total:
        end = min(start + max_chunk, total)
        if start < pulse < end:
            end = pulse
        result.append((start, end - start, start == pulse))
        start = end
    return result


def comparison_layout(protocol):
    lanes = [dict(policy=p, route=r, speed=s, alpha=a, force_N=f)
             for p, r, s, a, f in product(*(protocol[k] for k in
                 ("policies", "routes", "speeds", "endpoints", "forces")))]
    def context(row):
        return tuple(row[k] for k in ("policy", "route", "speed", "alpha"))
    zeros = {}
    for i, row in enumerate(lanes):
        if row["force_N"] == 0:
            key = context(row)
            if key in zeros:
                raise ValueError("duplicate zero-force context")
            zeros[key] = i
    try:
        return lanes, [zeros[context(row)] for row in lanes]
    except KeyError as exc:
        raise ValueError("missing matched zero-force branch") from exc


def paired_counts(rows, baseline, candidate, alpha):
    def collect(policy):
        result = {}
        for row in rows:
            if row["policy"] != policy or row["alpha"] != alpha:
                continue
            key = tuple(row[k] for k in ("route", "speed", "onset_s", "force_N"))
            if key in result:
                raise ValueError("duplicate paired case")
            if row["success"] not in (True, False):
                raise ValueError("missing binary outcome")
            result[key] = bool(row["success"])
        return result
    a, b = collect(baseline), collect(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("incomplete paired outcomes")
    improved = sum(not a[k] and b[k] for k in a)
    regressed = sum(a[k] and not b[k] for k in a)
    return dict(n=len(a), baseline_success=sum(a.values()), candidate_success=sum(b.values()),
                improved=improved, regressed=regressed, difference=improved-regressed)


def resolve_comparison_actors(campaign):
    """Resolve the fixed final actors only after every declared training job ends."""
    import json
    from pathlib import Path
    from .preference_best import digest
    root = Path(campaign).resolve()
    definition = json.loads((root / 'campaign.json').read_text())
    result = {}
    for name in definition['jobs']:
        job = root / name
        status = json.loads((job / 'status.json').read_text())
        if status.get('phase') != 'completed' or status.get('completed_updates') != 50:
            raise ValueError('training not complete: ' + name)
        plan = json.loads((job / 'plan.json').read_text())
        manifest = json.loads((job / 'manifest.json').read_text())
        receipt = json.loads((job / 'last_completed.json').read_text())
        checkpoint = Path(receipt['checkpoint'])
        actor = checkpoint.with_name('actor_0050.pkl')
        if receipt['update'] != 50 or checkpoint.name != 'update_0050.pt' or checkpoint.parent.resolve() != (job / 'checkpoints').resolve():
            raise ValueError('wrong final checkpoint: ' + name)
        if digest(checkpoint) != receipt['checkpoint_sha256']:
            raise ValueError('checkpoint changed: ' + name)
        # Bind the exported Flax actor to the authenticated learner checkpoint.
        import pickle
        import torch
        import numpy as np
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        with actor.open('rb') as stream:
            exported = pickle.load(stream)['params']
        if saved['update'] != 50 or saved['identity'] != manifest['identity']:
            raise ValueError('learner checkpoint identity mismatch')
        for i, layer in enumerate((0, 2, 4, 6)):
            for field, tensor in [('kernel', saved['policy'][f'actor.{layer}.weight'].numpy().T),
                                  ('bias', saved['policy'][f'actor.{layer}.bias'].numpy())]:
                if not np.array_equal(np.asarray(exported[f'Dense_{i}'][field]), tensor):
                    raise ValueError('exported actor differs from learner checkpoint')
        identity = manifest['identity']
        if identity['arm'] != plan['arm'] or identity['alpha'] != plan['alpha']:
            raise ValueError('wrong policy identity: ' + name)
        key = (plan['arm'] + '_update50', plan['alpha'])
        if key in result:
            raise ValueError('duplicate trained policy')
        result[key] = dict(actor=str(actor), actor_sha256=digest(actor), checkpoint=str(checkpoint), checkpoint_sha256=receipt['checkpoint_sha256'], identity=identity)
    if set(result) != {(arm + '_update50', alpha) for arm in ('baseline', 'static', 'dynamic') for alpha in (0, 1)}:
        raise ValueError('incomplete comparison policy set')
    plan = json.loads((root / definition['jobs'][0] / 'plan.json').read_text())
    package = json.loads(Path(plan['package']).read_text())
    for alpha in (0, 1):
        record = package['model_selection'][f'alpha{alpha}']
        for key in ('actor', 'checkpoint'):
            if digest(record[key]) != record[key + '_sha256']:
                raise ValueError('parent identity changed')
        result[('parent', alpha)] = record
    return result
