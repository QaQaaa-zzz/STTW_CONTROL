"""Replay the real correction interface, never the physics or counterfactual Actor."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle

import jax
import jax.numpy as j
import numpy as np

from sttw_control.piecewise_reference import (
    matched_target_families, piecewise_action, validate_piecewise_schedule,
)
from sttw_control.reference_prefix import inverse_offset_target
from sttw_control.path_command_policy import correction_tick


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    worktree = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(exist_ok=False)
    source = worktree / 'runs/reference_prefix_return_feasibility_v1/evaluation'
    plan = json.loads((source / 'plan.json').read_text())
    rows = json.loads((source / 'results.json').read_text())['cases']
    authenticated = json.loads((worktree / 'runs/safe_reference_choice_audit_v3/analysis.json').read_text())
    declaration = Path(plan['source_declaration'])
    cfg = json.loads(declaration.read_text())['config']
    static, temporal = matched_target_families(20261024)
    roster = np.concatenate([static, temporal])
    delay, ends = validate_piecewise_schedule(.04, [.2, .2], .005)
    length = delay + ends[-1]
    sources = {str(p): digest(p) for p in [Path(__file__), source / 'plan.json', source / 'results.json', declaration]}
    for name in ['piecewise_reference', 'reference_prefix', 'path_command_policy']:
        p = worktree / f'learning/src/sttw_control/{name}.py'
        sources[str(p)] = digest(p)
    summaries = []
    for route, speed, force in [('left90_R1p1', 1.8, 32.), ('right90_R1p1', 1.5, -32.)]:
        matches = [x for x in rows if x['route'] == route and x['speed'] == speed
                   and x['force_N'] == force and x['candidate'] == 'continue_original']
        assert len(matches) == 1
        row = matches[0]
        snapshot = Path(row['snapshot'])
        assert digest(snapshot) == row['snapshot_sha256']
        raw_path = source / (row['name'] + '.npz')
        assert digest(raw_path) == authenticated['raw_sha256'][str(raw_path)]
        sources[str(snapshot)] = digest(snapshot)
        sources[str(raw_path)] = digest(raw_path)
        with snapshot.open('rb') as stream:
            state = pickle.load(stream)
        with np.load(raw_path) as data:
            nominal = j.asarray(data['nominal'][:length])
            parent = j.asarray(data['original_policy_latent'][:length])
        assert nominal.shape == parent.shape == (length, 2)
        assert np.isfinite(nominal).all() and np.isfinite(parent).all()
        requested = j.asarray(roster) + j.asarray(state.offsets)[None, None, :]
        latent, realized, clipped = jax.vmap(jax.vmap(lambda t: inverse_offset_target(t, cfg)))(requested)

        def profile(targets):
            def tick(carry, inputs):
                k, nom, original = inputs
                action, _ = piecewise_action(original, targets, j.array(ends), k,
                                             delay, True, j.array([True, True]))
                filtered, offsets, governed, flags = correction_tick(*carry, action, nom, cfg)
                return (filtered, offsets), (governed, flags['reference_clip_channels'], flags['rate_clipped'])
            return jax.lax.scan(tick, (j.asarray(state.filtered), j.asarray(state.offsets)),
                                (j.arange(length), nominal, parent))[1]

        governed, reference_clip, rate_clip = jax.jit(jax.vmap(profile))(latent)
        governed, reference_clip, rate_clip = map(np.asarray, (governed, reference_clip, rate_clip))
        active = governed[:, delay:delay + ends[-1]]
        # Joint dimensionless RMS only for detecting redundant profiles.
        differences = (active[16:, None] - active[None, :16]) / np.array([.15, .04])
        distances = np.sqrt(np.mean(differences ** 2, axis=(2, 3)))
        nearest = distances.min(axis=1)
        assert np.isfinite(governed).all()
        np.savez_compressed(out / (route + '_profiles.npz'), target_deltas=roster,
                            requested_targets=np.asarray(requested), realized_targets=np.asarray(realized),
                            target_clips=np.asarray(clipped), governed=governed,
                            baseline_nominal=np.asarray(nominal), baseline_parent=np.asarray(parent),
                            reference_clip=reference_clip, rate_clip=rate_clip,
                            temporal_to_static_normalized_rms=distances)
        summaries.append(dict(route=route, speed=speed, force_N=force,
                              capture_offsets=np.asarray(state.offsets).tolist(),
                              target_clipped_candidates=int(np.any(clipped, axis=(1, 2)).sum()),
                              reference_clipped_candidates=int(reference_clip.any(axis=(1, 2)).sum()),
                              rate_limited_candidates=int(rate_clip.any(axis=1).sum()),
                              temporal_distinct_from_all_static=int((nearest > 1e-6).sum()),
                              nearest_static_normalized_rms_range=[float(nearest.min()), float(nearest.max())]))
    receipt = dict(seed=20261024, static_candidates=16, temporal_candidates=16,
                   delay_ticks=delay, segment_end_ticks=ends,
                   target_deltas=roster.tolist(), contexts=summaries, sources=sources,
                   new_physics_ticks=0, training_steps=0, model_fits=0,
                   scope='Counterfactual interface profiles along recorded baseline nominal/parent inputs; no candidate dynamics, outcome or deployability inferred.')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
