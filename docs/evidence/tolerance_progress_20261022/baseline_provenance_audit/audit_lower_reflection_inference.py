"""Inference-only reflection diagnostic on authenticated stored local histories.

Reflected observations are constructed probes, not simulated reachable states.
No Actor, snapshot, history or physical state is modified on disk.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np

from sttw_control.lower_command import SCALES
from sttw_control.network import ResidualActor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(reference, output):
    fields = ['roll', 'roll_rate', 'steer', 'steer_rate', 'measured_forward_speed',
              'measured_world_yaw_rate', 'rear_speed_proxy', 'front_speed_proxy',
              'local_command_speed', 'local_command_steer', 'local_command_speed_rate',
              'local_command_steer_rate', 'previous_local_speed_reference',
              'previous_local_steer_reference', 'previous_final_front_rate',
              'previous_final_rear_rate', 'previous_applied_front_residual',
              'previous_applied_rear_residual', 'ESO_equilibrium_shift',
              'local_command_equilibrium_reference']
    odd = {0, 1, 2, 3, 5, 9, 11, 13, 14, 16, 18, 19}
    signs = j.array([-1. if i in odd else 1. for i in range(20)])
    policy = ResidualActor((256, 128), activation='elu')
    records, hashes, seen = [], {}, set()
    for bank in ('bank3', 'bank0'):
        plan_path = reference / bank / 'plan.json'
        plan = json.loads(plan_path.read_text())
        declaration = Path(plan['source_declaration'])
        assert digest(declaration) == plan['source_declaration_sha256']
        candidate = json.loads(declaration.read_text())['source']['candidate']
        assert candidate['normalizer']['fields'] == fields
        np.testing.assert_array_equal(np.asarray(SCALES), np.array(candidate['normalizer']['scales'], np.float32))
        assert digest(candidate['actor']) == candidate['actor_sha256']
        hashes[str(declaration)] = digest(declaration)
        hashes[candidate['actor']] = digest(candidate['actor'])
        with Path(candidate['actor']).open('rb') as f:
            params = jax.tree.map(j.asarray, pickle.load(f))
        infer = jax.jit(lambda obs: policy.apply(params, obs))
        for case in plan['cases']:
            path = Path(case['capture_snapshot'])
            if str(path) in seen:
                continue
            seen.add(str(path))
            assert digest(path) == case['capture_snapshot_sha256']
            hashes[str(path)] = digest(path)
            with path.open('rb') as f:
                state = pickle.load(f)
            lower = state.lower
            frames, mask = j.asarray(lower.frames), j.asarray(lower.mask)
            assert frames.shape == (10, 20) and mask.shape == (10,)
            obs = j.concatenate([j.clip(frames / SCALES, -5., 5.).reshape(-1), mask])
            mirror = j.concatenate([j.clip(frames * signs / SCALES, -5., 5.).reshape(-1), mask])
            np.testing.assert_array_equal(mirror[-10:], obs[-10:])
            action, reflected_action = np.asarray(infer(obs)), np.asarray(infer(mirror))
            expected = action * np.array([-1., 1.])
            error = reflected_action - expected
            records.append(dict(bank=bank, route=case['route'], requested_speed=case['speed'],
                                snapshot=str(path), valid_history_frames=int(np.asarray(mask).sum()),
                                action_from_stored_history=action.tolist(),
                                action_from_constructed_reflected_history=reflected_action.tolist(),
                                exact_reflection_action=expected.tolist(),
                                normalized_reflection_error=error.tolist(),
                                prelimit_command_reflection_error=(error * np.array([1.5, 10.])).tolist()))
    output.mkdir(exist_ok=False)
    errors = np.array([r['normalized_reflection_error'] for r in records])
    result = dict(records=records, frame_signs=dict(zip(fields, np.asarray(signs).tolist())),
                  max_abs_normalized_error=np.abs(errors).max(axis=0).tolist(),
                  sha256=hashes, new_physical_ticks=0, fits=0,
                  scope='Inference on original stored histories and constructed reflected histories. Not executed mirrored actions, physically reachable probes, or a closed-loop transfer test. Non-equivariance does not imply reflected closed-loop failure.')
    (output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    assert all(digest(path) == value for path, value in hashes.items())
    print(json.dumps({'records': len(records), 'max_abs_normalized_error': result['max_abs_normalized_error'], 'raw_unchanged': True}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.reference.resolve(), args.output.resolve())
