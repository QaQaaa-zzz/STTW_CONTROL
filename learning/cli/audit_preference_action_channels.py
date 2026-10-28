"""Audit alpha's immediate neural action channels on authenticated saved observations.

No environment, simulator, controller update, or training module is imported. The
archived policy implementation is loaded by its exact file path, independently of
the current worktree's Python package. Outputs are unsmoothed target offsets, not
executed references or actuator actions.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Identity changed: {path}: {actual} != {expected}')
    return actual


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def numpy_forward(parameters, observation, alpha_index, architecture):
    """Independent float32 Dense -> modulation -> ELU implementation."""
    p = parameters.get('params', parameters)
    x = np.asarray(observation, dtype=np.float32)
    q = np.float32(2) * x[..., alpha_index:alpha_index+1] - np.float32(1)
    for index in range(4):
        layer = p[f'Dense_{index}']
        x = x @ np.asarray(layer['kernel']) + np.asarray(layer['bias'])
        if index == 3:
            break
        if architecture == 'alpha_modulated_v1' and index > 0:
            x = ((np.float32(1) + q * np.asarray(p[f'gain_{index+1}'])) * x
                 + q * np.asarray(p[f'shift_{index+1}']))
        negative = x <= 0
        x = x.copy()
        x[negative] = np.expm1(x[negative])
    return x


def numpy_targets(latent, config):
    c = config['residual_interface']
    a = np.tanh(latent)
    return np.stack([
        a[..., 0] * np.where(a[..., 0] >= 0, c['speed_positive_scale_m_s'],
                            c['speed_negative_scale_m_s']),
        a[..., 1] * c['steer_scale_rad']], axis=-1)


def load_policy(source_file):
    if any(k == 'sttw_control' or k.startswith('sttw_control.') for k in sys.modules):
        raise RuntimeError('Start a clean Python process without current sttw_control imports')
    spec = importlib.util.spec_from_file_location('authenticated_archived_path_policy', source_file)
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)
    return policy


def synthetic_forward_check(policy, config):
    """Nonzero modulation exercises both gain and shift, independent of H75 values."""
    import jax.numpy as j
    rng = np.random.default_rng(80421)
    p = {}
    widths = (351, 128, 128, 64, 2)
    for i in range(4):
        p[f'Dense_{i}'] = {
            'kernel': rng.normal(0, .035, (widths[i], widths[i+1])).astype('float32'),
            'bias': rng.normal(0, .1, widths[i+1]).astype('float32')}
    for key, shape in policy.MODULATION_SHAPES.items():
        p[key] = rng.normal(.1, .2, shape).astype('float32')
    obs = rng.normal(0, .5, (3, 351)).astype('float32')
    obs[:, policy.ALPHA_INDEX] = [0, .5, 1]
    parameters = {'params': p}
    actual = np.asarray(policy.make_path_actor(config).apply(parameters, j.asarray(obs)))
    independent = numpy_forward(parameters, obs, policy.ALPHA_INDEX, 'alpha_modulated_v1')
    np.testing.assert_allclose(actual, independent, atol=1e-6, rtol=0)
    return {'seed': 80421, 'synthetic_rows': 3, 'all_modulation_parameters_nonzero': True,
            'max_latent_error': float(np.max(np.abs(actual-independent))),
            'absolute_tolerance': 1e-6}


def provenance_rows(receipt_path, contract_path, actor_sha):
    receipt = json.loads(receipt_path.read_text())
    contract = json.loads(contract_path.read_text())
    cases = [x for x in receipt['command_cases'] if x['kind'] == 'authentic_prefix_next_control']
    metadata = {}
    evidence = {}
    for name, expected in receipt['snapshots_sha256'].items():
        path = Path(name)
        if path.name != 'prefix_results.json':
            continue
        evidence[str(path)] = checked(path, expected)
        prefixes = json.loads(path.read_text())
        if not prefixes['complete_upper_history']:
            raise ValueError('Incomplete historical observation history')
        prepare = path.with_name('prepare_plan.json')
        plan = json.loads(prepare.read_text())
        if plan['model']['actor_sha256'] != actor_sha:
            raise ValueError('Prefix collection model identity differs')
        evidence[str(prepare)] = digest(prepare)
        for row in prefixes['cases']:
            metadata[row['state_id']] = row
    rows = []
    for case in cases:
        row = metadata[case['case_id']]
        expected = receipt['snapshots_sha256'][row['snapshot']]
        if expected != row['snapshot_sha256']:
            raise ValueError('Conflicting saved snapshot identities')
        evidence[row['snapshot']] = checked(row['snapshot'], expected)
        rows.append({key: row[key] for key in (
            'state_id', 'route', 'speed', 'alpha', 'fraction', 'task_time_s',
            'actual_progress_m', 'actual_speed_m_s', 'roll_rad', 'snapshot_sha256')})
    return rows, evidence, contract['upper_actor']['actor_sha256'] == actor_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--check-receipt', required=True, type=Path,
                        help='Frozen inference receipt containing input_source and input_sha256')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--input-key', default='upper_observation')
    parser.add_argument('--provenance-receipt', type=Path)
    parser.add_argument('--provenance-contract', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    os.environ['JAX_PLATFORMS'] = 'cpu'
    args.input = args.input.resolve()
    plan = json.loads(args.source_plan.read_text())
    check = json.loads(args.check_receipt.read_text())
    if Path(check['input_source']).resolve() != args.input:
        raise ValueError('Input path differs from frozen inference receipt')
    input_sha = checked(args.input, check['input_sha256'])
    actor = plan['h75']
    actor_sha = checked(actor['actor'], actor['actor_sha256'])
    config = plan['config']
    network = config['network']
    expected = (network['actor_dim'], network['hidden'], network['activation'], network['actions'])
    if expected != (351, [128, 128, 64], 'elu', 2):
        raise ValueError(f'Unsupported architecture dimensions: {expected}')
    if network['architecture'] != 'alpha_modulated_v1':
        raise ValueError('This audit requires the explicit alpha-modulated architecture')
    source_file = Path(plan['native_runtime_root'])/'sttw_control/path_command_policy.py'
    source_sha = checked(source_file, plan['native_source_sha256'][str(source_file)])
    receipt_path = args.provenance_receipt or args.input.with_name('receipt.json')
    contract_path = args.provenance_contract or args.input.with_name('CONTRACT.json')
    rows, evidence, same_stored_model = provenance_rows(receipt_path, contract_path, actor_sha)
    with np.load(args.input, allow_pickle=False) as data:
        obs = data[args.input_key].copy()
        stored = data['upper_latent'].copy() if same_stored_model else None
    if obs.shape != (16, 351) or obs.dtype != np.float32 or not np.all(np.isfinite(obs)):
        raise ValueError('Bounded audit requires exactly 16 finite float32 observations of width351')
    if len(rows) != len(obs):
        raise ValueError('Provenance row count differs from saved observations')
    alpha_offset = network['history_frames']*network['frame_fields'] + network['mask_fields']
    alpha_fields = [(i, x) for i, x in enumerate(network['context_fields']) if x['name'] == 'alpha']
    if len(alpha_fields) != 1 or alpha_fields[0][0] != 0:
        raise ValueError('Unexpected alpha context location')
    alpha_scale = alpha_fields[0][1]['scale']
    if alpha_scale != 1.0:
        raise ValueError('Modulated architecture requires alpha scale1 for q=2*obs_alpha-1')
    if not np.array_equal(obs[:, alpha_offset], [x['alpha']/alpha_scale for x in rows]):
        raise ValueError('Saved alpha slot does not agree with prefix provenance')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = {'input': str(args.input), 'input_key': args.input_key, 'input_sha256': input_sha,
                'actor': actor, 'source_plan': str(args.source_plan.resolve()),
                'source_plan_sha256': digest(args.source_plan), 'archived_policy': str(source_file),
                'archived_policy_sha256': source_sha, 'check_receipt_sha256': digest(args.check_receipt),
                'provenance_receipt_sha256': digest(receipt_path),
                'provenance_contract_sha256': digest(contract_path), 'config': config,
                'prefix_evidence_sha256': evidence, 'alpha_index': alpha_offset, 'alpha_scale': alpha_scale,
                'alpha_queries': [0, .5, 1], 'physical_change_tolerance': 1e-6,
                'stored_latent_model_identity_matches': same_stored_model,
                'physics_steps': 0, 'optimizer_updates': 0,
                'implementation': str(Path(__file__).resolve()), 'implementation_sha256': digest(__file__),
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    # Freeze model/data/source identities before importing the inference runtime.
    write_json(output/'identity.json', identity)
    import jax
    import jax.numpy as j
    policy = load_policy(source_file)
    if policy.ALPHA_INDEX != alpha_offset:
        raise ValueError('Archive ALPHA_INDEX disagrees with configuration context offset')
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('CPU-only audit required')
    with Path(actor['actor']).open('rb') as stream:
        parameters = pickle.load(stream)
    policy.validate_actor_parameters(parameters, config)
    synthetic = synthetic_forward_check(policy, config)
    # Only this normalized input slot changes. Every other saved bit is retained.
    queries = np.repeat(obs[:, None, :], 3, axis=1)
    queries[:, :, alpha_offset] = np.asarray([0, .5, 1], np.float32)/alpha_scale
    others = np.arange(obs.shape[1]) != alpha_offset
    if not np.array_equal(queries[..., others], np.repeat(obs[:, None, others], 3, axis=1)):
        raise AssertionError('A non-alpha observation feature changed')
    flat = queries.reshape(48, 351)
    latent = np.asarray(policy.make_path_actor(config).apply(parameters, j.asarray(flat)))
    independent = numpy_forward(parameters, flat, alpha_offset, network['architecture'])
    np.testing.assert_allclose(latent, independent, atol=5e-6, rtol=3e-5)
    physical = np.asarray(jax.vmap(lambda z: policy.map_latent(z, config))(j.asarray(latent)))
    independent_physical = numpy_targets(independent, config)
    np.testing.assert_allclose(physical, independent_physical, atol=1e-6, rtol=0)
    latent = latent.reshape(16, 3, 2)
    physical = physical.reshape(16, 3, 2)
    stored_error = None
    if stored is not None:
        indices = [int(np.flatnonzero(np.asarray([0, .5, 1]) == x['alpha'])[0]) for x in rows]
        original = latent[np.arange(16), indices]
        np.testing.assert_allclose(original, stored, atol=5e-6, rtol=3e-5)
        stored_error = float(np.max(np.abs(original-stored)))
    ranges = np.ptp(physical.astype(np.float64), axis=1)
    long_rows, range_rows = [], []
    for i, row in enumerate(rows):
        range_rows.append(dict(row, row_index=i, speed_range_m_s=float(ranges[i, 0]),
                               steer_range_rad=float(ranges[i, 1])))
        for k, alpha in enumerate([0, .5, 1]):
            long_rows.append(dict(row_index=i, state_id=row['state_id'], source_alpha=row['alpha'],
                                  query_alpha=alpha, latent_speed=float(latent[i,k,0]),
                                  latent_steer=float(latent[i,k,1]),
                                  target_speed_offset_m_s=float(physical[i,k,0]),
                                  target_steer_offset_rad=float(physical[i,k,1])))
    for filename, values in [('actions.csv', long_rows), ('row_ranges.csv', range_rows)]:
        with (output/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    channels = {}
    for channel, index in [('speed_m_s', 0), ('steer_rad', 1)]:
        channels[channel] = {'maximum_range': float(np.max(ranges[:,index])),
                             'median_range': float(np.median(ranges[:,index])),
                             'changed_above_1e_6_count': int(np.sum(ranges[:,index] > 1e-6)),
                             'denominator': 16}
    checked(args.input, input_sha)
    checked(actor['actor'], actor_sha)
    checked(source_file, source_sha)
    summary = {'status': 'completed', 'physics_steps': 0, 'optimizer_updates': 0,
               'real_observation_rows': 16, 'alpha_queries_per_row': 3, 'real_actor_queries': 48,
               'channels': channels, 'physical_change_tolerance': 1e-6,
               'max_jax_numpy_latent_error': float(np.max(np.abs(latent.reshape(48,2)-independent))),
               'max_jax_numpy_physical_error_by_channel': np.max(
                   np.abs(physical.reshape(48,2)-independent_physical), axis=0).tolist(),
               'stored_original_alpha_latent_max_error': stored_error,
               'synthetic_nonzero_modulation_check': synthetic,
               'all_other_350_features_bitwise_unchanged': True,
               'jax_version': jax.__version__, 'devices': [str(x) for x in jax.devices()],
               'elapsed_s': time.monotonic()-started}
    write_json(output/'summary.json', summary)
    table = '\n'.join(f"| {x['row_index']} | {x['state_id']} | {x['speed_range_m_s']:.8f} | {x['steer_range_rad']:.8f} |"
                      for x in range_rows)
    speed, steer = channels['speed_m_s'], channels['steer_rad']
    readme = f'''# H75 immediate alpha action-channel audit

This is a limited neural-input intervention, not a counterfactual rollout or control evaluation.
Exactly 16 saved observations × alpha 0/.5/1 were queried on CPU. Only normalized
alpha slot {alpha_offset} changed (context scale {alpha_scale}); all other 350 features,
including physical/control history and filters, remain bitwise unchanged. No synthetic
history, physics integration, controller update, optimizer update, or new rollout occurred.

## Frozen identity and provenance

- H75 Actor SHA256: `{actor_sha}`.
- Input SHA256: `{input_sha}`; key `{args.input_key}`, float32 [16,351].
- Architecture: archived make_path_actor, alpha_modulated_v1, 351→128→128→64→2,
  Dense→alpha gain/shift at hidden layers 2/3→ELU; q=2*normalized_alpha−1.
- Historical CONTRACT.json model identity matches this H75: {same_stored_model}.
  Original-alpha latent versus stored upper_latent max error: {stored_error}.
- The saved execution_source.py collected env.observation from complete real saved
  prefixes. receipt.json order is matched to prefix_results.json state_id, the two
  prefix receipts and all 16 snapshot hashes were checked. prepare_plan.json binds
  the same H75 actor. The prior audit's six synthetic motor-command cases are excluded.
- Distribution: historical left/right125_R2p2 single bends; requested speeds 2.45/2.65 m/s;
  original episode alpha0/1; turn progress fractions .2/.7, first 20ms boundary reaching
  each target. Full 16-frame histories are retained. These are H75-induced correlated
  states from familiar runs, not independent training seeds, the current native S
  panel, disturbance aftermath, or deployment measurements. Querying a new alpha
  with the old alpha-conditioned history is a local input probe, potentially outside
  the policy's naturally reached joint state/history distribution.

## Immediate target offset changes

The archived map_latent computes tanh then speed × .25 m/s for positive values or
×1.0 m/s for negative values, and steering ×.2 rad. These are unsmoothed physical
target offsets before the correction filter, slew, reference clipping and lower
controller; they are not applied reference changes or motor actions.

Physical numerical change threshold: 1e-6 in each channel's own units.
Speed range maximum {speed['maximum_range']:.9f} m/s, median {speed['median_range']:.9f} m/s;
{speed['changed_above_1e_6_count']}/16 rows change above threshold.
Steering range maximum {steer['maximum_range']:.9f} rad, median {steer['median_range']:.9f} rad;
{steer['changed_above_1e_6_count']}/16 rows change above threshold.
No magnitude comparison across m/s and rad is made.

| Row | Saved prefix | Speed range (m/s) | Steering range (rad) |
|---:|---|---:|---:|
{table}

Independent NumPy Dense/ELU/modulation forward versus archived JAX maximum latent
error: {summary['max_jax_numpy_latent_error']:.9g}. Physical error by channel:
{summary['max_jax_numpy_physical_error_by_channel']}. A separate seeded synthetic
parameter check with nonzero gains/shifts also passed (3 synthetic inputs; no physics).

These results establish only that alpha changes immediate H75 outputs on this
limited saved-input set. They do not establish closed-loop causal benefit, recovery,
tolerance, safety, functional necessity, or superiority over simple control methods.
Speed-only or steering-only causal conclusions require separately authorized
closed-loop channel interventions; none were performed here.

## Files

- identity.json freezes input/model/config/source identities before inference.
- actions.csv contains all 48 queried latents and physical target offsets.
- row_ranges.csv contains all 16 per-observation ranges and prefix metadata.
- summary.json contains channel summaries and independent numerical checks.
'''
    (output/'README.md').write_text(readme)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
