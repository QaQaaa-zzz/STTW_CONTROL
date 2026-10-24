"""Read existing baseline artifacts; never step physics or change source runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, inputs):
    path = Path(path).resolve()
    inputs[str(path)] = sha(path)
    return json.loads(path.read_text())


def differences(old, new, prefix=""):
    out = {}
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key), new.get(key)
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(a, dict) and isinstance(b, dict):
            out.update(differences(a, b, name))
        elif a != b:
            out[name] = dict(historical=a, current=b)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = Path.cwd()
    inputs = {}
    historical = Path('/home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/runs/path_phase_C_200_20261010')
    old = load(historical/'manifest.json', inputs)
    current = load(base/'runs/reserve_feature_expansion_v1/post_fixed_final_v1/high_nominal_retention/H75/declaration.json', inputs)
    candidate = old['source']['candidate']
    assert sha(candidate['actor']) == candidate['actor_sha256']
    assert candidate['actor_sha256'] == current['source']['candidate']['actor_sha256']
    rows = []
    expected = round(old['config']['training_future_phase_C']['episode_s']/old['config']['timing']['lower_dt_s'])
    for trace in sorted((historical/'zero_upper').glob('*.npz')):
        if trace.stem.endswith('_path'):
            continue
        inputs[str(trace.resolve())] = sha(trace)
        geometry = trace.with_name(trace.stem+'_path.npz')
        inputs[str(geometry.resolve())] = sha(geometry)
        with np.load(geometry, allow_pickle=False) as z:
            assert all(np.isfinite(z[k]).all() for k in ['xy', 'heading', 's'])
        with np.load(trace, allow_pickle=False) as z:
            assert len(z['time']) == expected
            np.testing.assert_allclose(z['time'], np.arange(expected)*.005, atol=2e-6, rtol=0)
            assert all(np.isfinite(z[k]).all() for k in z.files)
            for key in ['latent_z', 'offsets', 'target_offset', 'filtered_offset']:
                np.testing.assert_array_equal(z[key], np.zeros_like(z[key]))
            flags = {key: bool(z[key].any()) for key in ['physical_failure', 'contact_failure', 'domain_exit', 'policy_fault', 'lower_fault']}
            assert not any(flags.values())
            rows.append(dict(name=trace.stem, steps=expected, duration_s=expected*.005,
                             flags=flags, peak_abs_roll_rad=float(np.abs(z['phi']).max()),
                             path_rmse_m=float(np.sqrt(np.mean(z['path_cross_track']**2))),
                             max_abs_path_m=float(np.abs(z['path_cross_track']).max()),
                             xy_path=str(trace.resolve()), task_success='NOT_RESCORED'))
    assert len(rows) == 6
    dual = {}
    for alpha in [0, 1]:
        root = base/f'runs/dual_endpoint_tolerance_revisit_v1/alpha{alpha}/evaluation'
        manifest = load(root/'manifest.json', inputs)
        cases = load(root/'results.json', inputs)['cases']
        assert len(cases) == 18
        dual[str(alpha)] = dict(cases=18, native_task_success=sum(x['task_recovered'] for x in cases),
                                domain_exits=sum(x['domain_failure'] for x in cases),
                                physical_failures=sum(x['physical_failure'] for x in cases),
                                manifest=str((root/'manifest.json').resolve()),
                                horizon_s=manifest['config']['training_future_phase_C']['episode_s'])
    relevant = ['timing', 'path', 'follower', 'residual_interface', 'path_reward']
    report = dict(project='STTW_CONTROL', purpose='Check existing baselines before new comparisons',
                  verification='raw_NPZ_and_results_checked; no simulation rerun',
                  new_physics_ticks=0, new_optimizer_updates=0,
                  authenticated_lower_actor_sha256=candidate['actor_sha256'],
                  historical_zero_middle=rows, historical_zero_middle_lower_ticks=sum(x['steps'] for x in rows),
                  dual_native=dual,
                  old_vs_H75_config_differences=differences({k:old['config'][k] for k in relevant}, {k:current['config'][k] for k in relevant}),
                  historical_horizon_s=old['config']['training_future_phase_C']['episode_s'],
                  current_horizon_s=current['config']['training_future_phase_C']['episode_s'],
                  inputs_sha256=inputs, audit_source_sha256=sha(__file__))
    (args.output/'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(zero_middle_cases=len(rows), historical_lower_ticks=report['historical_zero_middle_lower_ticks'], dual_native=dual,
                          configuration_differences=len(report['old_vs_H75_config_differences']), new_physics_ticks=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
