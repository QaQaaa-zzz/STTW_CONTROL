"""Prepare zero-latent and unchanged mode3 controls for the completed H75 panel."""
import argparse
import copy
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze

from prepare_preference_history_factorial import sha, write
from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
from sttw_control.reference_reserve_gate import GateConfig


def zero_output_actor(parameters):
    """Preserve conditioning/hidden parameters, zero the final affine output."""
    result = unfreeze(parameters)
    values = result['params'] if 'params' in result else result
    for field in ('kernel', 'bias'):
        values['Dense_3'][field] = np.zeros_like(values['Dense_3'][field])
    return jax.tree.map(np.asarray, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, source, output = Path.cwd(), args.source.resolve(), args.output.resolve()
    old_plan = json.loads((source / 'plan.json').read_text())
    status = json.loads((source / 'status.json').read_text())
    process = json.loads((source / 'process.json').read_text())
    assert status['phase'] == process['stage'] == 'completed' and process['exit_code'] == 0
    assert status['completed_cases'] == 24
    for name, expected in old_plan['source_sha256'].items():
        assert sha(name) == expected, name
    cache = json.loads((source / 'analysis_verified/review.json').read_text())
    assert len(cache['records']) == 24
    for record in cache['records']:
        assert sha(record['raw_file']) == record['raw_sha256']
    output.mkdir(exist_ok=False)
    declaration = Path(old_plan['source_declaration'])
    cfg = json.loads(declaration.read_text())['config']
    model = old_plan['actors']['H75']
    assert sha(model['path']) == model['sha256']
    parameters = pickle.loads(Path(model['path']).read_bytes())
    zero = zero_output_actor(parameters)
    validate_actor_parameters(zero, cfg)
    actor = make_path_actor(cfg)
    rng = np.random.default_rng(20261027)
    obs = rng.uniform(-5., 5., (257, 351)).astype(np.float32)
    obs[:, 336] = np.linspace(-5., 5., len(obs))
    values = np.asarray(actor.apply(jax.tree.map(jnp.asarray, zero), jnp.asarray(obs)))
    np.testing.assert_array_equal(values, np.zeros((len(obs), 2)))
    zero_path = output / 'zero_output_actor.pkl'
    zero_path.write_bytes(pickle.dumps(zero))
    cases = []
    for original in old_plan['cases']:
        if original['future_alpha'] != 0.:
            continue
        for role, mode in (('zero_latent', 0), ('speed_support', 3)):
            row = copy.deepcopy(original)
            row.update(policy=role, baseline_role=role, gate_mode=mode)
            cases.append(row)
    assert len(cases) == 24
    files = list((base / 'learning/src/sttw_control').glob('*.py'))
    files += [Path(__file__).resolve(), base / 'learning/cli/prepare_preference_history_factorial.py',
              base / 'learning/cli/probe_sensor_gate_domain_hold.py', declaration,
              Path(model['path']), zero_path, source / 'plan.json', source / 'status.json',
              source / 'analysis_verified/review.json']
    files += [Path(r['snapshot']) for r in cases]
    files += [Path(r['raw_file']) for r in cache['records']]
    sources = {str(p.resolve()): sha(p) for p in files}
    archive = output / 'source_at_launch'
    for p in files:
        if p.suffix == '.py':
            target = archive / p.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    plan = {k: copy.deepcopy(old_plan[k]) for k in ('project', 'source_declaration',
            'path_evidence_directory', 'sensor_contract', 'pulse_duration_s', 'observation_s',
            'direction_contract', 'termination', 'primary_outcome')}
    plan.update(experiment_id='preference_history_baselines_v1',
                question='Do zero latent or existing speed-only support cover same-state future-alpha gains?',
                source_run=str(source), source_plan_sha256=sha(source / 'plan.json'),
                actors={'zero_latent': dict(path=str(zero_path), sha256=sha(zero_path)),
                        'speed_support': model}, cases=cases, source_sha256=sources,
                gate_config=vars(GateConfig()), declared_cases=24, cached_cases=24,
                maximum_lower_ticks=19200, maximum_allocated_lower_slots=21600,
                new_prefix_ticks=0, training_steps=0,
                baseline_contract='zero final affine outputs; mode3=max(original future0 speed latent, atanh(.96)), original steering unchanged',
                scope='DEV matched mechanism controls; no original-task or deployment benefit claim',
                stop_rule='Fixed 24 new branches once; preserve gains/losses on all12 conditions; no force/time/gate tuning')
    write(output / 'plan.json', plan)
    write(output / 'baseline_reuse.json', dict(source=str(source), cases=cache['records'],
                                               inherited_lower_ticks=cache['actual_lower_ticks'],
                                               new_cached_lower_ticks=0))
    write(output / 'zero_output_receipt.json', dict(source_actor_sha256=model['sha256'],
                zero_actor_sha256=sha(zero_path), changed_parameters='Dense_3.kernel and bias only',
                affine_output_has_no_bypass=True, sampled_observations=257,
                sampled_outputs_exactly_zero=True, real_snapshot_outputs_pending=True,
                warning='Zero latent is not zero governed correction immediately; existing filter/slew/offset carry remain'))
    write(output / 'status.json', dict(phase='prepared', declared_cases=24, completed_cases=0,
                                        active_lower_ticks=0, training_steps=0))
    print(json.dumps(dict(output=str(output), new_cases=24, cached_cases=24, new_physics_ticks=0)))


if __name__ == '__main__':
    main()
