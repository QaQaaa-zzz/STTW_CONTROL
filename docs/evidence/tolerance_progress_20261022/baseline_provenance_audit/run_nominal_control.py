"""Four declared zero-middle full-task baselines using existing batched evaluation."""
import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from sttw_control.preference_best import atomic_json, digest


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    source = Path('runs/reserve_feature_expansion_v1/post_fixed_final_v1/high_nominal_retention/H75/declaration.json').resolve()
    original = json.loads(source.read_text())
    cfg = copy.deepcopy(original['config'])
    cfg['stage_C_evaluation'].update(route_ids=['left90_R2', 'right90_R2'], low_speed_m_s=2.45, conflict_speed_m_s=2.65)
    cfg['network']['architecture'] = 'legacy'
    from sttw_control.path_command_policy import PathCommandActor, make_path_actor, validate_actor_parameters
    import jax.numpy as j
    zero = Path('runs/zero_middle_physical_left_v1/zero_middle_actor.pkl').resolve()
    with zero.open('rb') as stream:
        modulated = pickle.load(stream)
    legacy = dict(params={key: value for key, value in modulated['params'].items() if key.startswith('Dense_')})
    validate_actor_parameters(legacy, cfg)
    with np.load('runs/deployment_contract_audit_v1/real_prefix_inputs.npz') as z:
        real = z['upper_observation']
    random = np.random.default_rng(924402).uniform(-5, 5, (100, 351)).astype('float32')
    random[:, 336] = np.linspace(0, 1, 100)
    obs = j.asarray(np.concatenate([real, random]))
    old_output = np.asarray(make_path_actor(original['config']).apply(modulated, obs))
    new_output = np.asarray(PathCommandActor().apply(legacy, obs))
    np.testing.assert_array_equal(new_output, old_output)
    np.testing.assert_array_equal(new_output, 0.)
    actor_file = root/'constant_zero_actor.pkl'
    with actor_file.open('wb') as stream:
        pickle.dump(legacy, stream)
    from sttw_control.path_command_selection import cases
    mapping = [dict(name=name, builder=route, actual_requested_speed_m_s=speed) for name, route, speed in cases(cfg)]
    assert len(mapping) == 4
    files = [Path(__file__).resolve(), source, Path(__file__).with_name('NOMINAL_CONTROL_DESIGN.md').resolve()]
    files += [Path('learning/src/sttw_control')/(name+'.py') for name in ['path_command_evaluation', 'path_command_policy', 'path_command_training', 'path_command_env', 'contact_only_environment', 'controller', 'closed_loop_kernel', 'teleop_env']]
    plan = dict(project='STTW_CONTROL', purpose='Full10s nominal follower versus existing five-alpha H75',
                source_declaration=str(source), config=cfg, source=original['source'],
                original_H75_evaluation=str(source.parent/'evaluation'),
                constant_actor=str(actor_file), constant_actor_sha256=digest(actor_file),
                actor_semantics='legacy constant zero latent; no trained checkpoint', verified_equivalence_rows=len(obs),
                original_modulated_zero_actor=str(zero), original_modulated_zero_sha256=digest(zero),
                cases=mapping, declared_cases=4, maximum_lower_ticks=8000, training_steps=0,
                initialization='original complete prepared bank; genuine full-task follower rollout',
                source_sha256={str(x.resolve()): digest(x) for x in files},
                stop_rule='four cases only; preserve errors; no automatic retry or force extension')
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'status.json', dict(phase='prepared', completed_cases=0, declared_cases=4, active_lower_ticks=0, training_steps=0))
    sys.path.insert(0, str(Path('learning/cli').resolve()))
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_RUN.json', 'STTW_CONTROL 零中层完整10秒名义任务对照4例', root/'status.json')
    print(root, 'prepared; four cases,8000 maximum lower ticks,0 training')


def execute(root):
    import jax.numpy as j
    import jax
    from sttw_control.runtime import configure_compilation_cache
    from sttw_control.contact_only_environment import load_environment_contact_only
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.recovery_margin_scan import outcome_labels
    plan = json.loads((root/'plan.json').read_text())
    state = dict(phase='running', pid=os.getpid(), stage='loading', declared_cases=4,
                 completed_cases=0, active_lower_ticks=0, training_steps=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(root/'status.json', state)
    def account(ticks):
        state['active_lower_ticks'] += ticks
        if state['active_lower_ticks'] > plan['maximum_lower_ticks']:
            raise ValueError('Declared nominal physics budget exceeded')
        update()
    update()
    try:
        for path, sha in plan['source_sha256'].items():
            if digest(path) != sha:raise ValueError('Declared source changed '+path)
        if digest(plan['constant_actor']) != plan['constant_actor_sha256']:
            raise ValueError('Constant actor changed')
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        cfg = plan['config']
        env, initial = load_environment_contact_only(cfg, plan['source'], 'terminal_contract_v2')
        with Path(plan['constant_actor']).open('rb') as stream:
            params = jax.tree.map(j.asarray, pickle.load(stream))
        atomic_json(root/'manifest.json', dict(plan=plan, physical_identity=env.physics.bundle.identity,
                    termination='contact-only; stop on path-domain exit; physical UNKNOWN if incomplete',
                    constant_actor_has_no_trained_checkpoint=True))
        evaluator = BatchedPanelEvaluator(env, initial, cfg, status=update, account=account)
        traces = evaluator.evaluate(params, 0., root/'evaluation')
        results = []
        for row in plan['cases']:
            data = traces[row['name']]
            for key in ['latent_z', 'offsets', 'target_offset', 'filtered_offset']:
                np.testing.assert_array_equal(data[key], 0.)
            with np.load(root/'evaluation'/(row['name']+'_path.npz')) as z:
                goal_s = float(z['goal'])
            labels = outcome_labels(data, speed=row['actual_requested_speed_m_s'], goal_s=goal_s,
                                    working_roll=cfg['stage_B_baseline']['working_roll_rad'], horizon=10., hold_s=cfg['stage_B_baseline']['hold_s'])
            results.append(dict(**row, **labels, trace_sha256=digest(root/'evaluation'/(row['name']+'.npz'))))
        atomic_json(root/'results.json', dict(cases=results))
        update(phase='completed', stage='four_full_task_traces_ready', completed_cases=4)
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare_only else execute)(args.output.resolve())
