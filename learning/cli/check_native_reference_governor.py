"""Sealed native reference-cost engineering preview; no closed-loop campaign."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(root, source):
    source = source.resolve()
    base_path = Path(__file__).resolve().parents[1] / 'src/sttw_control/native_zero_comparison.py'
    base = load_file('native_comparator_receipt', base_path)
    original = base.verify(source)
    status = json.loads((source / 'status.json').read_text())
    if status['phase'] != 'completed' or status['completed_cases'] != 8:
        raise ValueError('completed native8 source required')
    first = original['cases'][0]
    if (first['alpha'], first['speed']) != (0., 2.6):
        raise ValueError('first declared native context differs')
    module = base_path.with_name('path_reference_governor.py')
    initial = source / 'evaluation/initial_carry.pkl'
    files = [base_path, module, Path(__file__).resolve(), initial, source / 'plan.json', source / 'seal.json']
    root.mkdir(exist_ok=False)
    (root / 'source').mkdir()
    for file in (module, Path(__file__).resolve()):
        (root / 'source' / file.name).write_bytes(file.read_bytes())
    plan = dict(project='STTW_CONTROL',experiment_id='native_reference_cost_engineering_v1',
        question='Does a middle-NN-independent native full-carry forecast produce valid original-user costs and usable candidate distinctions?',
        source=str(source),native_runtime_root=original['native_runtime_root'],
        native_source_sha256=original['native_source_sha256'],
        files_sha256={str(f):digest(f) for f in files},initial_carry=str(initial),source_lane=0,
        first_case=first,config=original['config'],initial_source=original['initial_source'],
        candidate_module=str(module),levels=[-.95,-.475,0.,.475,.95],candidate_count=26,
        previous_target=[0.,0.],zero_candidate_index=13,preview_upper_steps=50,horizon_s=10.,
        maximum_new_prediction_lower_ticks=5200,maximum_allocated_prediction_lower_slots=5200,
        actual_controller_episodes=0,optimizer_updates=0,middle_actor_inferences=0,
        objective='Original native V3 complete running cost and original full-remaining-task native failure replacement; no new terminal value at prediction tail',
        selection='argmin original_user_cost over all finite known candidates; first exact tie retains previous; unknown is engineering error, not negative label',
        privileges='Complete simulator state for empirical comparison; not a deployable estimator or certified safety backup',
        future='No future-force interface; zero external force forecast, current observer/history retained',
        stop='Exactly one26-candidate1s engineering prediction. No full8, candidate/horizon tuning, training or retry automatically authorized by completion.',
        information='Either a correct usable forecast with measured cost, or identified wiring/model/cost obstacle. No positive control-benefit prerequisite.')
    save(root / 'plan.json', plan)
    save(root / 'seal.json', dict(plan_sha256=digest(root / 'plan.json')))
    save(root / 'status.json', dict(phase='prepared',new_prediction_lower_ticks=0,actual_controller_episodes=0))


def run(root, execute=False):
    plan = json.loads((root / 'plan.json').read_text())
    assert digest(root / 'plan.json') == json.loads((root / 'seal.json').read_text())['plan_sha256']
    for path, sha in {**plan['files_sha256'], **plan['native_source_sha256']}.items():
        if digest(path) != sha:
            raise ValueError('sealed input changed: ' + path)
    if execute and json.loads((root / 'status.json').read_text())['phase'] != 'prepared':
        raise ValueError('no implicit restart')
    native = Path(plan['native_runtime_root'])
    if any(n == 'sttw_control' or n.startswith('sttw_control.') for n in sys.modules):
        raise ValueError('native namespace must be clean')
    sys.path.insert(0, str(native))
    import jax
    import jax.numpy as j
    import numpy as np
    import pickle
    from sttw_control.path_command_training import load_environment
    from sttw_control.geometric_path import build_path
    from sttw_control.runtime import configure_compilation_cache
    output = root / ('engineering' if execute else 'shape')
    output.mkdir(exist_ok=False)
    started = time.monotonic()

    def update(phase, **fields):
        status = dict(phase=phase,pid=os.getpid(),elapsed_s=time.monotonic()-started,
                      actual_controller_episodes=0,optimizer_updates=0,**fields)
        save(output / 'status.json', status)
        if execute:
            save(root / 'status.json', status)

    try:
        update('initializing',new_prediction_lower_ticks=0)
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env, snapshot = load_environment(plan['config'],plan['initial_source'],'terminal_contract_v2')
        with Path(plan['initial_carry']).open('rb') as f:
            stacked = pickle.load(f)
        origin = jax.tree.map(lambda x:j.asarray(x[plan['source_lane']]),stacked)
        case = plan['first_case']
        if int(origin.tick) != 0 or float(origin.alpha) != case['alpha']:
            raise ValueError('original task0 carry required')
        if bool(origin.physical.failed | origin.fault | origin.domain_exit):
            raise ValueError('invalid original complete carry')
        path = build_path(case['builder'],env.physics.helpers.pose(snapshot.data),plan['config'])
        speed = j.float32(case['speed'])
        env.set_log_template(origin,path,speed)
        module = load_file('native_reference_cost_functions',plan['candidate_module'])
        grid = module.native_cost_candidates(plan['previous_target'],plan['config'])
        if len(grid['latents']) != plan['candidate_count']:
            raise ValueError('candidate contract differs')
        preview = module.make_native_cost_preview(env,upper_steps=plan['preview_upper_steps'],
                                                  horizon_s=plan['horizon_s'],collect_trace=True)
        batch = jax.jit(jax.vmap(lambda z:preview(origin,z,path,speed)[1:]))
        shaped = jax.eval_shape(batch,j.asarray(grid['latents']))
        if any(v.shape != (26,) for v in shaped[0].values()):
            raise ValueError('native26 summary shapes differ')
        save(output / 'shape.json',dict(summary={k:list(v.shape) for k,v in shaped[0].items()},
                                      source_task_tick=int(origin.tick),source_physical_tick=int(origin.physical.physical_tick),
                                      actual_controller_episodes=0,new_prediction_lower_ticks=0))
        if not execute:
            update('completed',new_prediction_lower_ticks=0)
            return
        if not any(d.platform == 'gpu' for d in jax.devices()):
            raise ValueError('GPU required for declared numerical cost check')
        update('compiling',new_prediction_lower_ticks=0)
        clock = time.monotonic()
        executable = batch.lower(j.asarray(grid['latents'])).compile()
        compile_s = time.monotonic()-clock
        update('predicting',new_prediction_lower_ticks=None,allocated_prediction_lower_slots=5200,compile_s=compile_s)
        clock = time.monotonic()
        summary, trace = executable(j.asarray(grid['latents']))
        jax.block_until_ready(summary)
        execution_s = time.monotonic()-clock
        summary, trace = jax.device_get((summary,trace))
        np.savez_compressed(output / 'prediction_traces.npz',**trace)
        np.savez_compressed(output / 'prediction_summary.npz',**summary)
        np.savez_compressed(output / 'candidates.npz',**{k:v for k,v in grid.items() if isinstance(v,np.ndarray)})
        observed = int(summary['observed_lower_ticks'].sum())
        assert observed == int(trace['active_tick'].sum()) <= 5200
        expected_cost = -np.where(trace['active_tick'],trace['scored_tick_reward'],0).sum(axis=(1,2))
        np.testing.assert_allclose(summary['original_user_cost'],expected_cost,rtol=3e-5,atol=3e-5)
        selected = module.choose_native_cost(summary)
        if int(origin.tick) != 0:
            raise ValueError('original actual carry was mutated')
        save(output / 'receipt.json',dict(phase='completed',new_prediction_lower_ticks=observed,
            allocated_prediction_lower_slots=5200,actual_controller_episodes=0,optimizer_updates=0,
            middle_actor_inferences=0,compile_s=compile_s,execution_s=execution_s,
            selected_index=selected,selected_latent=grid['latents'][selected].tolist(),
            scores=np.asarray(summary['original_user_cost']).tolist(),
            score_spread=float(np.ptp(summary['original_user_cost'])),
            native_physical_failure=np.asarray(summary['native_physical_failure']).tolist(),
            native_domain_exit=np.asarray(summary['native_domain_exit']).tolist(),
            actual_control_benefit='not_tested',scope='One engineering forecast only; no closed-loop governor execution or safety proof'))
        update('completed',new_prediction_lower_ticks=observed,allocated_prediction_lower_slots=5200)
    except BaseException as exc:
        update('error',error=repr(exc),new_prediction_lower_ticks=None if execute else 0)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','shape','execute'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--source',type=Path,default=Path('runs/native_equilibrium_rule_v1'))
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.stage == 'prepare':
            prepare(root,args.source)
        else:
            run(root,execute=args.stage=='execute')
    except BaseException:
        if args.stage == 'execute':
            from engineer_reference_bias import notify
            notify(root,'STTW：原生动态参考工程检查报错','已停止，保留错误与未知计费，不自动重试。')
        raise
    else:
        if args.stage == 'execute':
            from engineer_reference_bias import notify
            notify(root,'STTW：原生动态参考工程检查完成','一次26候选预测已返回；尚无闭环治理或抗扰增益结论，随后分析实际预测成本。')
