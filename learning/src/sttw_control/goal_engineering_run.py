"""Fixed one-lane/one-second real-plant engineering, never a training campaign."""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json, digest


def prepare(root):
    root = Path(root).resolve()
    mount = Path('/media/qy/6A7E3DB07E3D75C3')
    stat = os.statvfs('/home/qy/STTW_CONTROL')
    if not os.path.ismount(mount) or not root.is_relative_to(mount) or stat.f_bavail * stat.f_frsize < 50 * 1024**3:
        raise ValueError('mounted data storage and at least50GiB system space required')
    source = Path('runs/live_preference_collector_v1_attempt2').resolve()
    p = json.loads((source / 'plan.json').read_text())
    if json.loads((source / 'status.json').read_text())['phase'] != 'completed':
        raise ValueError('completed historical engineering source required')
    for key in ('actor', 'checkpoint'):
        if digest(p['model'][key]) != p['model'][key + '_sha256']:
            raise ValueError('source model changed')
    case = p['cases'][0]
    if case['force_N'] != 0 or case['profile'] != 0:
        raise ValueError('fixed first unforced profile0 lane required')
    context = p['contexts'][case['context']]
    inputs = [source / n for n in ('plan.json', 'effective_config.json', 'source_declaration.json',
        'execution/initial_carry.pkl', 'execution/physical.npz', 'execution/receipt.json')]
    inputs += [Path(p['model'][key]) for key in ('actor', 'checkpoint')]
    inputs += [Path(context['path']), Path(context['initial'])]
    declaration = json.loads((source / 'source_declaration.json').read_text())['source']
    inputs += [Path(declaration['upper_run']) / 'alpha0/frozen_config_stage52.json',
               Path(declaration['prepared_bank'])]
    inputs += [Path(declaration['candidate'][key]) for key in ('actor', 'checkpoint', 'config')]
    for key in ('path', 'initial'):
        if digest(context[key]) != context[key + '_sha256']:
            raise ValueError('immutable path/start changed')
    receipt = json.loads((source / 'execution/receipt.json').read_text())
    if digest(source / 'execution/physical.npz') != receipt['raw_physical_sha256']:
        raise ValueError('historical raw receipt mismatch')
    folder = Path(__file__).resolve().parent
    sources = sorted(folder.rglob('*.py'))
    sources.append(Path('learning/cli/check_goal_collector_engineering.py').resolve())
    root.mkdir(exist_ok=False); (root / 'source').mkdir()
    for f in sources:
        relative = f.relative_to(folder) if f.is_relative_to(folder) else Path('cli') / f.name
        dest = root / 'source' / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
    plan = dict(project='STTW_CONTROL', experiment_id='STTW-GOAL-COLLECTOR-ENGINEERING-V1',
        purpose='Real plant interface/shape/recording only, no control efficacy or training',
        source=str(source), case_index=0, case=case, original_horizon_s=10.,
        upper_steps=50, maximum_lower_ticks=200, allocated_lower_slots=200,
        num_envs=1, goal_radius_m=1., goal_distance_scale_m=10., optimizer_steps=0,
        initialization='weights_only: parent Actor/logstd; zero goal branch and new GCS heads; original complete carry/key',
        inputs={str(f): digest(f) for f in inputs}, sources={str(f): digest(f) for f in sources},
        historical_runtime_source_differences={f: dict(historical_sha256=sha, current_sha256=digest(f))
            for f, sha in p.get('source_sha256', {}).items() if Path(f).exists() and digest(f) != sha},
        recording_tolerance=dict(mean=1e-3, logp=1e-2, goal_value=1e-3, cost_value=1e-3, survival_value=1e-3),
        tolerance_scope='Recorded distribution/value numerical consistency, not historical state replay; permits minor CPU/GPU differences',
        code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        scope='Selected index before inspection; related historical DEV source, not independent test/repeat',
        comparison='Stored parent same-lane/key first200ticks for engineering illustration only; no exact replay gate',
        acceleration=['existing persistent compilation cache', 'real per-lane vmap1 parameters', 'one scan50 block'],
        stop='shape failure or one scan50; preserve errors, no restart/extension or threshold/force tuning',
        success='real shape and finite eligible observations/values/density/cost/clock; partial G/S remain pending',
        notification='Chinese Bark explicitly; global hook manual absent, no service/autofix added')
    atomic_json(root / 'plan.json', plan)
    atomic_json(root / 'seal.json', dict(plan_sha256=digest(root / 'plan.json')))
    atomic_json(root / 'status.json', dict(phase='prepared', active_lower_ticks=0, optimizer_steps=0))


def _authenticate(root):
    p = json.loads((root / 'plan.json').read_text())
    if digest(root / 'plan.json') != json.loads((root / 'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed plan changed')
    for f, sha in dict(p['inputs'], **p['sources']).items():
        if digest(f) != sha: raise ValueError('source identity changed: ' + f)
    return p


def execute(root, *, shape_only=False):
    root = Path(root).resolve()
    target = root / ('shape_status.json' if shape_only else 'status.json')
    if target.exists() and json.loads(target.read_text())['phase'] != 'prepared':
        raise ValueError('fresh prepared stage required, no implicit restart')
    state = dict(phase='running', stage='loading', pid=os.getpid(), active_lower_ticks=0, optimizer_steps=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time()); atomic_json(target, state)
    update()
    try:
        import torch
        import jax
        import jax.numpy as j
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .geometric_path import Path as GeometricPath
        from .path_command_policy import make_torch_path_network
        from .goal_networks import GoalNetworkBundle, make_goal_collection_evaluator
        from .goal_live_collection import GoalLiveCollectionState, make_goal_live_collection_step
        from .goal_live_credit import resolve_live_goal_credit
        from .anticipatory_run import flatten_ticks
        p = _authenticate(root); source = Path(p['source'])
        old = json.loads((source / 'plan.json').read_text()); case = p['case']
        cfg = json.loads((source / 'effective_config.json').read_text())
        decl = json.loads((source / 'source_declaration.json').read_text())
        torch.set_num_threads(4)
        saved = torch.load(old['model']['checkpoint'], map_location='cpu', weights_only=False)
        parent = make_torch_path_network(351, 2, cfg); parent.load_state_dict(saved['actor'], strict=True)
        np.testing.assert_array_equal(saved['log_std'].detach().numpy(), np.asarray(old['model']['log_std'], np.float32))
        net = GoalNetworkBundle(parent, cfg, saved['log_std']); net.eval()
        fingerprint = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env, _ = load_environment_contact_only(cfg, decl['source'], 'terminal_contract_v2')
        with (source / 'execution/initial_carry.pkl').open('rb') as f: old_carry = pickle.load(f)
        lane = jax.tree.map(lambda x: j.asarray(x[p['case_index']]), old_carry)
        if int(lane.state.tick) != 0 or int(lane.state.physical.physical_tick) != 700:
            raise ValueError('original full task start clock required')
        np.testing.assert_array_equal(lane.key, jax.random.fold_in(jax.random.PRNGKey(old['seed']), case['rng_fold_in']))
        np.testing.assert_allclose(lane.state.alpha, case['alpha'], rtol=0, atol=1e-7)
        carry = GoalLiveCollectionState(lane.state, lane.key, lane.gate, lane.clean_blocks, j.bool_(False))
        context = old['contexts'][case['context']]
        with np.load(context['path']) as z:
            path = GeometricPath(*(j.asarray(z[k]) for k in ('s', 'xy', 'heading', 'curvature', 'turn_end', 'goal')))
            goal = j.asarray([np.interp(float(z['goal']), z['s'], z['xy'][:, i]) for i in (0, 1)])
        env.set_log_template(carry.state, path, j.float32(case['speed']))
        params = net.sampling_parameters()
        with Path(old['model']['actor']).open('rb') as f: original_actor = pickle.load(f)
        if jax.tree.structure(params['actor']) != jax.tree.structure(original_actor):
            raise ValueError('parent Actor tree structure mismatch')
        for a, b in zip(jax.tree.leaves(params['actor']), jax.tree.leaves(original_actor)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        stack = lambda x: jax.tree.map(lambda y: y[None], x)
        batch_carry, batch_params, batch_path = stack(carry), stack(params), stack(path)
        speed = j.asarray([case['speed']], j.float32); profile = j.asarray([case['profile']], j.int32)
        origin_tick = int(lane.state.physical.physical_tick)
        dt = float(cfg['timing']['lower_dt_s'])
        pulse = j.asarray([[(origin_tick+case['pulse_onset_task_tick'])*dt,
            (origin_tick+case['pulse_onset_task_tick'])*dt+case['pulse_duration_s'], 0., 0., 0.]], j.float32)
        step = make_goal_live_collection_step(env, make_goal_collection_evaluator(cfg),
            horizon_s=10., radius_m=1., distance_scale_m=10.)
        @jax.jit
        def chunk(st, parameters):
            return jax.lax.scan(lambda x, _: jax.vmap(step)(x, parameters, parameters['log_std'],
                batch_path, speed, pulse, profile, goal[None]), st, None, length=50)
        update(stage='shape' if shape_only else 'compiling_first_real_block', parameter_fingerprint=fingerprint)
        if shape_only:
            shaped = jax.eval_shape(chunk, batch_carry, batch_params)
            if shaped[1][0]['goal_context'].shape != (50, 1, 5): raise ValueError('goal context shape mismatch')
            atomic_json(root / 'shape_receipt.json', dict(goal_context=[50, 1, 5], physical_slots=[50, 1, 4],
                real_plant_traced=True, physics_integrations=0, optimizer_steps=0))
            update(phase='completed', stage='shape_complete'); return
        if not (root / 'shape_receipt.json').exists(): raise ValueError('real shape stage must complete first')
        if not any(d.platform == 'gpu' for d in jax.devices()): raise ValueError('actual GPU required')
        started = time.monotonic()
        update(stage='inflight_single_block', active_lower_ticks=None, allocated_lower_slots=200,
               actual_cost_status='inflight_unknown_until_device_return')
        final, (decisions, logs) = chunk(batch_carry, batch_params); jax.block_until_ready(final)
        host, decisions, logs = jax.device_get((final, decisions, logs)); logs = flatten_ticks(logs)
        n = int(logs['active_tick'].sum())
        update(stage='saving_raw', active_lower_ticks=n, allocated_lower_slots=200, actual_cost_status='known_from_actual_logs')
        np.savez_compressed(root / 'decisions.npz', **decisions)
        np.savez_compressed(root / 'physical.npz', **logs)
        with (root / 'final_carry.pkl').open('wb') as f: pickle.dump(host, f)
        n = int(logs['active_tick'].sum()); update(stage='checking', active_lower_ticks=n, allocated_lower_slots=200)
        log_fault = logs['active_tick'] & (logs['nonfinite'] | logs['lower_fault'] | logs['policy_fault'])
        if n > 200 or np.any(host.state.fault) or np.any(log_fault):
            raise ValueError('physical engineering fault or integration count exceeded')
        d = {k: v[:, 0] for k, v in decisions.items()}; raw = {k: v[:, 0] for k, v in logs.items()}
        valid = d['valid']
        if not valid.any(): raise ValueError('no actual legal decision in engineering block')
        with torch.no_grad():
            obs = torch.tensor(d['observation'][valid]); view = torch.tensor(d['critic_observation'][valid])
            ctx = torch.tensor(d['goal_context'][valid]); actions = torch.tensor(d['raw_action'][valid])
            logp, g, c, s = net.evaluate_actions(obs, view, ctx, actions)
            mean = net(obs, view, ctx)[0]
        differences = {k: float(np.max(np.abs(a.detach().numpy() - b))) for k, a, b in (
            ('mean', mean, d['mean'][valid]), ('logp', logp, d['log_prob'][valid]),
            ('goal_value', g, d['old_goal_value'][valid]), ('cost_value', c, d['old_cost_value'][valid]),
            ('survival_value', s, d['physical_value'][valid]))}
        atomic_json(root / 'recording_differences.json', dict(actual=differences, tolerated=p['recording_tolerance']))
        if not all(np.isfinite(list(differences.values()))): raise ValueError('nonfinite recorded network outputs')
        if any(value > p['recording_tolerance'][key] for key, value in differences.items()):
            raise ValueError('recorded distribution/value mismatch beyond minor numerical tolerance; raw preserved')
        credit = resolve_live_goal_credit(d, raw, goal_xy=np.asarray(goal), radius_m=1., config=cfg,
            collection_id=fingerprint, learner_id=fingerprint, horizon_s=10.)['credit']
        if hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest() != fingerprint:
            raise ValueError('sampling parameters changed without optimizer')
        receipt = dict(actual_lower_ticks=n, allocated_lower_slots=200, actual_decisions=int(valid.sum()),
            partial=True, original_horizon_s=10., simulated_prefix_s=n*.005,
            goal_outcome=credit['goal_outcome'], survival_outcome=credit['physical']['outcome'],
            resolved_goal_actor_decisions=int(credit['goal_actor_mask'].sum()),
            resolved_survival_actor_decisions=int(credit['physical']['actor_mask'].sum()),
            old_value_max_abs_difference=differences, parameter_fingerprint_unchanged=True,
            optimizer_steps=0, elapsed_compile_and_block_s=time.monotonic()-started,
            raw_physical_sha256=digest(root/'physical.npz'), raw_decisions_sha256=digest(root/'decisions.npz'))
        atomic_json(root / 'receipt.json', receipt)
        _plot(root, source, path, raw)
        update(phase='completed', stage='engineering_complete', **receipt)
    except BaseException as exc:
        (root / ('shape_traceback.txt' if shape_only else 'traceback.txt')).write_text(traceback.format_exc())
        update(phase='error', error=repr(exc)); raise


def _plot(root, source, path, raw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(source / 'execution/physical.npz') as z:
        old_active = z['active_tick'][:200, 0]
        old_xy = z['actual_xy'][:200, 0][old_active]; old_reward = z['tick_reward'][:200, 0][old_active]
        old_time = (z['physical_tick'][:200, 0][old_active]+1)*.005
    active = raw['active_tick']; xy = raw['actual_xy'][active]
    t = (raw['physical_tick'][active]+1)*.005; reward = raw['tick_reward'][active]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].plot(np.asarray(path.xy)[:,0], np.asarray(path.xy)[:,1], ':', color='gray', label='Original route')
    ax[0].plot(old_xy[:,0], old_xy[:,1], '--', label='Stored parent prefix (same source/key)')
    ax[0].plot(xy[:,0], xy[:,1], label='Goal collector prefix')
    cloud = np.vstack([xy, old_xy]); lo = cloud.min(0)-.2; hi=cloud.max(0)+.2
    ax[0].set(xlim=(lo[0],hi[0]), ylim=(lo[1],hi[1]), xlabel='World X (m)', ylabel='World Y (m)')
    ax[0].set_aspect('equal', adjustable='box'); ax[0].legend(fontsize=6)
    for a, y, old, label in [(ax[1], reward, old_reward, 'Running reward / lower tick'),
                            (ax[2], np.cumsum(reward), np.cumsum(old_reward), 'Cumulative running reward')]:
        a.plot(old_time, old, '--', label='Stored parent prefix'); a.plot(t, y, label='Goal collector prefix')
        a.set(xlabel='Task time (s)', ylabel=label); a.legend(fontsize=6)
    fig.suptitle('Engineering only: left90 R1.1, request1.8m/s, alpha0.5, F=0, task64 parent; 1s of original10s')
    fig.tight_layout(); fig.savefig(root/'engineering_xy_rewards.png', dpi=150); plt.close(fig)
