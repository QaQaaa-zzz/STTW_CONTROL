"""One sealed, prospective function check of frozen native reference policies.

Only the historical native package is imported inside run(). All methods keep
its reset, follower, filter, lower controller, reward and termination rules.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_model_inputs(plan):
    for key in ('actor', 'checkpoint'):
        if digest(plan['h75'][key]) != plan['h75'][key + '_sha256']:
            raise ValueError('frozen H75 identity changed: ' + key)
    spec = Path(plan['initial_source']['upper_run']) / 'alpha0/frozen_config_stage52.json'
    if digest(spec) != plan['initial_source']['upper_config_file_sha256']['0']:
        raise ValueError('native plant/frame normalization spec changed')


def prepare(root, protocol, review, source):
    protocol, review, source = map(lambda p: Path(p).resolve(), (protocol, review, source))
    proposal = json.loads(protocol.read_text())
    for file, sha in proposal['sources_sha256'].items():
        if digest(file) != sha:
            raise ValueError('candidate source changed: ' + file)
    old = json.loads((source / 'plan.json').read_text())
    check_model_inputs(old)
    assert old['candidate_count'] == 26 and old['preview_upper_steps'] == 50
    assert old['hold_upper_steps'] == 10 and old['horizon_s'] == 10.
    for file, sha in {**old['files_sha256'], **old['native_source_sha256']}.items():
        if digest(file) != sha:
            raise ValueError('native scientific source changed: ' + file)
    assert digest(old['initial_source']['prepared_bank']) == old['initial_source']['prepared_bank_sha256']
    for key in ('actor', 'checkpoint'):
        assert digest(old['h75'][key]) == old['h75'][key + '_sha256']
    contexts = [dict(route=f"{direction}{g['angle_deg']}_R{g['radius_m']:g}".replace('.', 'p'),
                     builder=direction + '90_R2', angle_deg=g['angle_deg'], radius_m=g['radius_m'],
                     speed=proposal['request_speed_m_s'])
                for g in proposal['geometries'] for direction in proposal['directions']]
    assert len(contexts) == 4
    roles = ['reference_grid_5hz', 'H75_5hz', 'H75_50hz', 'equilibrium_50hz', 'zero_50hz']
    cases = []
    for role in roles:
        for alpha in ([0.] if role == 'zero_50hz' else proposal['user_alphas']):
            for context in contexts:
                cases.append(dict(context, role=role, state_alpha=alpha,
                                  execution_alpha=None if role == 'zero_50hz' else alpha,
                                  case_id=f"{role}/{context['route']}/a{alpha:g}"))
    assert len(cases) == 52
    folder = Path(__file__).resolve().parent
    owned = [Path(__file__).resolve(), folder.parent.parent / 'cli/run_native_frozen_validation.py',
             folder / 'path_reference_governor.py', folder / 'native_equilibrium_rule.py']
    root.mkdir(exist_ok=False)
    (root / 'source').mkdir()
    for file in owned:
        (root / 'source' / file.name).write_bytes(file.read_bytes())
    (root / 'scientific_review.md').write_bytes(review.read_bytes())
    (root / 'protocol_candidate.json').write_bytes(protocol.read_bytes())
    files = [protocol, review, source / 'plan.json', source / 'seal.json',
             Path(old['initial_source']['prepared_bank']), Path(old['h75']['actor']), Path(old['h75']['checkpoint']),
             Path(old['initial_source']['upper_run']) / 'alpha0/frozen_config_stage52.json'] + owned
    plan = dict(proposal, source=str(source), code_revision=os.popen('git rev-parse HEAD').read().strip(),
                code_scope='New runner source snapshot; numerical native runtime sealed separately',
                native_runtime_root=old['native_runtime_root'], native_source_sha256=old['native_source_sha256'],
                input_sha256={str(f): digest(f) for f in files}, config=old['config'],
                initial_source=old['initial_source'], h75=old['h75'], contexts=contexts, cases=cases,
                role_order=roles, prediction_lanes=12, maximum_prediction_active_lower_ticks=2995200,
                padded_path_samples=4001, maximum_actual_slots=104000,
                rule=dict(source=str(folder / 'native_equilibrium_rule.py'),
                          budget_rad=old['config']['path_reward']['working_roll_start_rad'], grid_points=65),
                governor_source=str(folder / 'path_reference_governor.py'))
    save(root / 'plan.json', plan)
    save(root / 'effective_config.json', dict(common=plan['config'], per_context_path=contexts,
         padding='Outgoing straight extension only after original support; original goal unchanged'))
    save(root / 'seal.json', dict(plan_sha256=digest(root / 'plan.json')))
    save(root / 'status.json', dict(phase='prepared', actual_lower_ticks=0, prediction_lower_ticks=0,
                                  allocated_actual_slots=0, allocated_prediction_slots=0, optimizer_updates=0))


def run(root, shape_only=False):
    plan = json.loads((root / 'plan.json').read_text())
    assert digest(root / 'plan.json') == json.loads((root / 'seal.json').read_text())['plan_sha256']
    check_model_inputs(plan)
    for file, sha in {**plan['input_sha256'], **plan['native_source_sha256']}.items():
        if digest(file) != sha:
            raise ValueError('sealed input changed: ' + file)
    if not shape_only and json.loads((root / 'status.json').read_text())['phase'] != 'prepared':
        raise ValueError('no implicit restart')
    if any(n == 'sttw_control' or n.startswith('sttw_control.') for n in sys.modules):
        raise ValueError('native namespace must be clean')
    sys.path.insert(0, plan['native_runtime_root'])
    import jax
    import jax.numpy as j
    import numpy as np
    import pickle
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.path_command_reward import configured_cost_keys
    from sttw_control.geometric_path import build_path
    from sttw_control.direct_command_policy import q_ratio
    from sttw_control.runtime import configure_compilation_cache
    started = time.monotonic()
    output = root / ('shape' if shape_only else 'evaluation')
    output.mkdir(exist_ok=False)
    actual = prediction = actual_slots = prediction_slots = 0
    phase = 'initializing'

    def update(new_phase, **values):
        nonlocal phase
        phase = new_phase
        value = dict(phase=new_phase, pid=os.getpid(), elapsed_s=time.monotonic() - started,
                     actual_lower_ticks=actual, prediction_lower_ticks=prediction,
                     allocated_actual_slots=actual_slots, allocated_prediction_slots=prediction_slots,
                     optimizer_updates=0)
        value.update(values)
        save(output / 'status.json', value)
        if not shape_only:
            save(root / 'status.json', value)

    try:
        update('initializing')
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        cfg = plan['config']
        env, snapshot = load_environment(cfg, plan['initial_source'], 'terminal_contract_v2')
        pose = np.asarray(env.physics.helpers.pose(snapshot.data))
        paths_by_route = {}
        padding = []
        for context in plan['contexts']:
            local = copy.deepcopy(cfg)
            local['path'].update(turn_angle_rad=math.radians(context['angle_deg']), turn_radius_m=context['radius_m'])
            path = build_path(context['builder'], pose, local)
            old_n = len(path.s)
            count = plan['padded_path_samples'] - old_n
            assert count >= 0 and float(path.s[-1]) > 70.
            distance = (np.arange(count) + 1) * float(path.s[-1] - path.s[-2])
            heading = float(path.heading[-1])
            s = np.concatenate([np.asarray(path.s), float(path.s[-1]) + distance])
            xy = np.concatenate([np.asarray(path.xy), np.asarray(path.xy[-1]) + distance[:, None] * np.array([math.cos(heading), math.sin(heading)])])
            padded = path.replace(s=j.asarray(s), xy=j.asarray(xy),
                                  heading=j.asarray(np.r_[np.asarray(path.heading), np.repeat(heading, count)]),
                                  curvature=j.asarray(np.r_[np.asarray(path.curvature), np.zeros(count)]))
            assert np.array_equal(np.asarray(padded.xy)[:old_n], np.asarray(path.xy))
            assert np.array_equal(padded.goal, path.goal) and np.all(np.diff(np.asarray(padded.s)) > 0)
            paths_by_route[context['route']] = padded
            ff = np.arctan(env.cc.wheelbase * np.asarray(path.curvature) / math.cos(env.cc.caster))
            chi = np.clip((abs(ff / float(q_ratio(j.float32(context['speed']), env.cc))) - .18) / .12, 0, 1)
            angle = math.degrees(float(path.heading[-1] - path.heading[0]))
            expected = context['angle_deg'] * (1 if context['builder'].startswith('left') else -1)
            assert abs(angle - expected) < .01
            padding.append(dict(route=context['route'], original_samples=old_n, padded_samples=len(padded.s),
                                original_support_m=float(path.s[-1]), goal_s_m=float(path.goal),
                                integrated_angle_deg=angle, maximum_abs_curvature_m_inv=float(abs(path.curvature).max()),
                                original_reward_chi_max=float(chi.max())))
            np.savez_compressed(output / (context['route'] + '_path.npz'), **{k: np.asarray(getattr(padded, k)) for k in ('s', 'xy', 'heading', 'curvature', 'turn_end', 'goal')})
        cases = plan['cases']
        paths = jax.tree.map(lambda *x: j.stack(x), *[paths_by_route[c['route']] for c in cases])
        support = np.array([next(p['original_support_m'] for p in padding if p['route'] == c['route']) for c in cases])
        speeds = j.asarray([c['speed'] for c in cases])
        alphas = j.asarray([c['state_alpha'] for c in cases])
        states = jax.jit(jax.vmap(lambda p, v, a: env.reset(snapshot, p, v, a)))(paths, speeds, alphas)
        jax.block_until_ready(states)
        assert not np.any(np.asarray(states.fault | states.physical.failed | states.domain_exit))
        assert np.array_equal(np.asarray(states.tick), np.zeros(52))
        env.set_log_template(jax.tree.map(lambda x: x[0], states), paths_by_route[cases[0]['route']], speeds[0])
        save(output / 'initialization.json', dict(prepared_bank_sha256=plan['initial_source']['prepared_bank_sha256'],
             bank_index=plan['initial_source']['bank_index'], seed=plan['initial_source']['seed'], padding=padding,
             snapshot_source='Same complete original physical prepared snapshot, original native env.reset'))
        with (output / 'initial_carry.pkl').open('wb') as stream:
            pickle.dump(jax.device_get(states), stream)
        with Path(plan['h75']['actor']).open('rb') as stream:
            params = jax.tree.map(j.asarray, pickle.load(stream))
        actor = make_path_actor(cfg)
        governor = load_module('_frozen_validation_governor', plan['governor_source'])
        rule = load_module('_frozen_validation_equilibrium', plan['rule']['source'])
        preview = governor.make_native_cost_preview(env, upper_steps=50, horizon_s=10.)
        fixed = j.asarray(governor.native_cost_candidates([0., 0.], cfg)['latents'][1:])
        keys = list(configured_cost_keys(cfg))
        keep = ('active_tick', 'time', 'actual_xy', 'reference_xy', 'actual_forward_speed', 'wheel_speed_proxy', 'v_user',
                'path_cross_track', 'path_heading_error', 'path_progress', 'goal_section_signed_distance',
                'phi', 'phi_dot', 'actual_delta', 'peak_roll', 'physical_failure', 'contact_failure', 'nonfinite',
                'domain_exit', 'policy_fault', 'lower_fault', 'governed', 'nominal', 'applied_force_impulse',
                'scored_tick_reward', 'tick_reward', 'latent_z', 'offsets', 'filtered_offset', 'target_offset', 'lower_action')
        subset = lambda tree, a, b: jax.tree.map(lambda x: x[a:b], tree)
        assert all(c['role'] == 'H75_50hz' for c in cases[24:36])
        assert all(c['role'] == 'equilibrium_50hz' for c in cases[36:48])

        def propose(st, path, speed):
            obs, _, fault = env.observation(st, path, speed, 10.)
            return actor.apply(params, obs), fault

        propose_many = jax.jit(jax.vmap(propose))

        def one(st, target, path, speed):
            _, _, fault = env.observation(st, path, speed, 10.)
            end, logs = env.policy_step(st.replace(fault=st.fault | fault), target, path, speed, 10., j.zeros(5))
            return end, dict({k: logs[k] for k in keep}, effective_costs=j.stack([logs['effective_components'][k] for k in keys], axis=-1))

        def advance_fn(st, held):
            def step(s, _):
                # Only H75 lanes infer H75; ordinary baselines have no proposal dependency.
                mu, _ = jax.vmap(propose)(subset(s, 24, 36), subset(paths, 24, 36), speeds[24:36])
                static = jax.vmap(lambda x: rule.target(x.physical.raw, x.alpha, cfg,
                    lambda v: q_ratio(v, env.cc), plan['rule']['budget_rad'], plan['rule']['grid_points'])[0])(subset(s, 36, 48))
                targets = j.concatenate([held[:24], mu, static, j.zeros((4, 2))])
                return jax.vmap(one)(s, targets, paths, speeds)
            return jax.lax.scan(step, st, None, length=10)

        advance = jax.jit(advance_fn)
        def forecast_one(st, zs, path, speed):
            def candidate(z):
                end, summary = preview(st, z, path, speed)
                return dict(summary, end_progress_m=end.projection.progress)
            return jax.vmap(candidate)(zs)
        forecast = jax.jit(jax.vmap(forecast_one))
        first4 = subset(states, 0, 4)
        latents = j.concatenate([j.zeros((4, 1, 2)), j.broadcast_to(fixed, (4, 25, 2))], axis=1)
        forecast_shape = jax.eval_shape(forecast, first4, latents, subset(paths, 0, 4), speeds[:4])
        actual_shape = jax.eval_shape(advance, states, j.zeros((52, 2)))
        assert all(v.shape == (4, 26) for v in forecast_shape.values())
        assert actual_shape[1]['active_tick'].shape == (10, 52, 4)
        save(output / 'shape.json', dict(forecast_shape={k: list(v.shape) for k, v in forecast_shape.items()},
             actual_active_shape=[10, 52, 4], new_actual_lower_ticks=0, new_prediction_lower_ticks=0))
        if shape_only:
            update('completed', completed_cases=0)
            return
        if not any(d.platform == 'gpu' for d in jax.devices()):
            raise RuntimeError('GPU required for declared evaluation')
        save(output / 'manifest.json', dict(cases=cases, reward_components=keys, config=cfg,
             initial_source=plan['initial_source'], h75=plan['h75'], plan_sha256=digest(root / 'plan.json')))
        update('compiling')
        clock = time.monotonic()
        forecast_exe = forecast.lower(first4, latents, subset(paths, 0, 4), speeds[:4]).compile()
        advance_exe = advance.lower(states, j.zeros((52, 2))).compile()
        mu_exe = propose_many.lower(subset(states, 12, 24), subset(paths, 12, 24), speeds[12:24]).compile()
        compile_s = time.monotonic() - clock
        save(output / 'compilation.json', dict(compile_s=compile_s, warmup_integrations=0))
        (output / 'chunks').mkdir()
        (output / 'forecasts').mkdir()
        previous = j.zeros((12, 2))
        saved, timings = [], []
        for block in range(50):
            clock = time.monotonic()
            selected_targets, selections = [], []
            for group in range(3):
                a, b = 4 * group, 4 * (group + 1)
                zs = j.concatenate([previous[a:b, None, :], j.broadcast_to(fixed, (4, 25, 2))], axis=1)
                update('predicting', planning_block=block, alpha_group=group)
                summary = forecast_exe(subset(states, a, b), zs, subset(paths, a, b), speeds[a:b])
                jax.block_until_ready(summary)
                host = jax.device_get(summary)
                prediction += int(host['observed_lower_ticks'].sum())
                prediction_slots += 4 * 26 * 200
                np.savez_compressed(output / 'forecasts' / f'block{block:02d}_group{group}.npz', **host, candidate_latents=np.asarray(zs))
                if not np.all(host['finite']) or not np.isfinite(host['original_user_cost']).all():
                    raise RuntimeError('unknown forecast, no candidate deletion or automatic retry')
                if np.any(host['end_progress_m'] + cfg['follower']['lookahead_max_m'] >= support[a:b, None]):
                    raise RuntimeError('forecast reached appended support; cannot claim unchanged original route')
                chosen = np.argmin(host['original_user_cost'], axis=1)
                selected_targets.append(j.asarray(np.asarray(zs)[np.arange(4), chosen]))
                selections.extend(chosen.tolist())
            forecast_s = time.monotonic() - clock
            selected = j.concatenate(selected_targets)
            mu, fault = mu_exe(subset(states, 12, 24), subset(paths, 12, 24), speeds[12:24])
            jax.block_until_ready(mu)
            if np.any(fault) or not np.isfinite(np.asarray(mu)).all():
                raise RuntimeError('unknown H75 held proposal')
            held = j.concatenate([selected, mu, j.zeros((28, 2))])
            update('advancing', planning_block=block)
            clock = time.monotonic()
            states, log = advance_exe(states, held)
            jax.block_until_ready(states)
            advance_s = time.monotonic() - clock
            host = jax.device_get(log)
            actual += int(host['active_tick'].sum())
            actual_slots += 52 * 40
            np.savez_compressed(output / 'chunks' / f'block{block:02d}.npz', **host)
            saved.append(host)
            for lane in range(52):
                active_lane = host['active_tick'][:, lane]
                if np.any(host['path_progress'][:, lane][active_lane] + cfg['follower']['lookahead_max_m'] >= support[lane]):
                    raise RuntimeError('actual execution reached appended support')
            previous = selected
            timings.append(dict(block=block, forecast_s=forecast_s, advance_s=advance_s, selected=selections))
            save(output / 'timings.json', timings)
            update('running', completed_blocks=block + 1)
            if np.any(np.asarray(states.fault)) or np.any(host['policy_fault'] | host['lower_fault']):
                raise RuntimeError('native engineering fault; preserve partial observations')
        combined = {k: np.concatenate([x[k] for x in saved], axis=0) for k in saved[0]}
        rows = []
        for lane, case in enumerate(cases):
            raw = {k: v[:, lane].reshape((-1,) + v.shape[3:]) for k, v in combined.items()}
            mask = raw['active_tick'].astype(bool)
            raw = {k: v[mask] for k, v in raw.items()}
            file = output / (case['case_id'].replace('/', '__') + '.npz')
            np.savez_compressed(file, **raw)
            rows.append(dict(case, raw_file=str(file), raw_sha256=digest(file)))
        assert actual <= 104000 and prediction <= 2995200 and prediction_slots <= 3120000
        save(output / 'results.json', rows)
        update('completed', completed_cases=52)
    except BaseException as exc:
        update('error', error=repr(exc), inflight_prediction_cost_unknown=phase == 'predicting',
               inflight_actual_cost_unknown=phase == 'advancing')
        raise
