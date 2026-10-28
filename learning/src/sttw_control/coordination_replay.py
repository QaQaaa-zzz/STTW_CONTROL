"""Bounded saved-observation upper-chain replay; never advances physics.

Observations, nominal follower outputs, permission and path progress are external
saved inputs. Only coordinator memory and correction-filter state recur here.
This is not a reconstruction of lower measurements, ESO or a closed-loop plant.
"""
import ast
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

WINDOWS = ((0, 848, 860), (0, 188, 200), (1, 208, 220), (3, 800, 1296))


def _json(value):
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)) or hasattr(value, '__array__'):
        return np.asarray(value).tolist()
    return value


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function(path, name):
    """Reuse a pure NumPy reference without importing analyzer/native packages."""
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def initial_state(raw, start):
    if start <= 0 or start % 4:
        raise ValueError('saved predecessor and upper-boundary start required')
    return dict(prev_alpha=float(raw['coord_prev_alpha'][start]),
                latched=bool(raw['coord_latched_before'][start]),
                filtered=np.asarray(raw['filtered_offset'][start-1], np.float32).tolist(),
                offsets=np.asarray(raw['offsets'][start-1], np.float32).tolist(),
                held=None, next_tick=start)


def replay_window(raw, case, start, stop, callbacks, *, state=None):
    """Stateful engine with injectable pure actor/q/decision/filter/path callbacks.

    State is JSON serializable even within an upper hold block. The caller can
    resume at any lower tick; no logged state is reread after initialization.
    """
    required = ('active_tick', 'time', 'query_observation', 'coord_latents',
                'pre_upper_yaw', 'coord_nominal', 'nominal', 'coord_pre_progress',
                'coord_permission', 'coord_prev_alpha', 'coord_latched_before',
                'filtered_offset', 'offsets', 'alpha_exec', 'rule_latched',
                'coord_probabilities', 'coord_qmin', 'coord_task_mask',
                'coord_proxy_cost', 'coord_degraded', 'coord_guard_fallback_selected',
                'target_offset', 'governed', 'latent_z', 'composed_latent')
    missing = set(required)-set(raw)
    if missing:
        raise ValueError('missing replay fields: '+','.join(sorted(missing)))
    if not 0 <= start < stop <= len(raw['active_tick']) or not np.all(raw['active_tick'][start:stop]):
        raise ValueError('complete active saved interval required')
    np.testing.assert_allclose(raw['time'][start:stop], np.arange(start, stop)*.005, atol=3e-6, rtol=0)
    state = json.loads(json.dumps(_json(initial_state(raw, start) if state is None else state)))
    if state['next_tick'] != start:
        raise ValueError('resume clock mismatch')
    packets = []
    for tick in range(start, stop):
        if tick % 4 == 0:
            obs = np.asarray(raw['query_observation'][tick], np.float32)
            if obs.shape != (351,):
                raise ValueError('H75 observation351 required')
            candidates = np.repeat(obs[None], 3, axis=0)
            candidates[:, 336] = [0., .5, 1.]
            latents = np.asarray(callbacks['actor'](candidates), np.float32)
            probabilities, offsets = callbacks['probabilities'](obs, latents, float(raw['pre_upper_yaw'][tick]))
            nominal = np.asarray(raw['coord_nominal'][tick], np.float32)
            targets = callbacks['targets'](nominal, offsets)
            path_id, curvature = callbacks['path'](tick, float(raw['coord_pre_progress'][tick]))
            result = callbacks['decision'](targets, curvature, nominal, case['speed'],
                np.asarray(probabilities).min(axis=-1), bool(raw['coord_permission'][tick]),
                state['latched'], state['prev_alpha'], case['effective_user_alpha'])
            result = _json(result)
            alpha = float(result['alpha'])
            selected = int(np.argmin(abs(np.array([0., .5, 1.])-alpha)))
            latent = latents[selected]
            permission = bool(raw['coord_permission'][tick])
            state['held'] = dict(alpha=alpha, latched=bool(result['latched']),
                latent=latent.tolist(), issued=(latent if permission else np.zeros(2, np.float32)).tolist(),
                probabilities=_json(probabilities), qmin=_json(np.asarray(probabilities).min(axis=-1)),
                targets=_json(targets), task_mask=result['task_mask'], proxy_cost=result['proxy_cost'],
                degraded=result['degraded'], guard_selected=result['guard_fallback_selected'],
                permission=permission, path_id=path_id, curvature=float(curvature),
                prev_before=state['prev_alpha'], latch_before=state['latched'], latents=latents.tolist())
            state['prev_alpha'] = alpha
            state['latched'] = bool(result['latched'])
        held = state['held']
        if held is None:
            raise ValueError('resuming held block requires serialized held decision')
        filtered, offsets, governed, target = callbacks['filter'](
            np.asarray(state['filtered'], np.float32), np.asarray(state['offsets'], np.float32),
            np.asarray(held['issued'], np.float32), np.asarray(raw['nominal'][tick], np.float32))
        state['filtered'], state['offsets'] = _json(filtered), _json(offsets)
        state['next_tick'] = tick+1
        packets.append(dict(tick=tick, new_decision=tick % 4 == 0, **held,
            filtered=_json(filtered), offsets=_json(offsets), governed=_json(governed), target=_json(target)))
    return packets, state


# Post-observation engineering acceptance bounds, not scientific outcome criteria.
# Vector bounds are [speed m/s, steering rad]; latent/probability/proxy are unitless.
COMPARISONS = {
    'alpha': ('alpha_exec', 0), 'latched': ('rule_latched', 0),
    'latent': ('composed_latent', 2e-4), 'issued': ('latent_z', 2e-4),
    'latents': ('coord_latents', 2e-4),
    'probabilities': ('coord_probabilities', 1e-4), 'qmin': ('coord_qmin', 1e-4),
    'task_mask': ('coord_task_mask', 0), 'proxy_cost': ('coord_proxy_cost', 2e-4),
    'degraded': ('coord_degraded', 0), 'guard_selected': ('coord_guard_fallback_selected', 0),
    'permission': ('coord_permission', 0),
    'filtered': ('filtered_offset', [1e-4, 2e-5]),
    'offsets': ('offsets', [1e-4, 2e-5]), 'governed': ('governed', [1e-4, 2e-5]),
    'target': ('target_offset', [2e-4, 4e-5]),
    'prev_before': ('coord_prev_alpha', 0), 'latch_before': ('coord_latched_before', 0)}


def compare_packets(packets, raw, start, stop):
    """Collect every field error before enforcing bounded values/exact decisions."""
    errors, components, failures = {}, {}, []
    for name, (field, tolerance) in COMPARISONS.items():
        actual = np.asarray([p[name] for p in packets]); expected = raw[field][start:stop]
        difference = abs(actual.astype(float)-expected.astype(float))
        errors[name] = float(np.max(difference))
        if difference.ndim > 1:
            components[name] = np.max(difference, axis=tuple(range(difference.ndim-1))).tolist()
        bounded = np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
        bounded = bounded and bool(np.all(difference <= np.asarray(tolerance)))
        if not bounded:
            failures.append(name)
    q = np.asarray([p['qmin'] for p in packets]); logged = raw['coord_qmin'][start:stop]
    gate_equal = {str(t): bool(np.array_equal(q >= t, logged >= t)) for t in (.8, .85)}
    failures.extend('q_gate_'+k for k, equal in gate_equal.items() if not equal)
    return dict(maximum_absolute_errors=errors, maximum_component_absolute_errors=components,
                exact_q_threshold_qualifications=gate_equal, failed_fields=failures)


def _callbacks(source, plan, runtime):
    base = Path(__file__).resolve().parents[3]
    actor_fn = _function(base/'docs/evidence/tolerance_progress_20261022/deployment_contract_audit/audit.py', 'numpy_actor')
    probability_fn = _function(base/'learning/cli/analyze_native_goal_confirmation.py', 'tolerance_numpy_probabilities')
    actor_file = Path(plan['models']['H75']['actor'])
    if _digest(actor_file) != plan['models']['H75']['actor_sha256']:
        raise ValueError('H75 actor identity drift')
    with actor_file.open('rb') as stream:
        parameters = pickle.load(stream)
    q_receipt = json.loads((source/'q/receipt.json').read_text())
    for name, sha in q_receipt['files_sha256'].items():
        if _digest(source/'q'/name) != sha:
            raise ValueError('q export identity drift')
    if _digest(source/'q/receipt.json') != plan['frozen_predictor_export']['receipt_sha256']:
        raise ValueError('q receipt differs from plan')
    with np.load(source/'q/parameters.npz') as data:
        q = dict(data)
    import jax
    coordination = _module(Path(__file__).with_name('tolerance_coordination.py'), '_upper_replay_coordinator')
    policy = _module(Path(plan['native_runtime_root'])/'sttw_control/path_command_policy.py', '_upper_replay_native_policy')
    choose = jax.jit(lambda targets, curvature, nominal, speed, quality, permission, latch, previous, user:
        coordination.choose_alpha(targets, curvature, nominal, speed, quality, permission, latch,
                                  previous, use_q=True, user_alpha=user, task_aware_fallback=True, guard_fallback=True))
    correction = jax.jit(lambda f, o, z, n: policy.correction_tick(f, o, z, n, plan['config']))
    interface = plan['config']['residual_interface']
    def probabilities(obs, latents, yaw):
        p, o = probability_fn(q, obs[None], latents[None], np.array([yaw], np.float32), interface, 0)
        return p[0], o[0]
    def targets(nominal, offsets):
        lo, hi = interface['speed_reference_bounds_m_s']; steer = interface['steer_reference_bound_rad']
        return np.clip(nominal[None]+offsets, np.array([lo, -steer], np.float32), np.array([hi, steer], np.float32))
    def filtering(f, o, z, n):
        ff, oo, gg, flags = correction(f, o, z, n)
        return np.asarray(ff), np.asarray(oo), np.asarray(gg), np.asarray(flags['target'])
    return dict(actor=lambda obs: actor_fn(parameters, obs, modulated=True),
                probabilities=probabilities, targets=targets, decision=choose, filter=filtering)


def replay_upper_windows(source, output, *, runtime=None):
    """Write four bounded stateful upper replays; output must be fresh."""
    source, output = Path(source).resolve(), Path(output).resolve()
    plan = json.loads((source/'plan.json').read_text())
    if plan['mode'] != 'guard_fallback_evaluation' or plan['protocol']['stage'] != 'extension':
        raise ValueError('declared four windows require extension saved source')
    seal = json.loads((source/'seal.json').read_text())
    if _digest(source/'plan.json') != seal['plan_sha256']:
        raise ValueError('source plan drift')
    index = json.loads((source/'evaluation/case_index.json').read_text())
    decisions = json.loads((source/'evaluation/replanning_decisions.json').read_text())
    if isinstance(decisions, dict):
        decisions = decisions['decisions']
    callbacks = _callbacks(source, plan, runtime)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for number, (lane, start, stop) in enumerate(WINDOWS):
        case = index[lane]
        if _digest(case['raw_file']) != case['raw_sha256']:
            raise ValueError('raw identity drift')
        with np.load(case['raw_file']) as data:
            raw = dict(data)
        with np.load(source/'evaluation'/f"{case['path_key']}_path.npz") as data:
            original = dict(data)
        changed = [d for d in decisions if d['case_index'] == lane]
        replan_tick = changed[0]['task_tick'] if changed else None
        replanned = None
        if changed:
            with np.load(source/'evaluation'/f'replanned_path_{lane:02d}.npz') as data:
                replanned = dict(data)
        def path(tick, progress):
            bound = replanned is not None and tick >= replan_tick
            pp = replanned if bound else original
            lower = np.clip(progress, pp['s'][0], pp['s'][-1]); upper = min(lower+6., pp['s'][-1])
            mask = (pp['s'] >= lower)&(pp['s'] <= upper)
            curvature = max(abs(np.interp(lower, pp['s'], pp['curvature'])), abs(np.interp(upper, pp['s'], pp['curvature'])), float(np.max(abs(pp['curvature'][mask]))) if mask.any() else 0.)
            return ('replanned' if bound else 'original'), curvature
        cb = dict(callbacks, path=path)
        packets, final = replay_window(raw, case, start, stop, cb)
        # Intentionally split inside a held block, not just on a20ms boundary.
        split = start+(stop-start)//2+1
        if split >= stop:
            split = start+1
        first, state = replay_window(raw, case, start, split, cb)
        resumed, resumed_final = replay_window(raw, case, split, stop, cb, state=json.loads(json.dumps(state)))
        if first+resumed != packets or resumed_final != final:
            raise ValueError('serialization/resume changed stateful replay')
        comparison = compare_packets(packets, raw, start, stop)
        arrays = {name: np.asarray([p[name] for p in packets]) for name in packets[0] if name != 'path_id'}
        arrays.update(input_observation351=raw['query_observation'][start:stop], input_nominal=raw['nominal'][start:stop],
            input_nominal_at_decision=raw['coord_nominal'][start:stop], input_yaw=raw['pre_upper_yaw'][start:stop],
            input_path_progress=raw['coord_pre_progress'][start:stop], input_permission=raw['coord_permission'][start:stop],
            input_user_alpha=np.full(stop-start, case['effective_user_alpha']), path_replanned=np.array([p['path_id']=='replanned' for p in packets]))
        np.savez_compressed(output/f'window_{number}.npz', **arrays)
        records.append(dict(case_id=case['case_id'], raw_sha256=case['raw_sha256'], start_tick=start,
            stop_tick_exclusive=stop, initial_state=initial_state(raw, start), final_state=final,
            resume_tick=split, serialized_resume_exact=True, **comparison,
            replan_task_tick=replan_tick, path_bindings=sorted(set(p['path_id'] for p in packets)),
            packets=stop-start, decisions=sum(p['new_decision'] for p in packets)))
    receipt = dict(source=str(source), source_plan_sha256=_digest(source/'plan.json'),
        actor_sha256=plan['models']['H75']['actor_sha256'],
        q_receipt_sha256=_digest(source/'q/receipt.json'),
        replay_helper_sha256=_digest(__file__),
        physics_steps=0, optimizer_updates=0, windows=records, packet_dt_s=.005, decision_dt_s=.02,
        state_contract='Initialize prev/latch and predecessor filtered/offsets once; recursively carry thereafter, including serialized held decision. Logged351 observation/nominal/permission/progress/yaw are external inputs.',
        limitation='No lower210 observation, motor command, sensor history or ESO is reconstructed. H75/q/decision/filter software replay is not closed-loop sensor or plant replay; numeric comparisons are to saved outputs, not independent safety validation.',
        acceptance_bounds=_json(COMPARISONS),
        acceptance_role='Post-observation engineering numeric bounds, not scientific gain/loss criteria. Discrete outputs and q .8/.85 qualifications remain exact.',
        implementation='Existing NumPy actor and independent NumPy q, existing coordinator choose_alpha and frozen native correction_tick; no copied controller equations.')
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if any(w['failed_fields'] for w in records):
        raise ValueError('upper replay mismatch; all four-window errors saved in receipt.json')
    return receipt
