"""Finite reference prefix followed by the original frozen preference policy.

This is an offline opportunity probe, not an online safety controller.
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    from sttw_control.preference_best import atomic_json
    status = dict(phase='running', pid=os.getpid(), stage='loading',
                  declared_cases=len(plan['cases']), completed_cases=0,
                  active_lower_ticks=0, training_steps=0, model_fits=0)

    def update(**changes):
        status.update(changes, checked_unix=time.time())
        atomic_json(root / 'status.json', status)

    update()
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
        from sttw_control.path_command_reward import configured_cost_keys
        from sttw_control.reference_prefix import inverse_offset_target, prefix_action, prefix_schedule
        from sttw_control.piecewise_reference import piecewise_action, validate_piecewise_schedule
        from sttw_control.reference_confirmation import validate_path_evidence
        from sttw_control.geometric_path import build_path
        from sttw_control.balance_probe import make_lower_tick, classify_contact_balance
        from sttw_control.recovery_margin_scan import outcome_labels
        from sttw_control.runtime import configure_compilation_cache

        for path, sha in plan['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError('source identity changed: ' + path)
        if (root / 'manifest.json').exists():
            raise FileExistsError('immutable execution already started')
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        declaration = json.loads(Path(plan['source_declaration']).read_text())
        cfg = declaration['config']
        for key in ('actor', 'checkpoint'):
            record = declaration['policies']['H75']
            if hashlib.sha256(Path(record[key]).read_bytes()).hexdigest() != record[key + '_sha256']:
                raise ValueError('declared H75 identity mismatch: ' + key)
        horizon = plan['original_task_deadline_s']
        if horizon != cfg['training_future_phase_C']['episode_s']:
            raise ValueError('original task deadline differs')
        env, initial = load_environment_contact_only(cfg, declaration['source'], 'terminal_contract_v2')
        with Path(declaration['policies']['H75']['actor']).open('rb') as stream:
            parameters = jax.tree.map(j.asarray, pickle.load(stream))
        validate_actor_parameters(parameters, cfg)
        actor = make_path_actor(cfg)
        pose = np.asarray(env.physics.helpers.pose(initial.data))
        paths = {r['name']: build_path(r['builder'], pose, cfg) for r in declaration['routes']}
        evidence = Path(plan['path_evidence_directory'])
        path_checks = []
        for name, path in paths.items():
            with np.load(evidence / (name + '_path.npz')) as old:
                tolerance = float(plan.get('path_evidence_atol_m', 1e-6))
                difference = validate_path_evidence(path.xy, old['xy'], atol_m=tolerance)
                path_checks.append(dict(route=name, max_abs_difference_m=difference, atol_m=tolerance))
            np.savez_compressed(root / (name + '_path.npz'), xy=np.asarray(path.xy),
                                s=np.asarray(path.s), goal=np.asarray(path.goal))
        cases = plan['cases']
        cache = {}
        snapshots = []
        for row in cases:
            if hashlib.sha256(Path(row['snapshot']).read_bytes()).hexdigest() != row['snapshot_sha256']:
                raise ValueError('case snapshot identity differs')
            if row['snapshot'] not in cache:
                with Path(row['snapshot']).open('rb') as stream:
                    cache[row['snapshot']] = jax.tree.map(j.asarray, pickle.load(stream))
            s = cache[row['snapshot']]
            if bool(s.physical.failed | s.fault | s.domain_exit):
                raise ValueError('invalid authentic prefix')
            if float(s.alpha) != row['alpha'] or int(s.tick) % 4:
                raise ValueError('alpha or control-boundary mismatch')
            snapshots.append(s)
        stack = lambda values: jax.tree.map(lambda *items: j.stack(items), *values)
        states = stack(snapshots)
        path_batch = stack([paths[r['route']] for r in cases])
        speeds = j.asarray([r['speed'] for r in cases])
        starts = j.asarray([int(s.tick) for s in snapshots], dtype=j.int32)
        delay_ticks, prefix_ticks = prefix_schedule(
            float(plan.get('prefix_start_delay_s', 0)), plan['prefix_duration_s'], env.cc.dt)
        piecewise = 'piecewise_segment_durations_s' in plan
        if piecewise:
            checked_delay, segment_ends = validate_piecewise_schedule(
                float(plan.get('prefix_start_delay_s', 0)), plan['piecewise_segment_durations_s'], env.cc.dt)
            if checked_delay != delay_ticks or segment_ends[-1] != prefix_ticks:
                raise ValueError('piecewise schedule differs from declared prefix')
            segment_ends = j.asarray(segment_ends, dtype=j.int32)
        handback_ticks = delay_ticks + prefix_ticks
        deadline_tick = round(horizon / env.cc.dt)
        maximum_remaining_ticks = sum(deadline_tick - int(s.tick) for s in snapshots)
        if maximum_remaining_ticks > plan['maximum_new_lower_ticks'] or maximum_remaining_ticks <= 0:
            raise ValueError('remaining suffix budget exceeds declaration')
        if any(int(s.tick) + handback_ticks >= deadline_tick for s in snapshots):
            raise ValueError('prefix boundary or remaining deadline invalid')
        if piecewise:
            deltas = np.asarray([r['target_deltas'] for r in cases])
            if deltas.shape != (len(cases),len(segment_ends),2) or not np.isfinite(deltas).all():
                raise ValueError('invalid finite piecewise target roster')
            targets = j.stack([s.offsets[None,:] + j.asarray(r['target_deltas']) for s,r in zip(snapshots,cases)])
            fixed, realized, clipped = jax.vmap(jax.vmap(lambda target: inverse_offset_target(target,cfg)))(targets)
        else:
            targets = j.stack([s.offsets + j.asarray(r['target_delta']) for s, r in zip(snapshots, cases)])
            fixed, realized, clipped = jax.vmap(lambda target: inverse_offset_target(target, cfg))(targets)
        masks = j.asarray([r['override_mask'] for r in cases], dtype=j.bool_)
        enabled = j.any(masks, axis=1)
        headings = np.asarray([float(env.physics.helpers.pose(s.physical.data)[2]) for s in snapshots])
        directions = np.column_stack([-np.sin(headings), np.cos(headings), np.zeros(len(cases))])
        forces = np.asarray([r['force_N'] for r in cases])
        absolute = np.asarray([int(s.physical.physical_tick) * env.cc.dt for s in snapshots])
        pulses = j.asarray(np.column_stack([absolute, absolute + plan['pulse_duration_s'], forces[:, None] * directions]), dtype=j.float32)
        keys = configured_cost_keys(cfg)
        env.set_log_template(snapshots[0], paths[cases[0]['route']], speeds[0])
        lower_tick = make_lower_tick(env, snapshots[0].physical.governor.current_reference, contact_only=True)
        keep = ['active_tick', 'actual_xy', 'reference_xy', 'path_cross_track', 'path_heading_error',
                'path_progress', 'goal_section_signed_distance', 'actual_forward_speed', 'wheel_speed_proxy',
                'phi', 'phi_dot', 'actual_delta', 'peak_roll', 'physical_failure', 'nonfinite',
                'contact_failure', 'lower_fault', 'policy_fault', 'domain_exit', 'governed', 'nominal',
                'latent_z', 'offsets', 'filtered_offset', 'target_offset', 'applied_force_impulse',
                'scored_tick_reward', 'chi', 'reference_clip_channels', 'correction_rate_clipped']

        def normal(s, path, speed, pulse, start, target, mask, use):
            obs, _, fault = env.observation(s, path, speed, horizon)
            original = actor.apply(parameters, obs)
            if piecewise:
                action, active = piecewise_action(original,target,segment_ends,s.tick-start,delay_ticks,use,mask)
            else:
                action, active = prefix_action(original, target, s.tick - start, prefix_ticks, use, mask,
                                               start_ticks=delay_ticks)
            end, logs = env.policy_step(s.replace(fault=s.fault | fault), action, path, speed, horizon, pulse)
            data = {k: logs[k] for k in keep}
            data.update(time=logs['time'], relative_time=logs['time'] - start * env.cc.dt,
                        fallback_active=j.zeros(4, dtype=j.bool_), upper_active=logs['active_tick'],
                        original_policy_latent=j.broadcast_to(original, (4, 2)),
                        executed_policy_latent=j.broadcast_to(action, (4, 2)),
                        prefix_override=j.full(4, active),
                        upper_observation=j.broadcast_to(obs, (4, obs.shape[0])),
                        effective_costs=j.stack([logs['effective_components'][k] for k in keys], axis=-1))
            data['nonfinite'] = logs['nonfinite'] | logs['policy_fault']
            return end, data

        template = jax.eval_shape(normal, snapshots[0], paths[cases[0]['route']], speeds[0],
                                  pulses[0], starts[0], fixed[0], masks[0], enabled[0])[1]
        zero_block = jax.tree.map(lambda x: j.zeros(x.shape, x.dtype), template)
        zero_tick = jax.tree.map(lambda x: j.zeros(x.shape[1:], x.dtype), template)

        def fallback(s, pulse, start):
            def step(st, _):
                def active(st):
                    ref = st.physical.governor.current_reference
                    physical, lower, log = lower_tick(st.physical, st.lower, pulse,
                                                      st.tick * env.cc.dt, ref)
                    end = st.replace(physical=physical, lower=lower, tick=st.tick + 1,
                                     fault=st.fault | log['lower_fault'] | log['nonfinite'],
                                     yaw_rate=(physical.yaw_unwrapped - st.physical.yaw_unwrapped) / env.cc.dt)
                    data = dict(zero_tick)
                    for key in ['active_tick', 'actual_xy', 'actual_forward_speed', 'wheel_speed_proxy',
                                'phi', 'phi_dot', 'actual_delta', 'peak_roll', 'physical_failure',
                                'nonfinite', 'contact_failure', 'lower_fault', 'applied_force_impulse']:
                        data[key] = log[key]
                    for key in ['reference_xy', 'path_cross_track', 'path_heading_error', 'path_progress',
                                'goal_section_signed_distance', 'latent_z', 'target_offset', 'chi',
                                'scored_tick_reward', 'effective_costs', 'upper_observation',
                                'original_policy_latent', 'executed_policy_latent']:
                        data[key] = j.full_like(zero_tick[key], j.nan)
                    data.update(time=st.tick * env.cc.dt, relative_time=(st.tick - start) * env.cc.dt,
                                domain_exit=j.bool_(True), fallback_active=j.bool_(True),
                                governed=ref, nominal=st.physical.raw, offsets=st.offsets,
                                filtered_offset=st.filtered)
                    return end, data
                stop = st.physical.failed | st.fault | (st.tick >= deadline_tick)
                return jax.lax.cond(stop, lambda st: (st, zero_tick), active, st)
            return jax.lax.scan(step, s, None, length=4)

        def one(s, path, speed, pulse, start, target, mask, use):
            def active(st):
                return jax.lax.cond(st.domain_exit,
                                    lambda st: fallback(st, pulse, start),
                                    lambda st: normal(st, path, speed, pulse, start, target, mask, use), st)
            stop = s.physical.failed | s.fault | (s.tick >= deadline_tick)
            return jax.lax.cond(stop, lambda st: (st, zero_block), active, s)

        functions = {}
        def execute(states, length):
            if length not in functions:
                functions[length] = jax.jit(lambda st: jax.lax.scan(
                    lambda s, _: jax.vmap(one)(s, path_batch, speeds, pulses, starts, fixed, masks, enabled),
                    st, None, length=length))
            return functions[length](states)

        atomic_json(root / 'manifest.json', dict(plan=plan, config=cfg,
                    physical_identity=env.physics.bundle.identity, reward_components=list(keys),
                    original_actor=declaration['policies']['H75'],
                    requested_targets=np.asarray(targets).tolist(), realized_targets=np.asarray(realized).tolist(),
                    target_clipped_channels=np.asarray(clipped).tolist(),
                    executed_target_clipped_channels=np.asarray(clipped & (masks[:,None,:] if piecewise else masks)).tolist(),
                    maximum_remaining_lower_ticks=maximum_remaining_ticks,
                    path_evidence_checks=path_checks,
                    fallback_contract='hold last reference after domain exit; task ineligible; reward undefined, stored NaN',
                    deadline_tick=deadline_tick, prefix_ticks=prefix_ticks, prefix_delay_ticks=delay_ticks,
                    handback_ticks=handback_ticks, initialization='full_snapshot_no_reset',
                    information_role='known onset offline opportunity only; not a deployed force detector or selector'))
        saved = []
        lengths = [handback_ticks // 4]
        remaining = int(np.ceil((deadline_tick - int(np.min(starts)) - handback_ticks) / 4)) + 1
        lengths += [25] * int(np.ceil(remaining / 25))
        for block, length in enumerate(lengths):
            states, logs = execute(states, length)
            jax.block_until_ready(states)
            host = jax.device_get(logs)
            flat = {k: np.swapaxes(v, 1, 2).reshape((length * 4, len(cases)) + v.shape[3:]) for k, v in host.items()}
            saved.append(flat)
            status['active_lower_ticks'] += int(flat['active_tick'].sum())
            if status['active_lower_ticks'] > plan['maximum_new_lower_ticks']:
                raise RuntimeError('declared physics exceeded')
            if block == 0:
                handback = jax.device_get(states)
                with (root / 'handback_states.pkl').open('wb') as stream:
                    pickle.dump(handback, stream)
                atomic_json(root / 'handback_receipt.json', dict(
                    elapsed_ticks=(np.asarray(states.tick) - np.asarray(starts)).tolist(),
                    intended_ticks=handback_ticks, original_alpha=np.asarray(states.alpha).tolist(),
                    physical_tick=np.asarray(states.physical.physical_tick).tolist(),
                    task_tick=np.asarray(states.tick).tolist(), reached_without_stop=np.asarray(
                        ~(states.physical.failed | states.fault | states.domain_exit)).tolist(),
                    carry_action='same returned complete states passed directly to the next scan; no replacement at handback'))
            update(stage='scanning', blocks_completed=block + 1, total_blocks=len(lengths))
        all_data = {k: np.concatenate([block[k] for block in saved], axis=0) for k in saved[0]}
        results = []
        for i, row in enumerate(cases):
            full = {k: v[:, i] for k, v in all_data.items()}
            trace = {k: v[full['active_tick']] for k, v in full.items()}
            name = f'case_{i:03d}'
            np.savez_compressed(root / (name + '.npz'), **trace)
            relative = dict(full, time=full['relative_time'])
            duration = (deadline_tick - int(starts[i])) * env.cc.dt
            physical = classify_contact_balance(relative, horizon_s=duration)
            labels = outcome_labels(trace, speed=row['speed'], goal_s=float(paths[row['route']].goal),
                                    working_roll=cfg['stage_B_baseline']['working_roll_rad'],
                                    horizon=horizon, hold_s=cfg['stage_B_baseline']['hold_s'])
            # Undefined path metrics after handover stay unknown, not numerical fault.
            for key in ('peak_abs_path_m', 'tail_max_path_m'):
                if labels[key] is not None and not np.isfinite(labels[key]):
                    labels[key] = None
            impulse = trace['applied_force_impulse'].sum(axis=0, dtype=np.float64 if piecewise else None)
            expected = forces[i] * directions[i] * plan['pulse_duration_s']
            fault = bool(np.asarray(states.fault)[i])
            if fault:
                physical['balance_label'] = 'ENGINEERING_ERROR'
            results.append({**row, **labels, **physical, 'name': name,
                            'final_fault': fault, 'fallback_triggered': bool(trace['fallback_active'].any()),
                            'joint_success': bool(labels['task_recovered'] and physical['balance_label'] == 'NO_FLOOR_CONTACT'),
                            'logged_suffix_return_while_upper_active': float(np.nansum(trace['scored_tick_reward'], dtype=np.float64 if piecewise else None)),
                            'impulse_Ns': impulse.tolist(), 'expected_impulse_Ns': expected.tolist(),
                            'impulse_error_Ns': float(np.linalg.norm(impulse - expected))})
        atomic_json(root / 'results.json', dict(cases=results))
        errors = sum(r['balance_label'] == 'ENGINEERING_ERROR' for r in results)
        update(phase='error' if errors else 'completed', stage='labels_ready',
               completed_cases=len(results), engineering_error_cases=errors)
        if errors:
            raise RuntimeError('engineering faults retained separately from fall labels')
    except BaseException as exc:
        update(phase='error', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
