"""Capture full middle-controller history at geometric path phases, no state edits."""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
from sttw_control.preference_best import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    root = Path(a.run).resolve()
    plan = json.loads((root/'prepare_plan.json').read_text())
    state = dict(phase='running', pid=os.getpid(), stage='loading', completed_cases=0,
                 declared_cases=plan['declared_prefixes'], active_lower_ticks=0, training_steps=0)
    def update(**kw):
        state.update(kw, checked_unix=time.time())
        atomic_json(root/'prepare_status.json', state)
    update()
    try:
        for file, sha in plan['source_sha256'].items():
            if digest(file) != sha:
                raise ValueError('Declared source changed '+file)
        from sttw_control.path_command_training import load_environment
        from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
        from sttw_control.geometric_path import build_path
        from sttw_control.path_command_reward import configured_cost_keys
        from sttw_control.runtime import configure_compilation_cache
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        declaration = json.loads(Path(plan['source_declaration']).read_text())
        cfg = declaration['config']
        env, initial = load_environment(cfg, declaration['source'], 'terminal_contract_v2')
        pose = np.asarray(env.physics.helpers.pose(initial.data))
        path = build_path(plan['builder'], pose, cfg)
        with np.load(Path(plan['path_evidence_directory'])/(plan['route']+'_path.npz')) as old:
            np.testing.assert_allclose(np.asarray(path.xy), old['xy'], atol=1e-6, rtol=0)
        with Path(plan['model']['actor']).open('rb') as stream:
            parameters = jax.tree.map(j.asarray, pickle.load(stream))
        validate_actor_parameters(parameters, cfg)
        actor = make_path_actor(cfg)
        contexts = [dict(speed=speed, alpha=alpha) for speed in plan['speeds'] for alpha in plan['alphas']]
        speeds = j.asarray([c['speed'] for c in contexts])
        alphas = j.asarray([c['alpha'] for c in contexts])
        states = jax.jit(jax.vmap(lambda v, alpha: env.reset(initial, path, v, alpha)))(speeds, alphas)
        sample = jax.tree.map(lambda x: x[0], states)
        env.set_log_template(sample, path, speeds[0])
        keys = ['active_tick', 'time', 'actual_xy', 'actual_forward_speed', 'phi', 'phi_dot',
                'actual_delta', 'governed', 'nominal', 'path_progress', 'path_cross_track',
                'path_heading_error', 'scored_tick_reward']
        cost_keys = configured_cost_keys(cfg)
        def advance(s, speed):
            obs, _, fault = env.observation(s, path, speed, 10.)
            action = actor.apply(parameters, obs)
            end, log = env.policy_step(s.replace(fault=s.fault|fault), action, path, speed, 10., j.zeros(5))
            record = {k: log[k] for k in keys}
            record['effective_costs'] = j.stack([log['effective_components'][k] for k in cost_keys], axis=-1)
            return end, record
        shape = jax.eval_shape(advance, sample, speeds[0])[1]
        zeros = jax.tree.map(lambda x: j.zeros(x.shape, x.dtype), shape)
        def one(s, speed, done):
            return jax.lax.cond(done, lambda _: (s, zeros), lambda _: advance(s, speed), None)
        execute = jax.jit(jax.vmap(one))
        completed = np.zeros(len(contexts), bool)
        counts = np.zeros(len(contexts), int)
        targets = [float(cfg['path']['straight_prefix_m'])+f*(float(path.turn_end)-float(cfg['path']['straight_prefix_m'])) for f in plan['phase_fractions']]
        blocks, rows, snapshot_hashes = [], [], {}
        update(stage='genuine_upper_prefixes')
        for step in range(500):
            previous_progress = np.asarray(states.projection.progress)
            states, logs = execute(states, speeds, j.asarray(completed))
            jax.block_until_ready(states)
            host = jax.device_get(logs)
            blocks.append(host)
            state['active_lower_ticks'] += int(host['active_tick'].sum())
            if state['active_lower_ticks'] > plan['maximum_prepare_lower_ticks']:
                raise ValueError('Prefix budget exceeded')
            progress = np.asarray(states.projection.progress)
            invalid = np.asarray(states.fault|states.domain_exit|states.physical.failed)
            if (invalid & ~completed).any():
                raise ValueError('Nominal prefix invalid before both geometric phases')
            for i, context in enumerate(contexts):
                while counts[i] < 2 and progress[i] >= targets[counts[i]]:
                    fraction = plan['phase_fractions'][counts[i]]
                    sid = f"{plan['route']}_v{context['speed']:g}_a{context['alpha']:g}_phase{fraction:g}"
                    captured = jax.device_get(jax.tree.map(lambda x: x[i], states))
                    file = root/(sid+'_snapshot.pkl')
                    with file.open('wb') as stream:
                        pickle.dump(captured, stream)
                    snapshot_hashes[str(file)] = digest(file)
                    if not previous_progress[i] < targets[counts[i]] <= progress[i]:
                        raise ValueError('Capture did not use first geometric crossing')
                    if float(captured.alpha) != context['alpha'] or not np.asarray(captured.history.mask).all():
                        raise ValueError('Actual alpha/history differs from declared prefix')
                    m, _, body_speed = env.physics.observe(jax.tree.map(j.asarray, captured.physical.data))
                    rows.append(dict(state_id=sid, **context, route=plan['route'], fraction=fraction,
                                     target_progress_m=targets[counts[i]], actual_progress_m=float(progress[i]),
                                     previous_control_progress_m=float(previous_progress[i]),
                                     task_time_s=float(captured.tick)*.005,
                                     actual_speed_m_s=float(body_speed), roll_rad=float(m[0]),
                                     roll_rate_rad_s=float(m[1]), steer_rad=float(m[2]),
                                     governed_reference=np.asarray(captured.physical.governor.current_reference).tolist(),
                                     snapshot=str(file), snapshot_sha256=snapshot_hashes[str(file)],
                                     complete_upper_history=True))
                    counts[i] += 1
                completed[i] = counts[i] == 2
            if step % 25 == 0 or completed.all():
                update(completed_cases=len(rows), upper_steps_executed=step+1)
            if completed.all():
                break
        if len(rows) != plan['declared_prefixes']:
            raise ValueError('Geometric phases not reached within nominal10s')
        for i, context in enumerate(contexts):
            arrays = {key: np.concatenate([b[key][i] for b in blocks], axis=0) for key in blocks[0]}
            active = arrays['active_tick'].astype(bool)
            arrays = {key: value[active] for key, value in arrays.items()}
            np.savez_compressed(root/f"prefix_v{context['speed']:g}_a{context['alpha']:g}.npz", **arrays)
        atomic_json(root/'prefix_results.json', dict(cases=rows, complete_upper_history=True,
                    physical_identity=env.physics.bundle.identity, reward_components=list(cost_keys),
                    targets_m=targets, phase_clock='first20ms control boundary reaching target'))
        cases = []
        for row in rows:
            for force in plan['forces_N']:
                cases.append(dict(state_id=row['state_id'], policy=plan.get('policy_name', 'H75'), route=plan['route'],
                                  requested_speed=row['speed'], alpha=row['alpha'], fraction=row['fraction'],
                                  snapshot=row['snapshot'], gate_mode=0, force_N=force))
        scan = dict(project='STTW_CONTROL', purpose='Alpha endpoints at matched geometric phases: physical force grid',
                    source_declaration=plan['source_declaration'],
                    path_evidence_directory=plan['path_evidence_directory'],
                    source_sha256={**plan['source_sha256'], **snapshot_hashes},
                    actors={plan.get('policy_name', 'H75'): dict(path=plan['model']['actor'], sha256=plan['model']['actor_sha256'])},
                    cases=cases, sensor_contract=dict(rear_wheel_radius_proxy_m=.1),
                    pulse_duration_s=.1, observation_s=4., maximum_lower_ticks=plan['maximum_scan_lower_ticks'],
                    training_steps=0, force_direction='capture heading lateral world vector',
                    termination='contact_only + common domain hold; not full task success',
                    stop_rule=plan['stop_rule'])
        atomic_json(root/'scan_plan.json', scan)
        update(phase='completed', stage='declared_complete_upper_prefixes_saved', completed_cases=plan['declared_prefixes'])
    except BaseException as exc:
        update(phase='error', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
