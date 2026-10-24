"""Frozen DEV baseline, with the detector running from complete task origin."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

from sttw_control.preference_best import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    import numpy as np
    from prepare_short_recovery_campaign import register
    worktree = Path(__file__).resolve().parents[2]
    root.mkdir(exist_ok=False)
    sources = list((worktree / 'learning/src/sttw_control').rglob('*.py')) + [Path(__file__).resolve()]
    hashes = {str(p): digest(p) for p in sources}
    for source in sources:
        destination = root / 'source_at_launch' / source.relative_to(worktree)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        assert digest(destination) == hashes[str(source)]
    plans = []
    old_dirs = {3: worktree / 'runs/reference_prefix_return_feasibility_v1/evaluation',
                0: worktree / 'runs/reference_prefix_return_confirmation_v1_attempt2/bank0/evaluation'}
    for bank, old in old_dirs.items():
        old_plan = json.loads((old / 'plan.json').read_text())
        declaration = Path(old_plan['source_declaration'])
        source_declaration = json.loads(declaration.read_text())
        assert source_declaration['source']['bank_index'] == bank
        old_rows = json.loads((old / 'results.json').read_text())['cases']
        cases = []
        for route, speed in (('left90_R1p1', 1.8), ('right90_R1p1', 1.5)):
            matches = [r for r in old_rows if r['candidate'] == 'continue_original'
                       and r['route'] == route and r['speed'] == speed and r['alpha'] == 0 and r['force_N'] == 32]
            assert len(matches) == 1
            old_row = matches[0]
            raw = old / (old_row['name'] + '.npz')
            with np.load(raw) as data:
                onset_tick = round(float(data['time'][0]) / .005)
                assert abs(onset_tick * .005 - float(data['time'][0])) < 2e-6
            direction = np.asarray(old_row['expected_impulse_Ns']) / (32. * .1)
            np.testing.assert_allclose(np.linalg.norm(direction), 1., atol=1e-6)
            for force in (0., -32., 32.):
                for arm in ('original', 'causal'):
                    cases.append(dict(route=route, speed=speed, alpha=0., force_N=force, arm=arm,
                                      pulse_onset_task_tick=onset_tick, pulse_duration_s=.1,
                                      world_direction=direction.tolist(), capture_snapshot=old_row['snapshot'],
                                      capture_snapshot_sha256=old_row['snapshot_sha256'],
                                      schedule_source_raw=str(raw), schedule_source_raw_sha256=digest(raw)))
        directory = root / f'bank{bank}'
        directory.mkdir()
        plan = dict(project='STTW_CONTROL', cases=cases, declared_cases=12, bank_index=bank,
                    source_declaration=str(declaration), source_declaration_sha256=digest(declaration),
                    path_evidence_directory=old_plan['path_evidence_directory'], path_evidence_atol_m=1e-4,
                    source_sha256=hashes, original_task_deadline_s=10., maximum_new_lower_ticks=24000,
                    scope='DEV full-task causal baseline; no independent test or tolerance estimator')
        atomic_json(directory / 'plan.json', plan)
        plans.append(str(directory / 'plan.json'))
        register(directory / 'ACTIVE_RUN.json', f'STTW_CONTROL 完整任务因果ESO基线 bank{bank}', directory / 'status.json')
    design = dict(project='STTW_CONTROL', purpose='Test net benefit of a strictly causal simple baseline, including zero and opposite directions.',
                  plans=plans, plan_sha256={p: digest(p) for p in plans}, source_sha256=hashes,
                  declared_cases=24, maximum_new_lower_ticks=48000, training_steps=0, model_fits=0,
                  detector_origin='once at complete task origin; never reset at capture/pulse',
                  target_anchor='current offset at detection; differs from known-onset diagnostic',
                  information='Only environment receives fixed pulse schedules/world forces; gate reads existing actor input and current carry.',
                  stop='No parameter/delay/force/bank sweep. If no net causal benefit, stop this gate proposal. '
                       'Positive results establish only a simple DEV baseline, not a RA-L method.')
    atomic_json(root / 'design.json', design)
    atomic_json(root / 'status.json', dict(phase='prepared', declared_cases=24, completed_cases=0,
                                         active_lower_ticks=0, training_steps=0, model_fits=0))
    register(root / 'ACTIVE_RUN.json', 'STTW_CONTROL 完整任务因果ESO基线与全部对照分析', root / 'status.json')
    print(root / 'design.json')


def execute_bank(root):
    import jax
    import jax.numpy as j
    import numpy as np
    from sttw_control.contact_only_environment import load_environment_contact_only
    from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
    from sttw_control.path_command_reward import configured_cost_keys
    from sttw_control.reference_confirmation import validate_path_evidence
    from sttw_control.geometric_path import build_path
    from sttw_control.balance_probe import make_lower_tick, classify_contact_balance
    from sttw_control.recovery_margin_scan import outcome_labels
    from sttw_control.runtime import configure_compilation_cache
    from sttw_control.causal_reference_baseline import (initial_gate, causal_reference_action,
                                                       validate_causal_contract, validate_full_task_origin,
                                                       case_has_engineering_error)
    plan = json.loads((root / 'plan.json').read_text())
    state = dict(phase='running', pid=os.getpid(), stage='loading', declared_cases=12,
                 completed_cases=0, active_lower_ticks=0, training_steps=0, model_fits=0)
    def update(**changes):
        state.update(changes, checked_unix=time.time()); atomic_json(root / 'status.json', state)
    update()
    try:
        for path, sha in plan['source_sha256'].items():
            if digest(path) != sha: raise ValueError('source changed: ' + path)
        if digest(plan['source_declaration']) != plan['source_declaration_sha256']:
            raise ValueError('source declaration changed')
        if (root / 'manifest.json').exists(): raise FileExistsError('execution already started')
        declaration = json.loads(Path(plan['source_declaration']).read_text())
        cfg = declaration['config']
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env, initial = load_environment_contact_only(cfg, declaration['source'], 'terminal_contract_v2')
        validate_causal_contract(cfg, env.plant_spec['network']['frame_fields'])
        original_actor = declaration['policies']['H75']
        for key in ('actor', 'checkpoint'):
            if digest(original_actor[key]) != original_actor[key + '_sha256']: raise ValueError('H75 identity mismatch')
        with Path(original_actor['actor']).open('rb') as stream:
            parameters = jax.tree.map(j.asarray, pickle.load(stream))
        validate_actor_parameters(parameters, cfg)
        actor = make_path_actor(cfg)
        pose = np.asarray(env.physics.helpers.pose(initial.data))
        paths = {r['name']: build_path(r['builder'], pose, cfg) for r in declaration['routes']}
        path_checks = []
        for name, path in paths.items():
            with np.load(Path(plan['path_evidence_directory']) / (name + '_path.npz')) as data:
                diff = validate_path_evidence(path.xy, data['xy'], atol_m=plan['path_evidence_atol_m'])
            path_checks.append(dict(route=name, max_abs_difference_m=diff))
            np.savez_compressed(root / (name + '_path.npz'), xy=np.asarray(path.xy), s=np.asarray(path.s), goal=np.asarray(path.goal))
        cases = plan['cases']
        stack = lambda values: jax.tree.map(lambda *xs: j.stack(xs), *values)
        # Complete task reset exactly once; no mid-task capture is used as state.
        path_states = [env.reset(initial, paths[r['route']], r['speed'], r['alpha']) for r in cases]
        gate_states = [initial_gate() for _ in cases]
        for s, gate in zip(path_states, gate_states):
            validate_full_task_origin(s.tick, s.physical.physical_tick, initial.physical_tick, gate)
        captures = {}
        for row in cases:
            if digest(row['schedule_source_raw']) != row['schedule_source_raw_sha256'] or digest(row['capture_snapshot']) != row['capture_snapshot_sha256']:
                raise ValueError('environment-only pulse schedule source changed')
            if row['capture_snapshot'] not in captures:
                with Path(row['capture_snapshot']).open('rb') as stream: captures[row['capture_snapshot']] = pickle.load(stream)
            capture = captures[row['capture_snapshot']]
            if int(capture.tick) != row['pulse_onset_task_tick'] or int(capture.physical.physical_tick) != int(initial.physical_tick) + int(capture.tick):
                raise ValueError('pulse clock differs from authentic capture')
            heading = float(env.physics.helpers.pose(capture.physical.data)[2])
            np.testing.assert_allclose(row['world_direction'], [-np.sin(heading), np.cos(heading), 0.], atol=1e-6)
        carry = (stack(path_states), stack(gate_states))
        path_batch = stack([paths[r['route']] for r in cases])
        speeds = j.asarray([r['speed'] for r in cases])
        enabled = j.asarray([r['arm'] == 'causal' for r in cases])
        absolute_start = int(initial.physical_tick) * env.cc.dt
        pulses = j.asarray([[absolute_start + r['pulse_onset_task_tick'] * env.cc.dt,
                             absolute_start + r['pulse_onset_task_tick'] * env.cc.dt + .1,
                             *[r['force_N'] * v for v in r['world_direction']]] for r in cases])
        deadline = round(plan['original_task_deadline_s'] / env.cc.dt)
        if deadline != 2000 or len(cases)*deadline != plan['maximum_new_lower_ticks']:
            raise ValueError('full task timing or budget differs')
        env.set_log_template(path_states[0], paths[cases[0]['route']], speeds[0])
        lower_tick = make_lower_tick(env, initial.governor.current_reference, contact_only=True)
        cost_keys = configured_cost_keys(cfg)
        keep = ['active_tick', 'actual_xy', 'reference_xy', 'path_cross_track', 'path_heading_error',
                'path_progress', 'goal_section_signed_distance', 'actual_forward_speed', 'wheel_speed_proxy',
                'phi', 'phi_dot', 'actual_delta', 'peak_roll', 'physical_failure', 'nonfinite',
                'contact_failure', 'lower_fault', 'policy_fault', 'domain_exit', 'governed', 'nominal',
                'latent_z', 'offsets', 'filtered_offset', 'target_offset', 'applied_force_impulse',
                'scored_tick_reward', 'chi', 'reference_clip_channels', 'correction_rate_clipped']

        def normal(pair, path, speed, pulse, use):
            s, gate = pair
            obs, _, fault = env.observation(s, path, speed, 10.)
            original = actor.apply(parameters, obs)
            # No pulse, onset, route ID or true force is passed to this function.
            chosen, gate_after, receipt = causal_reference_action(original, obs, s.offsets, s.tick, gate, cfg)
            action = j.where(use, chosen, original)
            end, log = env.policy_step(s.replace(fault=s.fault | fault | receipt['engineering_fault']), action, path, speed, 10., pulse)
            data = {k: log[k] for k in keep}
            data.update(time=log['time'], relative_time=log['time'], upper_active=log['active_tick'],
                        fallback_active=j.zeros(4, dtype=j.bool_), upper_observation=j.broadcast_to(obs, (4, 351)),
                        original_policy_latent=j.broadcast_to(original, (4, 2)), executed_policy_latent=j.broadcast_to(action, (4, 2)),
                        override_active=j.full(4, use & receipt['override_active']),
                        gate_signal=j.full(4, receipt['signal']), gate_valid=j.full(4, receipt['valid']),
                        gate_detected=j.full(4, receipt['detected']), gate_fired=j.full(4, gate_after.fired),
                        gate_start_tick=j.full(4, gate_after.start_tick), gate_mask=j.broadcast_to(gate_after.mask, (4, 2)),
                        gate_target_offset=j.broadcast_to(gate_after.target_offset, (4, 2)),
                        gate_target_clipped=j.broadcast_to(receipt['target_clipped_channels'], (4, 2)),
                        gate_engineering_fault=j.full(4, receipt['engineering_fault']),
                        effective_costs=j.stack([log['effective_components'][k] for k in cost_keys], axis=-1))
            data['nonfinite'] |= log['policy_fault']
            return (end, gate_after), data
        template = jax.eval_shape(normal, (path_states[0], initial_gate()), paths[cases[0]['route']], speeds[0], pulses[0], enabled[0])[1]
        zero_block = jax.tree.map(lambda x: j.zeros(x.shape, x.dtype), template)
        zero_tick = jax.tree.map(lambda x: j.zeros(x.shape[1:], x.dtype), template)

        def fallback(pair, pulse):
            s, gate = pair
            def step(st, _):
                def active(st):
                    ref = st.physical.governor.current_reference
                    physical, lower, log = lower_tick(st.physical, st.lower, pulse, st.tick * env.cc.dt, ref)
                    end = st.replace(physical=physical, lower=lower, tick=st.tick+1,
                                     fault=st.fault | log['lower_fault'] | log['nonfinite'],
                                     yaw_rate=(physical.yaw_unwrapped-st.physical.yaw_unwrapped)/env.cc.dt)
                    data = dict(zero_tick)
                    for key in ('active_tick', 'actual_xy', 'actual_forward_speed', 'wheel_speed_proxy', 'phi', 'phi_dot',
                                'actual_delta', 'peak_roll', 'physical_failure', 'nonfinite', 'contact_failure', 'lower_fault', 'applied_force_impulse'):
                        data[key] = log[key]
                    for key in ('reference_xy', 'path_cross_track', 'path_heading_error', 'path_progress', 'goal_section_signed_distance',
                                'latent_z', 'target_offset', 'chi', 'scored_tick_reward', 'effective_costs', 'upper_observation',
                                'original_policy_latent', 'executed_policy_latent', 'gate_signal'):
                        data[key] = j.full_like(zero_tick[key], j.nan)
                    data.update(time=st.tick*env.cc.dt, relative_time=st.tick*env.cc.dt, domain_exit=j.bool_(True),
                                fallback_active=j.bool_(True), governed=ref, nominal=st.physical.raw,
                                offsets=st.offsets, filtered_offset=st.filtered, gate_fired=gate.fired,
                                gate_start_tick=gate.start_tick, gate_mask=gate.mask, gate_target_offset=gate.target_offset)
                    return end, data
                stop = st.physical.failed | st.fault | (st.tick >= deadline)
                return jax.lax.cond(stop, lambda st: (st, zero_tick), active, st)
            end, data = jax.lax.scan(step, s, None, length=4)
            return (end, gate), data
        def one(pair, path, speed, pulse, use):
            s, _ = pair
            def active(pair):
                return jax.lax.cond(pair[0].domain_exit, lambda p: fallback(p, pulse),
                                    lambda p: normal(p, path, speed, pulse, use), pair)
            stop = s.physical.failed | s.fault | (s.tick >= deadline)
            return jax.lax.cond(stop, lambda p: (p, zero_block), active, pair)
        # Paths/pulses are arguments, avoiding baking bank-specific coordinates into compilation.
        execute = jax.jit(lambda carry, paths, speeds, pulses, enabled: jax.lax.scan(
            lambda pair, _: jax.vmap(one)(pair, paths, speeds, pulses, enabled), carry, None, length=25))
        atomic_json(root / 'manifest.json', dict(plan=plan, config=cfg, original_actor=original_actor,
                    physical_identity=env.physics.bundle.identity, reward_components=list(cost_keys), path_checks=path_checks,
                    detector_initialization='once at task tick zero; carry never reset at pulse or capture',
                    initial_physical_tick=int(initial.physical_tick), detector_information='existing obs, current offset, original action, task tick only',
                    environment_only_pulses=np.asarray(pulses).tolist(), fallback='hold last reference; task ineligible and reward NaN'))
        saved = []
        for block in range(20):
            carry, logs = execute(carry, path_batch, speeds, pulses, enabled)
            jax.block_until_ready(carry)
            host = jax.device_get(logs)
            flat = {k: np.swapaxes(v, 1, 2).reshape((100, len(cases)) + v.shape[3:]) for k,v in host.items()}
            saved.append(flat)
            state['active_lower_ticks'] += int(flat['active_tick'].sum())
            if state['active_lower_ticks'] > plan['maximum_new_lower_ticks']: raise RuntimeError('physics budget exceeded')
            update(stage='full_task_scanning', blocks_completed=block+1, total_blocks=20)
        all_data = {k: np.concatenate([b[k] for b in saved], axis=0) for k in saved[0]}
        states, gates = jax.device_get(carry)
        results = []
        for i, row in enumerate(cases):
            full = {k: v[:, i] for k,v in all_data.items()}
            trace = {k:v[full['active_tick']] for k,v in full.items()}
            name = f'case_{i:03d}'
            np.savez_compressed(root / (name+'.npz'), **trace)
            physical = classify_contact_balance(full, horizon_s=10.)
            labels = outcome_labels(trace, speed=row['speed'], goal_s=float(paths[row['route']].goal),
                                    working_roll=cfg['stage_B_baseline']['working_roll_rad'], horizon=10., hold_s=cfg['stage_B_baseline']['hold_s'])
            for key in ('peak_abs_path_m', 'tail_max_path_m'):
                if labels[key] is not None and not np.isfinite(labels[key]): labels[key]=None
            impulse = trace['applied_force_impulse'].sum(axis=0, dtype=np.float64)
            expected = row['force_N'] * .1 * np.asarray(row['world_direction'])
            amount, intended = float(np.linalg.norm(impulse)), float(np.linalg.norm(expected))
            exposure = 'zero' if row['force_N']==0 else 'none' if amount==0 else 'full' if abs(amount-intended)<=.01*intended else 'partial'
            ix = np.arange(0, len(trace['time']), 4)
            hits = ix[trace['gate_detected'][ix] & trace['upper_active'][ix]]
            if len(hits)>1: raise ValueError('single-shot gate fired more than once')
            fault = bool(np.asarray(states.fault)[i])
            if fault: physical['balance_label']='ENGINEERING_ERROR'
            results.append({**row, **labels, **physical, 'name':name, 'final_fault':fault,
                            'joint_success':bool(labels['task_recovered'] and physical['balance_label']=='NO_FLOOR_CONTACT'),
                            'logged_suffix_return_while_upper_active':float(np.nansum(trace['scored_tick_reward'], dtype=np.float64)),
                            'impulse_Ns':impulse.tolist(), 'expected_impulse_Ns':expected.tolist(), 'force_exposure':exposure,
                            'detection_time_s':None if not len(hits) else float(trace['time'][hits[0]]),
                            'detection_signal':None if not len(hits) else float(trace['gate_signal'][hits[0]]),
                            'override_lower_ticks':int(trace['override_active'].sum()), 'baseline_gate_role':'shadow telemetry; only causal arm executes override'})
        atomic_json(root / 'results.json', dict(cases=results))
        errors = sum(case_has_engineering_error(r['final_fault'], r['balance_label']) for r in results)
        update(phase='error' if errors else 'completed', stage='labels_ready', completed_cases=len(results), engineering_error_cases=errors)
        if errors: raise RuntimeError('engineering faults retained separately from physical outcomes')
    except BaseException as exc:
        update(phase='error', error_type=type(exc).__name__, error=str(exc)); raise


def report(root, design):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'analysis';out.mkdir(exist_ok=False)
    pairs, hashes, ticks = [], {}, 0
    for plan_path in design['plans']:
        directory=Path(plan_path).parent
        rows=json.loads((directory/'results.json').read_text())['cases']
        manifest=json.loads((directory/'manifest.json').read_text())
        fig, axes=plt.subplots(2,3,figsize=(18,11),layout='constrained')
        for p in range(6):
            baseline, causal=rows[2*p:2*p+2]
            assert baseline['arm']=='original' and causal['arm']=='causal'
            assert (baseline['route'],baseline['speed'],baseline['force_N'])==(causal['route'],causal['speed'],causal['force_N'])
            pair=dict(bank=directory.name,route=baseline['route'],speed=baseline['speed'],alpha=0.,force_N=baseline['force_N'],
                      baseline_safe=baseline['balance_label']=='NO_FLOOR_CONTACT',causal_safe=causal['balance_label']=='NO_FLOOR_CONTACT',
                      baseline_joint=baseline['joint_success'],causal_joint=causal['joint_success'],
                      detection_time_s=causal['detection_time_s'],override_lower_ticks=causal['override_lower_ticks'],
                      baseline_exposure=baseline['force_exposure'],causal_exposure=causal['force_exposure'])
            ax=axes.flat[p];progress=[]
            detail, aa=plt.subplots(8,1,figsize=(11,17),layout='constrained')
            for row,color in ((baseline,'black'),(causal,'tab:orange')):
                path=directory/(row['name']+'.npz');hashes[str(path)]=digest(path)
                with np.load(path) as z:data={k:z[k] for k in z.files}
                ticks+=len(data['time']);assert len(data['time'])==row['active_ticks']
                assert abs(float(data['time'][0]))<1e-6
                active=data['upper_active'].astype(bool);override=data['override_active'].astype(bool)
                log_ticks=np.rint(data['time']/.005).astype(np.int64)
                expected_override=(active & (row['arm']=='causal') & data['gate_fired'].astype(bool)
                                   & data['gate_mask'].astype(bool).any(axis=1)
                                   & (log_ticks>=data['gate_start_tick']) & (log_ticks<data['gate_start_tick']+80))
                np.testing.assert_array_equal(override,expected_override)
                assert not override[~active].any()
                if row['arm']=='original':assert not override.any()
                untouched=active[:,None] & ~(override[:,None] & data['gate_mask'].astype(bool))
                np.testing.assert_array_equal(data['executed_policy_latent'][untouched],data['original_policy_latent'][untouched])
                assert bool(data['contact_failure'].any())==row['event_contact']
                if override.any():
                    t0=int(data['gate_start_tick'][override][0]);t=data['time'][override]
                    assert t.min()>=t0*.005-2e-6 and t.max()<(t0+80)*.005+2e-6
                label=f"{row['arm']}; {row['balance_label']}; task={row['task_recovered']}"
                ax.plot(*data['actual_xy'].T,color=color,label=label)
                ax.scatter(*data['actual_xy'][-1],color=color,marker='x' if row['event_contact'] else 's' if row['domain_failure'] else 'o')
                prog=data['path_progress'][np.isfinite(data['path_progress'])]
                if len(prog):progress.append(float(prog.max()))
                t=data['time']
                for a,key in zip(aa[:5],('actual_forward_speed','path_cross_track','path_heading_error','phi','gate_signal')):a.plot(t,data[key],color=color,label=label)
                aa[5].plot(t,data['governed'][:,0],color=color);aa[5].plot(t,data['governed'][:,1],'--',color=color)
                ix=np.arange(0,len(t),4)
                reward=np.add.reduceat(np.nan_to_num(data['scored_tick_reward']).astype(np.float64),ix)
                valid=np.logical_or.reduceat(np.isfinite(data['scored_tick_reward']),ix)
                costs=-.005*manifest['config']['path_reward']['scale']*np.add.reduceat(np.nan_to_num(data['effective_costs']).astype(np.float64),ix,axis=0)
                reward[~valid]=np.nan;costs[~valid]=np.nan
                np.testing.assert_allclose(np.nansum(reward),row['logged_suffix_return_while_upper_active'],atol=1e-8)
                remainder=reward-costs.sum(axis=1)
                aa[6].plot(t[ix],reward,color=color);aa[7].plot(t[ix],np.where(valid,np.nancumsum(reward),np.nan),color=color)
                np.savez_compressed(out/f'{directory.name}_{row["name"]}_rewards.npz',task_time=t[ix],step_reward=reward,signed_components=costs,failure_remainder=remainder,component_names=manifest['reward_components'])
                cf, ca=plt.subplots(figsize=(12,5),layout='constrained')
                for k,name in enumerate(manifest['reward_components']):ca.plot(t[ix],costs[:,k],label=name)
                ca.plot(t[ix],remainder,'k--',label='failure remainder');ca.set_yscale('symlog',linthresh=.02)
                ca.set(title=label,xlabel='task seconds',ylabel='signed reward /20ms');ca.legend(fontsize=7,ncol=3)
                cf.savefig(out/f'{directory.name}_{row["name"]}_components.png');plt.close(cf)
            with np.load(directory/(baseline['route']+'_path.npz')) as path:
                show=path['s']<=max(progress)+1;ax.plot(*path['xy'][show].T,'--',color='gray',label='original path')
            title=f"{directory.name} {baseline['route']} v{baseline['speed']} F={baseline['force_N']:+g}N"
            ax.set(title=title,xlabel='x /m',ylabel='y /m');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6)
            onset=baseline['pulse_onset_task_tick']*.005
            for a,label in zip(aa,('speed /m/s','cross-track /m','heading /rad','roll /rad','normalized ESO innovation','reference speed/steer','reward /20ms','cumulative reward')):
                a.set_ylabel(label);a.axvspan(onset,onset+.1,color='gray',alpha=.1);a.grid(alpha=.2)
                if causal['detection_time_s'] is not None:a.axvline(causal['detection_time_s'],ls=':',color='tab:orange')
            aa[4].axhline(.015,ls='--',color='gray');aa[4].axhline(-.015,ls='--',color='gray')
            aa[0].legend(fontsize=7);aa[-1].set_xlabel('task seconds')
            for a in aa[6:]:a.set_yscale('symlog',linthresh=.1)
            detail.suptitle(title);detail.savefig(out/f'{directory.name}_pair{p}_timeseries.png');plt.close(detail)
            pairs.append(pair)
        fig.savefig(out/f'{directory.name}_all_xy.png');plt.close(fig)
    assert len(pairs)==12 and ticks<=48000
    atomic_json(out/'analysis.json',dict(pairs=pairs,raw_sha256=hashes,active_lower_ticks=ticks,training_steps=0,model_fits=0))
    lines=['# 完整任务因果ESO简单基线', '', '[bank3全部XY](bank3_all_xy.png) · [bank0全部XY](bank0_all_xy.png)', '',
           '目的：裁决不读取起扰/真实方向、从任务起点连续检测的完整反馈规则是否保留净收益。24条真实全任务，全部zero/反向/提前终止均保留。此为固定DEV简单基线，不是容扰度估计器或新颖方法。', '',
           '|bank|路径/速度|力N|原不触地|因果不触地|原联合|因果联合|检测秒|干预tick|暴露原/因果|', '|---|---|---:|---|---|---|---|---|---:|---|']
    for p in pairs:lines.append(f"|{p['bank']}|{p['route']}/{p['speed']}|{p['force_N']}|{p['baseline_safe']}|{p['causal_safe']}|{p['baseline_joint']}|{p['causal_joint']}|{p['detection_time_s']}|{p['override_lower_ticks']}|{p['baseline_exposure']}/{p['causal_exposure']}|")
    lines+=['',f'新增{ticks} lower ticks，0训练/拟合。逐20ms奖励、有符号分项、失败余项、所有XY/检测/参考曲线与原始SHA已保存。',
            '两个初态同祖先，阈值/规则在已看DEV上开发；没有独立泛化、噪声鲁棒或安全认证结论。若无净收益停止本门控，不扫阈值/延迟。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return ticks


def execute(root):
    design=json.loads((root/'design.json').read_text())
    state=dict(phase='running',pid=os.getpid(),declared_cases=24,completed_cases=0,active_lower_ticks=0,training_steps=0,model_fits=0)
    def update(**changes):
        state.update(changes,checked_unix=time.time());atomic_json(root/'status.json',state)
    update(stage='preflight')
    try:
        for path,sha in {**design['source_sha256'],**design['plan_sha256']}.items():
            if digest(path)!=sha:raise ValueError('source or plan changed: '+path)
        for plan in design['plans']:
            directory=Path(plan).parent
            with (directory/'execution.log').open('w') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode','bank','--output',str(directory)],stdout=log,stderr=subprocess.STDOUT)
                update(stage=directory.name,child_pid=child.pid)
                code=child.wait()
            child_state=json.loads((directory/'status.json').read_text())
            state['active_lower_ticks']+=child_state['active_lower_ticks']
            if code or child_state['phase']!='completed':raise RuntimeError(f'{directory.name} exit {code}')
            update(completed_cases=state['completed_cases']+child_state['completed_cases'])
        update(stage='analysis')
        ticks=report(root,design);assert ticks==state['active_lower_ticks']
        update(phase='completed',stage='reported',report=str(root/'analysis/REPORT.md'))
    except BaseException as exc:
        update(phase='error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','bank'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'run':execute,'bank':execute_bank}[args.mode](args.output.resolve())
