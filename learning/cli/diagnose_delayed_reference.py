"""Posthoc diagnostic of existing gains after a fixed 40ms action delay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sttw_control.preference_best import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    worktree = Path(__file__).resolve().parents[2]
    root.mkdir(exist_ok=False)
    sources = list((worktree / 'learning/src/sttw_control').rglob('*.py'))
    sources += [Path(__file__).resolve(), worktree / 'learning/cli/probe_reference_prefix_return.py']
    hashes = {str(p): sha(p) for p in sources}
    archive = root / 'source_at_launch'
    for path in sources:
        dest = archive / path.relative_to(worktree)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        assert sha(dest) == hashes[str(path)]
    plans = []
    old_runs = {
        'bank3': worktree / 'runs/reference_prefix_return_feasibility_v1/evaluation',
        'bank0': worktree / 'runs/reference_prefix_return_confirmation_v1_attempt2/bank0/evaluation',
    }
    for bank, old in old_runs.items():
        old_plan = json.loads((old / 'plan.json').read_text())
        rows = json.loads((old / 'results.json').read_text())['cases']
        selected = []
        for route, speed, force, candidate in [('left90_R1p1', 1.8, 32., 'steer_minus'),
                                                ('right90_R1p1', 1.5, -32., 'speed_plus')]:
            for name in ('continue_original', candidate):
                match = [r for r in rows if r['route'] == route and r['speed'] == speed
                         and r['alpha'] == 0 and r['force_N'] == force and r['candidate'] == name]
                assert len(match) == 1
                row = match[0]
                assert sha(row['snapshot']) == row['snapshot_sha256']
                # Keep only inputs, not inherited result labels or case names.
                selected.append({k: row[k] for k in ('state_id', 'route', 'speed', 'alpha', 'snapshot',
                                'snapshot_sha256', 'candidate', 'target_delta', 'override_mask', 'force_N')})
        directory = root / bank
        directory.mkdir()
        plan = dict(project='STTW_CONTROL', cases=selected, declared_cases=4,
                    source_declaration=old_plan['source_declaration'],
                    path_evidence_directory=old_plan['path_evidence_directory'],
                    path_evidence_atol_m=1e-4, source_sha256=hashes,
                    prefix_start_delay_s=.04, prefix_duration_s=.4, pulse_duration_s=.1,
                    original_task_deadline_s=10., maximum_new_lower_ticks=8000,
                    role='posthoc delayed opportunity on known positive contexts; no online detector',
                    historical_comparison_directory=str(old),
                    historical_results_sha256=sha(old / 'results.json'))
        atomic_json(directory / 'plan.json', plan)
        plans.append(str(directory / 'plan.json'))
    design = dict(project='STTW_CONTROL', purpose='Determine whether the two existing kinds of gains survive 40ms causal action delay.',
                  selection='Posthoc bank3 and bank0 known-positive contexts; not confirmatory, not a general authority upper bound.',
                  declared_cases=8, maximum_new_lower_ticks=16000, training_updates=0, model_fits=0,
                  delay_s=.04, prefix_duration_s=.4, plans=plans, source_sha256=hashes,
                  stop='No new banks, delays, amplitudes or force levels. If both gains disappear, stop this direction-triggered proposal. '
                       'One surviving gain supports only that opportunity; does not justify other regimes. No threshold selection.',
                  exclusions='No zero-force or opposite-direction interventions; no online safety or nominal-preservation claim.')
    atomic_json(root / 'design.json', design)
    atomic_json(root / 'status.json', dict(phase='prepared', completed_cases=0, declared_cases=8,
                                         active_lower_ticks=0, training_steps=0, model_fits=0))
    print(root / 'design.json')


def report(root, design):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), layout='constrained')
    pairs, raw_hashes = [], {}
    ticks = 0
    for b, plan_path in enumerate(design['plans']):
        directory = Path(plan_path).parent
        rows = json.loads((directory / 'results.json').read_text())['cases']
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert len(rows) == 4 and manifest['prefix_delay_ticks'] == 8 and manifest['handback_ticks'] == 88
        ticks += sum(r['active_ticks'] for r in rows)
        for p in range(2):
            base, intervention = rows[2*p:2*p+2]
            assert base['snapshot_sha256'] == intervention['snapshot_sha256']
            assert base['force_N'] == intervention['force_N']
            pair = dict(bank=directory.name, state_id=base['state_id'], force_N=base['force_N'],
                        baseline_safe=base['balance_label'] == 'NO_FLOOR_CONTACT',
                        delayed_safe=intervention['balance_label'] == 'NO_FLOOR_CONTACT',
                        baseline_joint=base['joint_success'], delayed_joint=intervention['joint_success'])
            detail, aa = plt.subplots(7, 1, figsize=(11, 15), layout='constrained')
            max_progress = 0
            for row, color in ((base, 'black'), (intervention, 'tab:orange')):
                path = directory / (row['name'] + '.npz')
                raw_hashes[str(path)] = sha(path)
                with np.load(path) as z:
                    data = {k: z[k] for k in z.files}
                assert len(data['time']) == row['active_ticks']
                before = data['relative_time'] < .04 - 1e-5
                after = data['relative_time'] >= .44 - 1e-5
                valid = data['upper_active'].astype(bool)
                assert not data['prefix_override'][before | after].any()
                np.testing.assert_array_equal(data['executed_policy_latent'][(before | after) & valid],
                                              data['original_policy_latent'][(before | after) & valid])
                if row['candidate'] == 'continue_original':
                    assert not data['prefix_override'].any()
                else:
                    assert data['prefix_override'][valid & ~(before | after)].all()
                assert bool(data['contact_failure'].any()) == row['event_contact']
                label = f"{row['candidate']}: safe={row['balance_label']}, task={row['task_recovered']}"
                ax = axes[b, p]
                ax.plot(*data['actual_xy'].T, color=color, label=label)
                ax.scatter(*data['actual_xy'][-1], color=color, marker='x' if row['event_contact'] else 'o')
                prog = data['path_progress'][np.isfinite(data['path_progress'])]
                if len(prog): max_progress = max(max_progress, float(prog.max()))
                t = data['relative_time']
                for a, key in zip(aa[:4], ('actual_forward_speed', 'path_cross_track', 'phi', 'phi_dot')):
                    a.plot(t, data[key], color=color, label=label)
                aa[4].plot(t, data['governed'][:, 0], color=color)
                aa[4].plot(t, data['governed'][:, 1], '--', color=color)
                idx = np.arange(0, len(t), 4)
                reward = np.add.reduceat(np.nan_to_num(data['scored_tick_reward']), idx)
                scored = np.logical_or.reduceat(np.isfinite(data['scored_tick_reward']), idx)
                costs = -.005 * manifest['config']['path_reward']['scale'] * np.add.reduceat(
                    np.nan_to_num(data['effective_costs']), idx, axis=0)
                reward[~scored] = np.nan
                costs[~scored] = np.nan
                np.testing.assert_allclose(np.nansum(reward), row['logged_suffix_return_while_upper_active'], atol=1e-4)
                remainder = reward - costs.sum(axis=1)
                aa[5].plot(t[idx], reward, color=color)
                aa[6].plot(t[idx], np.where(scored, np.nancumsum(reward), np.nan), color=color)
                np.savez_compressed(out / f'{directory.name}_{row["name"]}_reward.npz', relative_time=t[idx],
                                    step_reward=reward, signed_components=costs, failure_remainder=remainder,
                                    component_names=manifest['reward_components'])
                cf, ca = plt.subplots(figsize=(12, 5), layout='constrained')
                for i, name in enumerate(manifest['reward_components']): ca.plot(t[idx], costs[:, i], label=name)
                ca.plot(t[idx], remainder, 'k--', label='failure remainder')
                ca.set(xlabel='seconds after pulse start', ylabel='signed reward /20ms', title=label)
                ca.set_yscale('symlog', linthresh=.02)
                ca.legend(fontsize=7, ncol=3)
                cf.savefig(out / f'{directory.name}_{row["name"]}_components.png'); plt.close(cf)
            with np.load(directory / (base['route'] + '_path.npz')) as path:
                show = path['s'] <= max_progress + 1
                axes[b, p].plot(*path['xy'][show].T, '--', color='gray', label='original path')
            axes[b, p].set(title=f"{directory.name} {base['state_id']} F={base['force_N']:+g}N", xlabel='x /m', ylabel='y /m')
            axes[b, p].set_aspect('equal', adjustable='datalim'); axes[b, p].legend(fontsize=7)
            for a, label in zip(aa, ('speed /m/s', 'cross track /m', 'roll /rad', 'roll rate /rad/s', 'reference speed/steer', 'reward /20ms', 'cumulative reward')):
                a.set_ylabel(label); a.axvspan(0, .1, alpha=.08, color='gray'); a.axvline(.04, ls=':', color='gray'); a.axvline(.44, ls=':', color='gray')
            aa[0].legend(fontsize=7); aa[-1].set_xlabel('seconds after pulse start')
            for a in aa[5:]: a.set_yscale('symlog', linthresh=.1)
            detail.savefig(out / f'{directory.name}_{p}_timeseries.png'); plt.close(detail)
            pairs.append(pair)
    figure.savefig(out / 'all_xy.png'); plt.close(figure)
    assert ticks <= design['maximum_new_lower_ticks']
    atomic_json(out / 'analysis.json', dict(pairs=pairs, raw_sha256=raw_hashes, active_lower_ticks=ticks,
                                          training_steps=0, model_fits=0, scope=design['selection']))
    lines = ['# 40ms延迟后的既有收益诊断', '', '[全部XY路径](all_xy.png)', '',
             '目的与方法见design.json：后见选择两类既有收益，不是独立验证。每个干预配新执行的原Actor；脉冲仍从0开始，原Actor保持40ms，干预40–440ms，原10s截止与全部底层/滤波/幅度不变。', '',
             '|bank|状态|原策略不触地|延迟干预不触地|原策略联合达标|延迟干预联合达标|', '|---|---|---|---|---|---|']
    for p in pairs: lines.append(f"|{p['bank']}|{p['state_id']}|{p['baseline_safe']}|{p['delayed_safe']}|{p['baseline_joint']}|{p['delayed_joint']}|")
    lines += ['', f'实际新增{ticks}物理步，0训练/拟合。完整有符号分项、失败余项、20ms奖励与轨迹已保存。',
              '不据此声称在线触发、对其它力方向有效或不损害名义任务；这些条件没有在本诊断执行。没有阈值/延迟扫描。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return ticks


def execute(root):
    design = json.loads((root / 'design.json').read_text())
    state = dict(phase='running', pid=os.getpid(), declared_cases=8, completed_cases=0,
                 active_lower_ticks=0, training_steps=0, model_fits=0)
    def update(**changes):
        state.update(changes, checked_unix=time.time()); atomic_json(root / 'status.json', state)
    update(stage='source_preflight')
    try:
        for path, expected in design['source_sha256'].items():
            assert sha(path) == expected, path
        for plan in design['plans']:
            directory = Path(plan).parent
            with (directory / 'execution.log').open('w') as log:
                child = subprocess.Popen([sys.executable, 'learning/cli/probe_reference_prefix_return.py', '--plan', plan], stdout=log, stderr=subprocess.STDOUT)
                update(stage=directory.name, child_pid=child.pid)
                result = child.wait()
            child_state = json.loads((directory / 'status.json').read_text())
            state['active_lower_ticks'] += child_state['active_lower_ticks']
            if result or child_state['phase'] != 'completed': raise RuntimeError(f'{directory.name} child exit {result}')
            update(completed_cases=state['completed_cases'] + child_state['completed_cases'])
        update(stage='analysis')
        ticks = report(root, design)
        assert ticks == state['active_lower_ticks']
        update(phase='completed', stage='reported', report=str(root / 'analysis/REPORT.md'))
    except BaseException as exc:
        update(phase='error', error_type=type(exc).__name__, error=str(exc)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare else execute)(args.output.resolve())
