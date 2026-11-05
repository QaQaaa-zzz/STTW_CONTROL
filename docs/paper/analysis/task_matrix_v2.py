"""Analyze the declared matched route matrix against the original task."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'learning/cli'))
from analyze_native_goal_confirmation import original_path_projection_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True,
                        metavar='POLICY_OR_PURE_PURSUIT_RUN')
    parser.add_argument('--policy', choices=['H75', 'U170'], default='H75')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(args.runs) not in (2, 4, 6):
        parser.error('provide complete low, low+mid or low+mid+high U170/PP pairs')
    args.out.mkdir(parents=True, exist_ok=True)
    cases = defaultdict(dict)
    results = []
    paths = {}
    expected = {'low': (72, 24), 'mid': (72, 24), 'high': (72, 24)}
    speed_names = list(expected)[:len(args.runs) // 2]
    palette = {'PP': '#4d4d4d', 'a0': '#0072B2', 'a0.5': '#009E73', 'a1': '#D55E00'}

    for speed_name, (policy, pp) in zip(speed_names, zip(args.runs[::2], args.runs[1::2])):
        for method, run, count in [(args.policy, policy, expected[speed_name][0]),
                                   ('PP', pp, expected[speed_name][1])]:
            plan = json.loads((run / 'plan.json').read_text())
            protocol = plan['protocol']
            lower_dt = float(plan['config']['timing']['lower_dt_s'])
            horizon_ticks = round(float(protocol['horizon_s']) / lower_dt)
            status = json.loads((run / 'evaluation/status.json').read_text())
            index = json.loads((run / 'evaluation/case_index.json').read_text())
            if (protocol['capability_variant'] != 'task_matrix_v2' or
                status['phase'] != 'completed' or len(index) != count or
                status['completed_cases'] != count or
                bool(protocol.get('zero_middle', False)) != (method == 'PP') or
                (method == 'PP' and any(c['model_id'] != 'STTW_PP_ZERO_MIDDLE' for c in index))):
                raise ValueError(f'incomplete or mismatched task matrix: {run}')
            for case in index:
                key = (speed_name, case['bank_index'], case['route'], case['event'])
                label = 'PP' if method == 'PP' else f"a{case['alpha']:g}"
                path_key = (str(run), case['path_key'])
                if path_key not in paths:
                    with np.load(run / 'evaluation' / f"{case['path_key']}_path.npz") as z:
                        paths[path_key] = {field: z[field] for field in z.files}
                path = paths[path_key]
                with np.load(case['raw_file']) as z:
                    raw = {field: z[field] for field in z.files}
                n = int(np.asarray(raw['active_tick'], bool).sum())
                if method == 'PP' and np.any(raw['latent_z'][:n] != 0):
                    raise ValueError(f'nonzero middle correction in Pure Pursuit baseline: {case["case_id"]}')
                trace = original_path_projection_trace(raw, path, plan['config'])
                time = raw['time'][:n]
                xy = raw['actual_xy'][:n]
                speed = raw['actual_forward_speed'][:n]
                cross = trace[:, 1]
                turn_end = float(path['turn_end'])
                turn_start = 5. if protocol['route_contexts'][next(i for i, c in enumerate(protocol['route_contexts']) if c['route'] == case['route'])].get('reference_shape') == 'sharp_90' else 3.
                bend = (trace[:, 0] >= turn_start) & (trace[:, 0] <= turn_end + 2.)
                goal = np.asarray(case['original_goal_xy'])
                arrived = np.flatnonzero(np.linalg.norm(xy - goal, axis=1) <= 1.)
                fault = raw['policy_fault'][:n] | raw['lower_fault'][:n] | raw['nonfinite'][:n] | raw['continuation_geometry_fault'][:n]
                contact = bool(np.any(raw['contact_failure'][:n] | raw['physical_failure'][:n]))
                pulse = np.asarray(raw['applied_force_impulse'][:n]).sum(axis=0)
                row = dict(speed_group=speed_name, bank_index=case['bank_index'], route=case['route'],
                           event=case['event'], method=method, alpha=case['alpha'] if method != 'PP' else None,
                           requested_speed_m_s=case['speed'], original_goal_xy=goal.tolist(),
                           original_goal_s_m=float(path['goal']), bend_end_s_m=turn_end,
                           original_path_file=str(run / 'evaluation' / f"{case['path_key']}_path.npz"),
                           raw_file=case['raw_file'], raw_sha256=case['raw_sha256'],
                           observed_ticks=n, observed_s=n * lower_dt, full_horizon=n == horizon_ticks,
                           arrived=bool(len(arrived)), first_arrival_s=float(time[arrived[0]]) if len(arrived) else None,
                           contact=contact, fault=bool(np.any(fault)),
                           maximum_abs_roll_deg=float(np.max(np.abs(raw['phi'][:n])) * 180 / np.pi),
                           maximum_original_cross_m=float(np.max(np.abs(cross))),
                           original_cross_mae_m=float(np.mean(np.abs(cross))),
                           original_cross_rmse_m=float(np.sqrt(np.mean(cross * cross))),
                           bend_samples=int(bend.sum()),
                           bend_mean_speed_m_s=float(np.mean(speed[bend])) if bend.any() else None,
                           bend_original_cross_mae_m=float(np.mean(np.abs(cross[bend]))) if bend.any() else None,
                           bend_original_cross_rmse_m=float(np.sqrt(np.mean(cross[bend] ** 2))) if bend.any() else None,
                           minimum_forward_speed_m_s=float(np.min(speed)),
                           final_original_progress_m=float(trace[-1, 0]),
                           applied_world_impulse_Ns=pulse.tolist(),
                           reference_shape='sharp_90' if turn_start == 5. else 'smooth')
                results.append(row)
                if label in cases[key]:
                    raise ValueError(f'duplicate matched case {key}/{label}')
                cases[key][label] = (row, xy, time, speed, raw['phi'][:n], path)

    if len(results) != 96 * len(speed_names) or len(cases) != 24 * len(speed_names) or any(set(group) != set(palette) for group in cases.values()):
        raise ValueError('incomplete matched method/alpha matrix')
    figures = []
    for (speed_name, bank, route, event), group in sorted(cases.items()):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
        path = group['PP'][-1]
        mask = path['s'] <= float(path['goal'])
        axes[0].plot(*path['xy'][mask].T, color='black', ls='--', lw=1.5, label='Original reference')
        axes[0].scatter(*group['PP'][0]['original_goal_xy'], c='black', marker='*', s=60, label='Original goal')
        for label, color in palette.items():
            row, xy, time, speed, roll, _ = group[label]
            name = 'Pure Pursuit' if label == 'PP' else f'{args.policy} $\\alpha={row["alpha"]:g}$'
            axes[0].plot(*xy.T, color=color, label=name)
            axes[0].scatter(*xy[-1], c=color, marker='x' if row['contact'] else 's', s=24)
            axes[1].plot(time, speed, color=color, label=name)
            axes[2].plot(time, roll * 180 / np.pi, color=color, label=name)
        axes[0].set(xlabel='World X (m)', ylabel='World Y (m)')
        axes[0].set_aspect('equal')
        axes[1].axhline(group['PP'][0]['requested_speed_m_s'], color='black', ls='--', lw=1)
        axes[1].set(xlabel='Time (s)', ylabel='True forward speed (m/s)')
        axes[2].set(xlabel='Time (s)', ylabel='Roll (deg)')
        if event != 'zero':
            for ax in axes[1:]:
                ax.axvspan(2.8, 2.95, color='#bdbdbd', alpha=.3)
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{route} | bank{bank} | {event} | {speed_name} | original task, goal 7 m after bend')
        name = f'{speed_name}_bank{bank}_{route}_{event}_xy_speed_roll.png'
        fig.savefig(args.out / name, dpi=150)
        plt.close(fig)
        figures.append(name)
    summary = []
    for speed_name in speed_names:
        for event_type in ('zero', 'pulse'):
            for label in palette:
                rows = [r for r in results if r['speed_group'] == speed_name and
                        (r['event'] == 'zero') == (event_type == 'zero') and
                        ('PP' if r['method'] == 'PP' else f"a{r['alpha']:g}") == label]
                values = [r['bend_mean_speed_m_s'] for r in rows if r['bend_mean_speed_m_s'] is not None]
                summary.append(dict(speed_group=speed_name, event=event_type, method=label,
                                    cases=len(rows), arrived=sum(r['arrived'] for r in rows),
                                    full_horizon=sum(r['full_horizon'] for r in rows),
                                    contact=sum(r['contact'] for r in rows), fault=sum(r['fault'] for r in rows),
                                    bend_speed_observed_cases=len(values),
                                    bend_mean_speed_m_s=float(np.mean(values)) if values else None,
                                    bend_original_cross_mae_observed_cases=sum(r['bend_original_cross_mae_m'] is not None for r in rows),
                                    bend_original_cross_mae_m=float(np.mean([r['bend_original_cross_mae_m'] for r in rows if r['bend_original_cross_mae_m'] is not None])) if values else None,
                                    mean_peak_abs_roll_deg=float(np.mean([r['maximum_abs_roll_deg'] for r in rows])),
                                    mean_max_original_cross_m=float(np.mean([r['maximum_original_cross_m'] for r in rows]))))
    (args.out / 'case_metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Matched original-task route matrix', '',
             f'Exploratory matched physical evidence. {6 * len(speed_names)} routes, two complete state banks, nominal and +36 N world-Y pulse, {args.policy} alpha=0/0.5/1 and Pure Pursuit with zero middle correction: {len(results)} cases.',
             'All goals are 7 m past the last bend or exact corner. Original path and goal are scored independently of the 10 Hz execution reconnection. The sharp-90 reference is an exact polyline; its reference curvature array is zero by construction, not a rounded turn.',
             'Each trace panel ends at the actual stop; x marks contact, square marks other endpoints. The gray band is the 2.8-2.95 s applied-force interval. Full raw NPZ identities and case metrics are in case_metrics.json.', '',
             '| Speed | Event | Method | Cases | Goal | H10 | Contact | Fault | Bend speed (m/s) | Bend cross MAE (m) | Mean peak cross (m) | Mean peak roll (deg) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        bend_speed = f"{r['bend_mean_speed_m_s']:.3f}" if r['bend_mean_speed_m_s'] is not None else 'NA'
        bend_cross = f"{r['bend_original_cross_mae_m']:.3f}" if r['bend_original_cross_mae_m'] is not None else 'NA'
        lines.append(f"| {r['speed_group']} | {r['event']} | {r['method']} | {r['cases']} | {r['arrived']} | {r['full_horizon']} | {r['contact']} | {r['fault']} | {bend_speed} | {bend_cross} | {r['mean_max_original_cross_m']:.3f} | {r['mean_peak_abs_roll_deg']:.1f} |")
    lines.extend(['', '## Every route and event (two prepared states per cell)', '',
                  '| Speed | Route | Event | Method | Goal | H10 | Contact | Fault | Bend speed (m/s) | Bend cross MAE (m) | Peak roll (deg) |',
                  '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for speed_name in speed_names:
        routes = sorted({r['route'] for r in results if r['speed_group'] == speed_name})
        for route in routes:
            for event_type in ('zero', 'pulse'):
                for label in palette:
                    selected = [r for r in results if r['speed_group'] == speed_name and
                                r['route'] == route and (r['event'] == 'zero') == (event_type == 'zero') and
                                ('PP' if r['method'] == 'PP' else f"a{r['alpha']:g}") == label]
                    speed_values = [r['bend_mean_speed_m_s'] for r in selected if r['bend_mean_speed_m_s'] is not None]
                    cross_values = [r['bend_original_cross_mae_m'] for r in selected if r['bend_original_cross_mae_m'] is not None]
                    speed_text = f'{np.mean(speed_values):.3f}' if speed_values else 'NA'
                    cross_text = f'{np.mean(cross_values):.3f}' if cross_values else 'NA'
                    lines.append(f"| {speed_name} | {route} | {event_type} | {label} | "
                                 f"{sum(r['arrived'] for r in selected)}/2 | {sum(r['full_horizon'] for r in selected)}/2 | "
                                 f"{sum(r['contact'] for r in selected)}/2 | {sum(r['fault'] for r in selected)}/2 | "
                                 f"{speed_text} | {cross_text} | "
                                 f"{np.mean([r['maximum_abs_roll_deg'] for r in selected]):.1f} |")
    lines.extend(['', '## Actual/reference XY and true-speed panels', ''])
    lines.extend(f'- [{name}]({name})' for name in figures)
    (args.out / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
