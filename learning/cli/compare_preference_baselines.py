"""Compare completed same-source preference runs using one original user objective."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sttw_control.preference_best import digest
from analyze_native_goal_confirmation import rescore_original_task


def compare(baselines, proposed, user_alpha):
    roots = [baselines.resolve(), proposed.resolve()]
    plans = [json.loads((r / 'plan.json').read_text()) for r in roots]
    baseline_returns = 'causal_preference_return' in plans[0]['protocol']
    rows = []
    for root, plan in zip(roots, plans):
        assert digest(root / 'plan.json') == json.loads((root / 'seal.json').read_text())['plan_sha256']
        assert json.loads((root / 'evaluation/status.json').read_text())['phase'] == 'completed'
        assert json.loads((root / 'analysis/receipt.json').read_text())['execution_reward_reconstructed']
        for row in json.loads((root / 'analysis/results.json').read_text()):
            assert digest(row['raw_file']) == row['raw_sha256']
            rows.append((root, plan, row))
    for key in ['config', 'initial_source', 'models']:
        assert plans[0][key] == plans[1][key], key
    proposed_contexts = {c['route']: c for c in plans[1]['contexts']}
    for context in plans[0]['contexts']:
        assert context == proposed_contexts[context['route']], context['route']
    out = roots[0] / 'analysis/matched_comparison'
    out.mkdir(exist_ok=True)
    groups = sorted({(r['route'], r['onset_s'], r['force_N']) for root, _, r in rows if root == roots[0]})
    records = []
    links = []
    for group_index, group in enumerate(groups):
        selected = [(root, plan, row) for root, plan, row in rows if (row['route'], row['onset_s'], row['force_N']) == group]
        assert len(selected) == 3 and sum(root == roots[0] for root, _, _ in selected) == 2, 'two baselines plus one proposed trajectory required'
        common = min(row['active_lower_ticks'] for _, _, row in selected)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
        axes = axes.ravel()
        source = selected[0][2]
        with np.load(selected[0][0] / 'evaluation' / (source['path_key'] + '_path.npz')) as z:
            path = dict(z)
        axes[0].plot(*path['xy'][path['s'] <= path['goal']].T, 'k:', label='Original reference')
        from matplotlib.patches import Circle
        axes[0].add_patch(Circle(source['original_goal_xy'], 1., fill=False, ec='black', ls=':', label='Original goal disk'))
        for index, (root, plan, row) in enumerate(selected):
            for key in ['force_N', 'duration_s', 'force_direction_world', 'source_task_start_s', 'original_goal_xy']:
                assert row[key] == source[key], key
            with np.load(row['raw_file']) as z:
                raw = dict(z)
            n = row['active_lower_ticks']
            ticks = row['replan_ticks']
            first = ticks[0] - round(row['source_task_start_s'] / .005) if ticks else None
            keys = json.loads((root / 'evaluation/manifest.json').read_text())['reward_components']
            task, audit = rescore_original_task(raw, path, plan['config'], plan['initial_source']['controller'], keys,
                first, broaden_cost_numerics=True, task_alpha=user_alpha,
                raw_alpha=row.get('raw_user_alpha', row['alpha']))
            baseline_label = f"Preset alpha={row['alpha']:g} + return" if baseline_returns else f"Fixed alpha={row['alpha']:g}"
            if row['alpha'] == user_alpha:
                baseline_label = f"Fixed user alpha={user_alpha:g}"
            label = baseline_label if root == roots[0] else 'Tolerance-guided + return'
            color, style = [('#0072B2', '--'), ('#D55E00', '-.'), ('#009E73', '-')][index]
            t = raw['time'][:n]
            marker = 'x' if row['endpoint']['physical_outcome'] == 'contact' else 's'
            axes[0].plot(*raw['actual_xy'][:n].T, color=color, ls=style, label=label)
            axes[0].scatter(*raw['actual_xy'][n-1], color=color, marker=marker)
            if row['endpoint']['goal_arrived']:
                arrival = np.argmin(abs(t-row['endpoint']['first_arrival_s']))
                axes[0].scatter(*raw['actual_xy'][arrival], color=color, marker='*', s=90, label='First goal arrival' if index == 0 else None)
            executed_alpha = np.full(n, row['alpha']) if root == roots[0] and not baseline_returns else raw['alpha_exec'][:n]
            values = [raw['actual_forward_speed'][:n], executed_alpha, task['original_cross_track_m'],
                      task['task_step_costs'].sum(axis=1), task['cumulative_task_costs'].sum(axis=1)]
            for ax, data in zip(axes[1:], values):
                ax.plot(t, data, color=color, ls=style, label=label)
                ax.scatter(t[-1], data[-1], color=color, marker=marker)
            name = f'group_{group_index:02d}_method_{index:02d}.npz'
            np.savez_compressed(out / name, time=t, actual_xy=raw['actual_xy'][:n],
                                actual_forward_speed=values[0], alpha_exec=values[1], **task)
            records.append(dict(label=label, case_id=row['case_id'], source_run=root.name,
                route=row['route'], force_N=row['force_N'], onset_s=row['onset_s'], endpoint=row['endpoint'], observed_ticks=n, common_ticks=common,
                full_window_costs=task['task_step_costs'].sum(axis=0).tolist(),
                common_window_costs=task['task_step_costs'][:common].sum(axis=0).tolist(),
                preference_deviation_integral=float(np.abs(values[1]-user_alpha).sum()*.005),
                raw_sha256=row['raw_sha256'], data_file=name, audit=audit))
        axes[0].set(xlabel='World X (m)', ylabel='World Y (m)', aspect='equal')
        for ax, label in zip(axes[1:], ['True forward speed (m/s)', 'Executing alpha', 'Original cross-track error (m)',
                                      'Original-user task cost / step', 'Cumulative original-user task cost']):
            ax.set(xlabel='Task time (s)', ylabel=label)
            ax.axvspan(source['onset_s'], source['onset_s']+source['duration_s'], color='gray', alpha=.15)
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0].legend(loc='upper center', bbox_to_anchor=(.5,-.25), ncol=2, fontsize=8)
        fig.suptitle(f"U170 | {group[0]} | gust {group[1]:g} s, {source['force_N']:g} N | user alpha={user_alpha:g}\n"
                     'Same complete source; x=contact, square=observed end; task costs exclude terminal penalty')
        filename = f'group_{group_index:02d}_comparison.png'
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
        links.append(filename)
    (out / 'results.json').write_text(json.dumps(dict(user_alpha=user_alpha, rows=records,
        new_physics_ticks=0, statistical_unit='paired deterministic branches; shared source',
        cost_scope='Original-path three running task channels, not total reward. Early failure shortens full-window costs.'), indent=2)+'\n')
    (out / 'README.md').write_text('# Matched preference comparison\n\n' + '\n\n'.join(
        f'![Actual/reference XY and original-user task costs]({name})' for name in links) +
        '\n\nSame original user objective; no smoothing. Full-window costs and common-prefix costs are separate; '
        'short failure costs do not indicate good task performance. See results.json for all endpoints and source identities.\n')
    print(json.dumps([{k:r[k] for k in ['label','onset_s','endpoint']} for r in records], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--proposed', type=Path, required=True)
    parser.add_argument('--user-alpha', type=float, default=0.)
    args = parser.parse_args()
    if not 0. <= args.user_alpha <= 1.:
        parser.error('user alpha must be in [0,1]')
    compare(args.baselines, args.proposed, args.user_alpha)
