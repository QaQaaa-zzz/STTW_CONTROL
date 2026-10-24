"""Read-only midpoint usefulness report; reuse endpoint physics without refits."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--midpoint', type=Path, required=True)
    parser.add_argument('--endpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = []
    records = []
    for directory, midpoint in [(args.endpoints, False), (args.midpoint, True)]:
        status = json.loads((directory / 'status.json').read_text())
        if status['phase'] != 'completed':
            raise ValueError(f'incomplete physics: {directory}')
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifests.append(manifest)
        cases = json.loads((directory / 'results.json').read_text())['cases']
        expected = 4 if midpoint else 16
        if len(cases) != expected:
            raise ValueError('unexpected case count')
        for row in cases:
            # Zero alpha contexts are physically redundant; retain one actual trace.
            if row['policy'] == 'zero' and row['alpha'] != 0:
                continue
            if row['policy'] not in ('zero', 'H75'):
                raise ValueError('unexpected policy')
            path = directory / (row['name'] + '.npz')
            with np.load(path) as data:
                trace = {key: data[key] for key in data.files}
            if np.any(trace['applied_force_impulse']):
                raise ValueError('nonzero force in nominal report')
            cfg = manifest['declaration']['config']
            radius = 1.1
            progress = trace['path_progress']
            turn = ((progress >= cfg['path']['straight_prefix_m']) &
                    (progress <= cfg['path']['straight_prefix_m'] + np.pi / 2 * radius +
                     cfg['path']['curvature_transition_length_m']))
            row = dict(row)
            row.update(
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source=str(path.resolve()),
                endpoint_s=float(trace['time'][-1] + .005),
                path_rmse_m=float(np.sqrt(np.mean(trace['path_cross_track'] ** 2))),
                speed_rmse_m_s=float(np.sqrt(np.mean((trace['actual_forward_speed'] - row['speed']) ** 2))),
                turn_path_rmse_m=float(np.sqrt(np.mean(trace['path_cross_track'][turn] ** 2))) if turn.any() else None,
                turn_speed_rmse_m_s=float(np.sqrt(np.mean((trace['actual_forward_speed'][turn] - row['speed']) ** 2))) if turn.any() else None,
                physical_outcome=('CONTACT' if row['physical_failure'] else
                                  'COMPLETE_NO_CONTACT' if row['complete_horizon'] else 'UNKNOWN'))
            indices = np.arange(0, len(trace['time']), 4)
            reward = np.add.reduceat(trace['scored_tick_reward'], indices)
            signed = -cfg['path_reward']['scale'] * .005 * np.add.reduceat(trace['effective_costs'], indices, axis=0)
            if not np.isclose(reward.sum(), row['logged_return'], atol=1e-5):
                raise ValueError('reward sum mismatch')
            records.append((row, trace, reward, signed, indices, directory))
    for field in ('physical_identity', 'source'):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f'identity mismatch: {field}')
    if manifests[0]['declaration']['config'] != manifests[1]['declaration']['config']:
        raise ValueError('effective config mismatch')
    out = args.output
    out.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    colors = {('H75', 0.): 'tab:blue', ('H75', .5): 'tab:green', ('H75', 1.): 'tab:orange', ('zero', 0.): 'black'}
    contexts = sorted(set((r['route'], r['speed']) for r, *_ in records))
    for ax, (route, speed) in zip(axes.flat, contexts):
        group = [item for item in records if (item[0]['route'], item[0]['speed']) == (route, speed)]
        if len(group) != 4:
            raise ValueError('missing matched context')
        with np.load(args.endpoints / (route + '_path.npz')) as path:
            max_progress = max(float(item[1]['path_progress'].max()) for item in group)
            xy = path['xy'][path['s'] <= max_progress + 1]
        ax.plot(*xy.T, '--', color='gray', label='original task path')
        detail, aa = plt.subplots(7, 1, figsize=(11, 16), layout='constrained')
        components, cc = plt.subplots(2, 2, figsize=(16, 9), layout='constrained')
        for slot, (row, trace, reward, signed, ix, _) in enumerate(sorted(group, key=lambda x: (x[0]['policy'], x[0]['alpha']))):
            color = colors[(row['policy'], row['alpha'])]
            label = f"{row['policy']} alpha={row['alpha']:g}; task={row['task_recovered']}; {row['physical_outcome']}"
            t = trace['time']
            marker = 'x' if row['physical_failure'] else 'o' if row['complete_horizon'] else 's'
            ax.plot(*trace['actual_xy'].T, color=color, label=label)
            ax.scatter(*trace['actual_xy'][-1], color=color, marker=marker)
            for axis, key in zip(aa[:5], ['actual_forward_speed', 'path_cross_track', 'path_heading_error', 'phi', 'actual_delta']):
                axis.plot(t, trace[key], color=color, label=label)
                axis.scatter(t[-1], trace[key][-1], color=color, marker=marker)
            aa[0].plot(t, trace['governed'][:, 0], '--', color=color, alpha=.4)
            aa[4].plot(t, trace['governed'][:, 1], '--', color=color, alpha=.4)
            aa[5].plot(t[ix], reward, color=color, label=label)
            aa[6].plot(t[ix], np.cumsum(reward), color=color)
            for k, name in enumerate(manifests[0]['reward_components']):
                cc.flat[slot].plot(t[ix], signed[:, k], label=name)
            remainder = reward - signed.sum(axis=1)
            cc.flat[slot].plot(t[ix], remainder, 'k--', label='failure remainder')
            cc.flat[slot].set_yscale('symlog', linthresh=.02)
            cc.flat[slot].set_ylabel('signed reward; symlog, linear |y|<=.02')
            cc.flat[slot].set_title(label, fontsize=9)
            cc.flat[slot].legend(fontsize=6, ncol=3)
            np.savez_compressed(out / (row['name'] + '_rewards.npz'),
                                time=t[ix], step_reward=reward, signed_components=signed,
                                component_names=manifests[0]['reward_components'], failure_remainder=remainder)
        ax.set(title=f'{route}, requested {speed:g} m/s', xlabel='world X (m)', ylabel='world Y (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=7)
        for axis, label in zip(aa, ['speed m/s; dashed ref', 'cross-track m', 'heading rad', 'roll rad', 'steer rad; dashed ref', 'reward / 20ms (own alpha)', 'cumulative reward (own alpha)']):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
        aa[0].axhline(speed, color='gray', ls=':')
        for axis in aa[5:]:
            axis.set_yscale('symlog', linthresh=.1)
            axis.set_ylabel(axis.get_ylabel() + '; symlog')
        aa[0].legend(fontsize=7)
        aa[-1].set_xlabel('task time (s); no external force')
        name = f'{route}_v{speed:g}'
        detail.savefig(out / (name + '_timeseries.png'))
        components.savefig(out / (name + '_components.png'))
        plt.close(detail)
        plt.close(components)
    fig.suptitle('One midpoint check; 4 new / 12 reused traces. Square = incomplete, not proven safe.')
    fig.savefig(out / 'all_paths.png')
    plt.close(fig)
    rows = [r for r, *_ in records]
    (out / 'analysis.json').write_text(json.dumps({'cases': rows, 'new_cases': 4, 'reused_cases': 12,
        'new_physics_ticks': json.loads((args.midpoint / 'status.json').read_text())['active_lower_ticks'],
        'new_policy_updates': 0, 'new_model_fits': 0,
        'limits': 'Adaptive deterministic DEV, same bank ancestor. RMSE over different observed horizons is descriptive, not an early-exit advantage. Rewards retain own alpha utility; not directly comparable across alpha.'}, indent=2) + '\n')
    lines = ['# 中间偏好是否有任务用途：一次有限检查', '',
             '目的：判断在相同低速紧弯参考权限下，冻结H75的alpha=.5能否提供端点之外的任务折中。',
             '方法：左右90度/R1.1、请求1.5/1.8m/s、同bank3和完整配置、10秒零外力；新运行4条alpha=.5，复用8条端点和4条真实zero-alpha0轨迹。不重新训练或拟合。', '',
             '[全部路径](all_paths.png)。逐步速度、带符号误差、参考、侧倾及奖励见timeseries；每20ms带符号奖励分项、失败余项和原始数组均保存。不同alpha奖励不是同一效用；提前退出不算完整不摔。', '',
             '|路径|请求速度|策略|alpha|任务达标|完整10s|物理结果|观察终点s|弯内路径RMSE m|弯内速度RMSE m/s|',
             '|---|---:|---|---:|---|---|---|---:|---:|---:|']
    for r in sorted(rows, key=lambda r: (r['route'], r['speed'], r['policy'], r['alpha'])):
        fields = ['route', 'speed', 'policy', 'alpha', 'task_recovered', 'complete_horizon', 'physical_outcome', 'endpoint_s', 'turn_path_rmse_m', 'turn_speed_rmse_m_s']
        lines.append('|' + '|'.join(f'{r[k]:.5g}' if isinstance(r[k], float) else str(r[k]) for k in fields) + '|')
    lines += ['', '停止规则：不补alpha=.25/.75，不因中点数值位于端点之间就扩力扫描；须另有明确任务用途才考虑新的物理能力实验。任务成功、全程不触地和容扰度分别判断。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps([{'policy': p, 'alpha': a, 'task': sum(r['task_recovered'] for r in rows if r['policy'] == p and r['alpha'] == a)} for p, a in colors]))


if __name__ == '__main__':
    main()
