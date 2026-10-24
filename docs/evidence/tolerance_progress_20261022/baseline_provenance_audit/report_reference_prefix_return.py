"""All-case physical/task outcomes and paired opportunity bounds, no fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    status = json.loads((root / 'status.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = json.loads((root / 'results.json').read_text())['cases']
    if status['phase'] != 'completed' or status['completed_cases'] != len(rows) or len(rows) != 45:
        raise ValueError('all 45 branches must complete')
    out = args.output.resolve() if args.output else root / 'analysis'
    out.mkdir(exist_ok=False)
    names = [c['name'] for c in plan['candidates']]
    colors = dict(zip(names, ['black', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange']))
    contexts = sorted(set((r['state_id'], r['force_N']) for r in rows))
    pairs = []
    raw = {}
    figure, axes = plt.subplots(3, 3, figsize=(19, 16), layout='constrained')
    for ax, (sid, force) in zip(axes.flat, contexts):
        group = [r for r in rows if (r['state_id'], r['force_N']) == (sid, force)]
        if len(group) != 5 or {r['candidate'] for r in group} != set(names):
            raise ValueError('matched candidate roster differs')
        base = next(r for r in group if r['candidate'] == 'continue_original')
        observed_progress = []
        for row in group:
            with np.load(root / (row['name'] + '.npz')) as data:
                valid = data['path_progress'][np.isfinite(data['path_progress'])]
                if len(valid):
                    observed_progress.append(float(valid.max()))
        with np.load(root / (base['route'] + '_path.npz')) as path:
            display = path['s'] <= max(observed_progress) + 1
            ax.plot(*path['xy'][display].T, '--', color='gray', label='original task (observed range + 1m)')
        detail, aa = plt.subplots(8, 1, figsize=(11, 18), layout='constrained')
        components, cc = plt.subplots(3, 2, figsize=(16, 13), layout='constrained')
        for slot, row in enumerate(sorted(group, key=lambda r: names.index(r['candidate']))):
            file = root / (row['name'] + '.npz')
            raw[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            with np.load(file) as data:
                trace = {k: data[k] for k in data.files}
            if len(trace['time']) != row['active_ticks']:
                raise ValueError('active trace count differs')
            if not np.allclose(np.diff(trace['time']), .005, atol=2e-6):
                raise ValueError('task clock gap or reset')
            if bool(trace['contact_failure'].any()) != (row['balance_label'] == 'FLOOR_CONTACT'):
                raise ValueError('contact label differs')
            after = trace['relative_time'] >= plan['prefix_duration_s'] - 1e-5
            active = trace['upper_active'].astype(bool)
            if np.any(trace['prefix_override'][after]):
                raise ValueError('override continued after handback')
            np.testing.assert_array_equal(trace['executed_policy_latent'][after & active],
                                          trace['original_policy_latent'][after & active])
            t = trace['relative_time']
            color = colors[row['candidate']]
            label = f"{row['candidate']}: {row['balance_label']}, task={row['task_recovered']}"
            marker = 'x' if row['event_contact'] else 'o' if row['full_observation'] else 's'
            ax.plot(*trace['actual_xy'].T, color=color, label=label)
            ax.scatter(*trace['actual_xy'][-1], marker=marker, color=color)
            for axis, key in zip(aa[:6], ['actual_forward_speed', 'path_cross_track', 'path_heading_error', 'phi', 'actual_delta', 'governed']):
                if key == 'governed':
                    axis.plot(t, trace[key][:, 0], color=color)
                    axis.plot(t, trace[key][:, 1], '--', color=color)
                else:
                    axis.plot(t, trace[key], color=color, label=label)
            ix = np.arange(0, len(t), 4)
            scores = trace['scored_tick_reward']
            reward = np.add.reduceat(np.nan_to_num(scores), ix)
            valid = np.logical_or.reduceat(np.isfinite(scores), ix)
            reward[~valid] = np.nan
            signed = -manifest['config']['path_reward']['scale'] * .005 * np.add.reduceat(np.nan_to_num(trace['effective_costs']), ix, axis=0)
            signed[~valid] = np.nan
            if not np.isclose(np.nansum(reward), row['logged_suffix_return_while_upper_active'], atol=1e-5):
                raise ValueError('logged reward mismatch')
            remainder = reward - signed.sum(axis=1)
            aa[6].plot(t[ix], reward, color=color, label=label)
            cumulative = np.nancumsum(reward)
            cumulative[~valid] = np.nan
            aa[7].plot(t[ix], cumulative, color=color)
            for k, name in enumerate(manifest['reward_components']):
                cc.flat[slot].plot(t[ix], signed[:, k], label=name)
            cc.flat[slot].plot(t[ix], remainder, 'k--', label='failure remainder')
            cc.flat[slot].set_yscale('symlog', linthresh=.02)
            cc.flat[slot].set_title(label, fontsize=8)
            cc.flat[slot].legend(fontsize=6, ncol=3)
            np.savez_compressed(out / (row['name'] + '_rewards.npz'), relative_time=t[ix],
                                task_time=trace['time'][ix], step_reward=reward,
                                signed_components=signed, failure_remainder=remainder,
                                component_names=manifest['reward_components'])
            pairs.append(dict(state_id=sid, force_N=force, candidate=row['candidate'],
                              baseline_contact=base['event_contact'], contact=row['event_contact'],
                              baseline_joint=base['joint_success'], joint=row['joint_success'],
                              contact_to_no_contact=bool(base['event_contact'] and row['balance_label'] == 'NO_FLOOR_CONTACT'),
                              no_contact_to_contact=bool(base['balance_label'] == 'NO_FLOOR_CONTACT' and row['event_contact']),
                              contact_to_joint_success=bool(base['event_contact'] and row['joint_success']),
                              joint_gain=bool(row['joint_success'] and not base['joint_success']),
                              joint_loss=bool(base['joint_success'] and not row['joint_success'])))
        cc.flat[-1].set_visible(False)
        for axis, label in zip(aa, ['speed m/s', 'cross-track m', 'heading rad', 'roll rad', 'steer rad',
                                  'reference: speed solid / steer dashed', 'reward / 20ms; symlog', 'cumulative reward; symlog']):
            axis.axvspan(0, plan['pulse_duration_s'], color='gray', alpha=.12)
            axis.axvline(plan['prefix_duration_s'], color='gray', ls=':')
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
        for axis in aa[6:]:
            axis.set_yscale('symlog', linthresh=.1)
        aa[0].axhline(base['speed'], color='gray', ls='--')
        aa[0].legend(fontsize=7)
        aa[-1].set_xlabel('seconds since captured state; physical observation after domain exit has no task reward')
        title = f'{sid}_F{force:+g}'
        detail.suptitle(title + '; dotted = handback, shaded = pulse')
        detail.savefig(out / (title + '_timeseries.png'))
        components.savefig(out / (title + '_components.png'))
        plt.close(detail)
        plt.close(components)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('world X (m)')
        ax.set_ylabel('world Y (m)')
        ax.legend(fontsize=6)
    figure.savefig(out / 'all_paths.png')
    plt.close(figure)
    if sum(r['active_ticks'] for r in rows) != status['active_lower_ticks']:
        raise ValueError('new physics count differs')
    summary = []
    for force_group in ('zero', 'nonzero'):
        for name in names:
            rr = [r for r in rows if r['candidate'] == name and (r['force_N'] == 0) == (force_group == 'zero')]
            pp = [r for r in pairs if r['candidate'] == name and (r['force_N'] == 0) == (force_group == 'zero')]
            summary.append(dict(force_group=force_group, candidate=name, cases=len(rr),
                                contact=sum(r['event_contact'] for r in rr),
                                joint=sum(r['joint_success'] for r in rr),
                                unknown=sum(r['balance_label'] == 'UNKNOWN' for r in rr),
                                domain_handover=sum(r['fallback_triggered'] for r in rr),
                                contact_to_no_contact=sum(r['contact_to_no_contact'] for r in pp),
                                no_contact_to_contact=sum(r['no_contact_to_contact'] for r in pp),
                                contact_to_joint_success=sum(r['contact_to_joint_success'] for r in pp),
                                joint_gain=sum(r['joint_gain'] for r in pp), joint_loss=sum(r['joint_loss'] for r in pp)))
    opportunities = []
    for sid, force in contexts:
        group = [r for r in rows if (r['state_id'], r['force_N']) == (sid, force)]
        speed_set = [r for r in group if r['candidate'] in ('continue_original', 'speed_plus', 'speed_minus')]
        opportunities.append(dict(state_id=sid, force_N=force,
                                  finite_joint_oracle=any(r['joint_success'] for r in group),
                                  speed_union_joint_oracle=any(r['joint_success'] for r in speed_set)))
    result = dict(cases=rows, summary=summary, paired=pairs, opportunities=opportunities,
                  source_raw_sha256=raw, new_lower_ticks=status['active_lower_ticks'], training_updates=0, model_fits=0,
                  limits='Deterministic DEV, all three nominal-task-success states conditional subset. Known onset privileged. Per-event oracle is not a deployable selector or physical upper bound. Unchanged original task deadline.')
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'report_source.json').write_text(json.dumps(dict(
        reporter=str(Path(__file__).resolve()), reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_results_sha256=hashlib.sha256((root / 'results.json').read_bytes()).hexdigest(),
        changes='Post-run display crop and explicit contact-to-safe versus task recovery; raw outcomes unchanged.'), indent=2) + '\n')
    lines = ['# 有限参考干预能否交回原偏好策略', '',
             '目的：判断现有参考接口是否存在同时防摔和保住原任务的短暂干预，先核对有限机会，再考虑学习选择器。',
             '方法：完整8状态中全部3个名义任务成功状态；同完整快照、0/±32N×5候选，共45条。干预.4s，之后原alpha/H75继续，到原10s截止；原幅度/滤波/slew/历史不重置。速度和转向分别仅改自身通道。起扰时刻已知仅作离线机会诊断。', '',
             '[全部路径](all_paths.png)。各状态/力档的真实速度、带符号误差、参考、逐20ms奖励分项均在同目录；域退出后的奖励和路径为NaN，不虚构任务得分。', '',
             '|力组|候选|例数|触地|防摔且任务达标|UNKNOWN|域交接|相对原策略联合改善|联合损失|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('|' + '|'.join(str(r[k]) for k in ['force_group', 'candidate', 'cases', 'contact', 'joint', 'unknown', 'domain_handover', 'joint_gain', 'joint_loss']) + '|')
    lines += ['', f"新增lower ticks：{status['active_lower_ticks']}；0训练、0拟合。", '',
              '逐事件有限候选最优与速度候选并集仅是事后机会；若速度规则覆盖全部联合机会，则没有理由据此训练复杂模型。任一安全候选若仍丢原任务，不计联合恢复。候选清单、时长、力档不按本结果扩展。', '',
              '本组实证：speed_plus/steer_plus在同一个右1.5alpha0/−32N事件把触地变为不触地，但原任务仍未完成；steer_minus在左1.8alpha0/+32N实现任务恢复，但原策略该例本就未触地。原触地→未触地且任务成功的案例数为0，不能把两种改善拼成同一个联合救回。',
              '候选名称仅指相对初始offset的目标正负，不保证实际参考差或实际运动方向；handback_dose_audit.json保留对原Actor的实际参考差。初始分析使用完整70m路径导致显示过远，当前图仅裁参考显示范围至观测progress+1m，实际轨迹和所有case均保留。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
