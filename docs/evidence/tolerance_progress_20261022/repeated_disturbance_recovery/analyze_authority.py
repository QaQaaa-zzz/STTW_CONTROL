"""Audit all A3 authority cases, including authenticated immediate-hold reuse."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sttw_control.balance_probe import classify_contact_balance
from sttw_control.preference_best import atomic_json, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    a = p.parse_args()
    root = Path(a.run).resolve()
    status = json.loads((root/'status.json').read_text())
    if status['phase'] != 'completed' or status['completed_cases'] != 240:
        raise ValueError('Wait for every declared A3 case')
    declaration = json.loads((root/'declaration.json').read_text())
    rows, traces, hashes = [], {}, {}

    def add(case, recorded, file, *, reused, profile=None):
        with np.load(file) as stream:
            trace = {k: stream[k] for k in stream.files}
        actual = classify_contact_balance(trace, horizon_s=4.)
        if actual['balance_label'] != recorded['balance_label'] or actual['active_ticks'] != recorded['active_ticks']:
            raise ValueError('Raw physical outcome differs from result')
        if actual['balance_label'] == 'ENGINEERING_ERROR':
            raise ValueError('Engineering error cannot be a physical negative')
        active = trace['active_tick'].astype(bool)
        if profile is not None:
            np.testing.assert_allclose(trace['reference'][active], profile[active], atol=1e-7)
        key = (case['state_id'], case['second_delay_s'], case['mode'], case['force_N'])
        if key in traces:
            raise ValueError('Duplicate context across reused and new results')
        traces[key] = {k: v[active] for k, v in trace.items()}
        hashes[str(file)] = digest(file)
        rows.append(dict(**case, **actual, reused=reused, raw_file=str(file)))

    for receipt in declaration['reused_immediate_hold']:
        if digest(receipt['raw_file']) != receipt['raw_sha256'] or digest(receipt['snapshot']) != receipt['snapshot_sha256']:
            raise ValueError('Reused physical identity changed')
        case = dict(receipt['case'], mode='hold', second_delay_s=0.)
        add(case, receipt['result'], Path(receipt['raw_file']), reused=True)
    for job in declaration['jobs']:
        folder = Path(job['plan']).parent
        plan = json.loads(Path(job['plan']).read_text())
        results = json.loads((folder/'results.json').read_text())['cases']
        if len(results) != job['cases']:
            raise ValueError('Missing probe cases')
        profiles = np.load(plan['reference_profiles_file'])
        for case, result, profile in zip(plan['cases'], results, profiles):
            if case['force_N'] != result['force_N']:
                raise ValueError('Force layout drift')
            add(case, result, folder/(result['name']+'.npz'), reused=False, profile=profile)
    if len(rows) != 270:
        raise ValueError('Combined context coverage incomplete')
    out = root/'report'
    out.mkdir(exist_ok=False)
    (out/'cases').mkdir()
    groups = defaultdict(list)
    for row in rows:
        groups[(row['state_id'], row['second_delay_s'])].append(row)
    paired, summary = [], []
    for (sid, delay), group in sorted(groups.items()):
        for mode in ['hold', 'accel', 'brake']:
            rr = [r for r in group if r['mode'] == mode and r['force_N'] != 0]
            zero = next(r for r in group if r['mode'] == mode and r['force_N'] == 0)
            assert len(rr) == 4
            summary.append(dict(state_id=sid, delay_s=delay, mode=mode,
                                nonzero_cases=4, complete_no_contact=sum(r['balance_label']=='NO_FLOOR_CONTACT' for r in rr),
                                contacts=sum(r['balance_label']=='FLOOR_CONTACT' for r in rr),
                                unknown=sum(r['balance_label']=='UNKNOWN' for r in rr),
                                zero_second_label=zero['balance_label']))
        for mode in ['accel', 'brake']:
            gains, losses, transitions = 0, 0, defaultdict(int)
            for force in [-48., -24., 24., 48.]:
                before = next(r for r in group if r['mode']=='hold' and r['force_N']==force)['balance_label']
                after = next(r for r in group if r['mode']==mode and r['force_N']==force)['balance_label']
                transitions[before+'→'+after] += 1
                gains += before=='FLOOR_CONTACT' and after=='NO_FLOOR_CONTACT'
                losses += before=='NO_FLOOR_CONTACT' and after=='FLOOR_CONTACT'
            paired.append(dict(state_id=sid, delay_s=delay, mode=mode,
                               contact_to_complete=gains, complete_to_contact=losses,
                               transitions=dict(transitions)))
    atomic_json(out/'analysis.json', dict(cases=rows, summary=summary, paired=paired,
        raw_sha256=hashes, new_lower_ticks=status['active_lower_ticks'], reused_cases=30,
        scope='Selected historical lower-only states, same complete state per intervention. '
              'No upper Actor, task reward, alpha-method benefit, or exact force threshold claim.'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(hold='black', accel='#0072B2', brake='#D55E00')
    links = []
    for (sid, delay), group in sorted(groups.items()):
        for force in [-48., -24., 0., 24., 48.]:
            fig, axes = plt.subplots(6, 1, figsize=(11, 13), layout='constrained')
            for mode in ['hold', 'accel', 'brake']:
                row = next(r for r in group if r['mode']==mode and r['force_N']==force)
                trace = traces[(sid, delay, mode, force)]
                time = trace['time']
                marker = 'x' if row['balance_label']=='FLOOR_CONTACT' else 'o' if row['balance_label']=='NO_FLOOR_CONTACT' else 's'
                xy = trace['actual_xy']-trace['actual_xy'][0]
                axes[0].plot(*xy.T, color=colors[mode], label=mode+' '+row['balance_label'])
                axes[0].scatter(*xy[-1], color=colors[mode], marker=marker)
                for ax, field in zip(axes[1:5], ['actual_forward_speed', 'phi', 'phi_dot', 'actual_delta']):
                    ax.plot(time, trace[field], color=colors[mode], label=mode)
                    ax.scatter(time[-1], trace[field][-1], color=colors[mode], marker=marker, s=18)
                axes[1].plot(time, trace['reference'][:, 0], color=colors[mode], ls='--', alpha=.6)
                axes[4].plot(time, trace['reference'][:, 1], color=colors[mode], ls='--', alpha=.6)
                axes[5].plot(time, np.linalg.norm(trace['applied_force_impulse'], axis=1)/.005,
                             color=colors[mode], label=mode)
            axes[0].set(xlabel='X relative to capture [m]', ylabel='Y [m]', aspect='equal')
            for ax, ylabel in zip(axes[1:], ['speed/ref [m/s]', 'roll [rad]', 'roll rate [rad/s]', 'steer/ref [rad]', 'force magnitude [N]']):
                ax.set(xlabel='time after capture [s]', ylabel=ylabel)
                ax.axvspan(delay, delay+.1, color='gray', alpha=.15)
            for ax in axes:
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            tag = f'{sid}_delay{delay:g}_F{force:+g}'
            fig.suptitle(tag+' | first pulse already in full physical history; no upper Actor')
            fig.savefig(out/'cases'/(tag+'.png'), dpi=100)
            plt.close(fig)
            links.append(tag)
    lines = ['# A3：首推后参考权限与再次受扰', '',
             '240新增分支＋30认证复用，共270例；冻结底层，三条固定可实现参考曲线。'
             '同一capture状态比较干预，二推世界方向固定，不读取未来二推信息。'
             'capture在首推释放后.05s；二推延迟0/.25/.6s。', '',
             '|状态|二推延迟s|模式|完整无触地/4受扰|触地|未知|二推零力|',
             '|---|---:|---|---:|---:|---:|---|']
    for row in summary:
        lines.append(f"|{row['state_id']}|{row['delay_s']:g}|{row['mode']}|{row['complete_no_contact']}|{row['contacts']}|{row['unknown']}|{row['zero_second_label']}|")
    lines += ['', '## 全部90工况真实轨迹', '',
              '每图含相同首推历史下三个模式的XY/真实终点、速度与实际参考、侧倾/角速度/转向、实际外力。'
              '没有中层任务奖励，不伪造V3 reward或任务恢复；幅度/时刻/状态为相关DEV，不能当独立训练重复。', '']
    lines += [f'- [{tag}](cases/{tag}.png)' for tag in links]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(out, 'all270 cases and90 figures verified')


if __name__ == '__main__':
    main()
