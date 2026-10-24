"""All-case alpha physical grid audit; own-prefix differences are not alpha-switch effects."""
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
    status = json.loads((root/'campaign_status.json').read_text())
    preparation = json.loads((root/'prepare_plan.json').read_text())
    count = preparation['declared_cases']
    if status['phase'] != 'completed' or status['completed_cases'] != count:
        raise ValueError('Wait for every declared physical branch')
    prefix = json.loads((root/'prefix_results.json').read_text())['cases']
    plan = json.loads((root/'scan_plan.json').read_text())
    recorded = json.loads((root/'results.json').read_text())['cases']
    if len(prefix) != preparation['declared_prefixes'] or len(recorded) != count:
        raise ValueError('Coverage differs from fixed declaration')
    rows, traces, sources = [], {}, {}
    for row in recorded:
        file = root/(row['name']+'.npz')
        with np.load(file) as stream:
            trace = {k: stream[k] for k in stream.files}
        actual = classify_contact_balance(trace, horizon_s=4.)
        if actual['balance_label'] != row['balance_label'] or actual['active_ticks'] != row['active_ticks']:
            raise ValueError('Raw contact/time mismatch')
        if actual['balance_label'] == 'ENGINEERING_ERROR':
            raise ValueError('Engineering error cannot be a negative physical label')
        active = trace['active_tick'].astype(bool)
        trace = {k: v[active] for k, v in trace.items()}
        contact = np.flatnonzero(trace['contact_failure'])
        fallback = trace['fallback_active'].astype(bool)
        scope = ('contact_after_domain_handover' if fallback[contact[0]] else 'contact_with_middle_active') if len(contact) else ('complete_with_domain_handover' if fallback.any() else 'complete_middle_active') if actual['full_observation'] else 'UNKNOWN'
        row = dict(row, controller_scope=scope,
                   peak_abs_path_while_defined_m=float(np.nanmax(np.abs(trace['path_cross_track']))),
                   peak_abs_heading_while_defined_rad=float(np.nanmax(np.abs(trace['path_heading_error']))))
        rows.append(row)
        key = (row['requested_speed'], row['fraction'], row['alpha'], row['force_N'])
        if key in traces:
            raise ValueError('Duplicate physical context')
        traces[key] = trace
        sources[str(file)] = digest(file)
    if sum(row['active_ticks'] for row in rows) != json.loads((root/'status.json').read_text())['active_lower_ticks']:
        raise ValueError('Total active ticks differ')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['requested_speed'], row['fraction'], row['alpha'])].append(row)
    grids = []
    for key, group in sorted(groups.items()):
        zero = next(row for row in group if row['force_N'] == 0)
        for direction in [-1, 1]:
            ordered = [zero]+sorted([r for r in group if direction*r['force_N'] > 0], key=lambda r: abs(r['force_N']))
            safe = [abs(r['force_N']) for r in ordered if r['balance_label']=='NO_FLOOR_CONTACT']
            contact = [abs(r['force_N']) for r in ordered if r['balance_label']=='FLOOR_CONTACT']
            label = ('ZERO_FORCE_NOT_SAFE' if zero['balance_label']!='NO_FLOOR_CONTACT' else
                     'NONMONOTONIC_GRID' if safe and contact and max(safe)>min(contact) else
                     'UNKNOWN_IN_GRID' if any(r['balance_label']=='UNKNOWN' for r in ordered) else
                     'GRID_BRACKET' if contact else 'RIGHT_CENSORED_GRID')
            grids.append(dict(speed=key[0], fraction=key[1], alpha=key[2], direction=direction,
                              status=label, observed_max_safe_N=max(safe) if safe else None,
                              observed_min_contact_N=min(contact) if contact else None,
                              observations=[dict(force_N=r['force_N'], label=r['balance_label'], controller_scope=r['controller_scope']) for r in ordered]))
    paired = []
    for speed in [2.45, 2.65]:
        for fraction in [.2, .7]:
            for force in preparation['forces_N']:
                rr = [next(r for r in rows if r['requested_speed']==speed and r['fraction']==fraction and r['alpha']==alpha and r['force_N']==force) for alpha in [0., 1.]]
                paired.append(dict(speed=speed, fraction=fraction, force_N=force,
                                   alpha0=rr[0]['balance_label'], alpha1=rr[1]['balance_label'],
                                   alpha0_scope=rr[0]['controller_scope'], alpha1_scope=rr[1]['controller_scope']))
    out = root/'report'
    out.mkdir(exist_ok=False)
    (out/'cases').mkdir()
    atomic_json(out/'analysis.json', dict(cases=rows, prefixes=prefix, directional_grids=grids,
        matched_task_contexts=paired, raw_sha256=sources, total_new_lower_ticks=status['active_lower_ticks'],
        scope='Own complete middle prefixes, geometric position matched, common domain hold. '
              'Not same physical state across alpha; grid brackets require monotonicity for threshold interpretation.'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    links = []
    colors = {0.: '#0072B2', 1.: '#D55E00'}
    path_file = root/(plan['cases'][0]['route']+'_path.npz')
    with np.load(path_file) as stream:
        path_xy = stream['xy']
    for pair in paired:
        speed, fraction, force = pair['speed'], pair['fraction'], pair['force_N']
        fig, axes = plt.subplots(9, 1, figsize=(11, 17), layout='constrained')
        for alpha in [0., 1.]:
            trace = traces[(speed, fraction, alpha, force)]
            row = next(r for r in rows if r['requested_speed']==speed and r['fraction']==fraction and r['alpha']==alpha and r['force_N']==force)
            t, color = trace['time'], colors[alpha]
            marker = 'x' if row['balance_label']=='FLOOR_CONTACT' else 'o' if row['balance_label']=='NO_FLOOR_CONTACT' else 's'
            label = f"alpha={alpha:g} {row['balance_label']}"
            axes[0].plot(*trace['actual_xy'].T, color=color, label=label)
            axes[0].scatter(*trace['actual_xy'][-1], color=color, marker=marker)
            fields = ['actual_forward_speed', 'path_cross_track', 'path_heading_error', 'phi', 'phi_dot', 'actual_delta']
            for ax, field in zip(axes[1:7], fields):
                ax.plot(t, trace[field], color=color, label=f'alpha={alpha:g}')
            axes[1].plot(t, trace['reference'][:, 0], color=color, ls='--', alpha=.6)
            axes[6].plot(t, trace['reference'][:, 1], color=color, ls='--', alpha=.6)
            axes[7].plot(t, np.linalg.norm(trace['applied_force_impulse'], axis=1)/.005, color=color, label=f'alpha={alpha:g}')
            axes[8].plot(t, trace['upper_active'].astype(float), color=color, label=f'alpha={alpha:g}')
            if row['fallback_start_time_s'] is not None:
                for ax in axes[1:]:
                    ax.axvline(row['fallback_start_time_s'], color=color, ls=':', alpha=.6)
        # Select original route points near the actual observed world region;
        # preserve endpoints and do not extend a curve through missing history.
        xy = np.concatenate([traces[(speed, fraction, a, force)]['actual_xy'] for a in [0., 1.]])
        lo, hi = xy.min(0)-.5, xy.max(0)+.5
        mask = ((path_xy >= lo) & (path_xy <= hi)).all(1)
        visible_path = np.where(mask[:, None], path_xy, np.nan)
        axes[0].plot(*visible_path.T, color='gray', ls=':', label='original route in observed region')
        axes[0].set(xlabel='world X [m]', ylabel='world Y [m]', aspect='equal')
        axes[1].axhline(speed, color='gray', ls=':', label='original speed request')
        for ax, ylabel in zip(axes[1:], ['speed/ref [m/s]', 'signed path [m]', 'heading [rad]', 'roll [rad]', 'roll rate [rad/s]', 'steer/ref [rad]', 'force magnitude [N]', 'upper active']):
            ax.set(xlabel='time since geometric capture [s]', ylabel=ylabel)
            if force:
                ax.axvspan(0, .1, color='gray', alpha=.15)
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=6)
        tag = f'v{speed:g}_phase{fraction:g}_F{force:+g}'
        fig.suptitle(tag+' | same geometric phase, own true states; common domain hold')
        fig.savefig(out/'cases'/(tag+'.png'), dpi=95)
        plt.close(fig)
        links.append(tag)
    lines = ['# 冻结H75：几何相位匹配的α物理抗扰力档', '',
             f"八个完整真实中层前缀，{count}有符号分支；域内原H75，域退出后统一保持最后参考。"
             '跨α自身状态不同；这是完整偏好闭环＋共同fallback的能力，不是同状态切换救回或机械极限。', '',
             '|速度|相位|α|'+'|'.join(f'{f:+g}' for f in sorted(preparation['forces_N']))+'|',
             '|---:|---:|---:|'+'|'.join('---' for _ in preparation['forces_N'])+'|']
    for key, group in sorted(groups.items()):
        values = []
        for force in sorted(preparation['forces_N']):
            row = next(r for r in group if r['force_N']==force)
            values.append(('S' if row['balance_label']=='NO_FLOOR_CONTACT' else 'F' if row['balance_label']=='FLOOR_CONTACT' else '?')+('†' if row['fallback_triggered'] else ''))
        lines.append(f'|{key[0]:g}|{key[1]:g}|{key[2]:g}|'+'|'.join(values)+'|')
    lines += ['', 'S=完整4s无触地，F=真实非轮触地，?=未知；†表示曾交接到hold-last-reference。'
              '所有零力、失败、handover和实际前缀状态在analysis.json。方向网格不预设单调，最高已测通过只给下界。'
              '路径在fallback后NaN，不能将物理观察写作任务成功，也不虚构域外V3 reward。', '',
              f'## 全部{len(paired)}配对工况', '']
    lines += [f'- [{tag}](cases/{tag}.png)' for tag in links]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(out, count, 'raw outcomes;', len(paired), 'paired-context figures')


if __name__ == '__main__':
    main()
