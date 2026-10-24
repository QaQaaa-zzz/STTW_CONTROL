"""Compare every zero-middle branch with existing H75 endpoints; no simulation."""
import argparse
import json
import hashlib
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from sttw_control.balance_probe import classify_contact_balance
from sttw_control.preference_best import atomic_json, digest


def read(path, receipts):
    path = Path(path).resolve()
    receipts[str(path)] = digest(path)
    return json.loads(path.read_text())


def load_trace(path, row, receipts):
    path = Path(path).resolve()
    receipts[str(path)] = digest(path)
    with np.load(path, allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    label = classify_contact_balance(trace, horizon_s=4.)
    if label['balance_label'] != row['balance_label']:
        raise ValueError('Contact label differs from raw log')
    if label['balance_label'] not in ['NO_FLOOR_CONTACT', 'FLOOR_CONTACT']:
        raise ValueError('Unresolved outcome; preserve it before interpreting comparison')
    active = trace['active_tick'].astype(bool)
    trace = {k: v[active] for k, v in trace.items()}
    if int(active.sum()) != row['active_ticks']:
        raise ValueError('Raw tick count mismatch')
    actual_impulse = trace['applied_force_impulse'].sum(0)
    expected = np.asarray(row['expected_impulse_Ns'])
    # Contact may truncate a pulse. Keep that valid failure rather than require
    # the full impulse after the simulator has already stopped.
    norm = np.linalg.norm(expected)
    if norm:
        projection = np.dot(actual_impulse, expected)/norm
        np.testing.assert_allclose(actual_impulse-projection*expected/norm, 0., atol=2e-4)
        if not -2e-4 <= projection <= norm+2e-4:
            raise ValueError('Impulse outside declared pulse')
    else:
        np.testing.assert_allclose(actual_impulse, 0., atol=2e-4)
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zero-campaign', type=Path)
    parser.add_argument('--h75-campaign', type=Path)
    parser.add_argument('--midpoint-campaign', type=Path)
    parser.add_argument('--nominal-utility', type=Path)
    parser.add_argument('--nominal-source-commit')
    args = parser.parse_args()
    base = Path.cwd()
    receipts, grouped, rows, new_ticks = {}, {}, [], 0
    if bool(args.zero_campaign) != bool(args.h75_campaign):
        raise ValueError('Both campaign paths required together')
    if args.midpoint_campaign and not (args.zero_campaign and args.nominal_utility):
        raise ValueError('Midpoint needs both physical comparators and frozen nominal utility')
    if args.midpoint_campaign:
        mid_status = read(args.midpoint_campaign/'campaign_status.json', receipts)
        if mid_status['phase'] != 'completed' or mid_status['completed_cases'] != 24:
            raise ValueError('Wait for every midpoint branch')
    pair_status = read(args.zero_campaign/'campaign_status.json' if args.zero_campaign else base/'runs/zero_middle_pair_status.json', receipts)
    if pair_status['phase'] != 'completed' or pair_status['completed_cases'] != 24:
        raise ValueError('Wait for complete pair; no partial success report')
    for side, h75run in [('left', 'alpha_physical_boundary_v1'), ('right', 'alpha_physical_boundary_mirror_v1')]:
        campaigns = [('zero_middle', f'zero_middle_physical_{side}_v1'), ('H75', h75run)]
        if args.midpoint_campaign:
            campaigns.append(('midpoint', str(args.midpoint_campaign)))
        for model, run in campaigns:
            campaign = args.zero_campaign if model == 'zero_middle' else args.midpoint_campaign if model == 'midpoint' else args.h75_campaign
            root = (campaign/side).resolve() if campaign else base/'runs'/run
            status = read(root/'campaign_status.json', receipts)
            if status['phase'] != 'completed':
                raise ValueError('Incomplete source '+run)
            preparation = read(root/'prepare_plan.json', receipts)
            if model == ('midpoint' if args.midpoint_campaign else 'zero_middle'):
                new_ticks += status['active_lower_ticks']
                if status['active_lower_ticks'] > preparation['maximum_total_lower_ticks']:
                    raise ValueError('Declared route budget exceeded')
            if model == 'zero_middle':
                if args.zero_campaign:
                    if preparation['verified_zero_inputs'] < 116:
                        raise ValueError('Missing constant policy verification')
                    if digest(preparation['model']['actor']) != preparation['model']['actor_sha256']:
                        raise ValueError('Zero actor changed')
                else:
                    read(root/'preexecution_verification.json', receipts)
            if model == 'midpoint':
                parent = read(args.h75_campaign/side/'prepare_plan.json', receipts)
                for field in ['model', 'source_declaration', 'path_evidence_directory', 'speeds',
                              'phase_fractions', 'forces_N', 'pulse_duration_s', 'observation_s']:
                    if preparation[field] != parent[field]:
                        raise ValueError('Midpoint protocol differs: '+field)
                if preparation['alphas'] != [.5]:
                    raise ValueError('Not the one declared midpoint')
                if digest(preparation['model']['actor']) != preparation['model']['actor_sha256']:
                    raise ValueError('Midpoint actor changed')
            prefixes = read(root/'prefix_results.json', receipts)['cases']
            results = read(root/'results.json', receipts)['cases']
            if len(results) != preparation['declared_cases']:
                raise ValueError('Declared case coverage mismatch')
            for row in results:
                if row['force_N'] not in [0., -80., 80.]:
                    continue
                trace = load_trace(root/(row['name']+'.npz'), row, receipts)
                label = 'zero_middle' if model == 'zero_middle' else f"H75_a{row['alpha']:g}"
                key = (side, row['requested_speed'], row['fraction'], row['force_N'])
                members = grouped.setdefault(key, {})
                if label in members:
                    raise ValueError('Duplicate matching key')
                prefix = next(x for x in prefixes if x['state_id'] == row['state_id'])
                snapshot = Path(prefix['snapshot'])
                if digest(snapshot) != prefix['snapshot_sha256']:
                    raise ValueError('Snapshot identity changed')
                prefix_file = root/f"prefix_v{prefix['speed']:g}_a{prefix['alpha']:g}.npz"
                receipts[str(prefix_file)] = digest(prefix_file)
                with np.load(prefix_file, allow_pickle=False) as z:
                    mask = z['time'] < prefix['task_time_s']-1e-6
                    prefix_xy = z['actual_xy'][mask]
                if model == 'zero_middle':
                    for field in ['original_policy_latent', 'executed_policy_latent', 'latent_z']:
                        np.testing.assert_array_equal(trace[field][trace['upper_active']], 0.)
                finite_path = np.isfinite(trace['path_cross_track'])
                scope = 'with_domain_hold' if row['fallback_triggered'] else 'follower_or_middle_active'
                details = dict(side=side, model=label, **row, scope=scope,
                               path_rmse_while_defined_m=float(np.sqrt(np.mean(trace['path_cross_track'][finite_path]**2))),
                               peak_abs_roll_rad=float(np.abs(trace['phi']).max()),
                               mean_speed_m_s=float(trace['actual_forward_speed'].mean()),
                               prefix_time_s=prefix['task_time_s'], prefix_speed_m_s=prefix['actual_speed_m_s'])
                rows.append(details)
                members[label] = dict(row=details, trace=trace, prefix_xy=prefix_xy, root=root)
    expected_rows = 96 if args.midpoint_campaign else 72
    if len(grouped) != 24 or len(rows) != expected_rows or new_ticks > 27200:
        raise ValueError('Matched coverage or budget differs')
    required = {'zero_middle', 'H75_a0', 'H75_a1'}
    if args.midpoint_campaign:
        required.add('H75_a0.5')
    if any(set(x) != required for x in grouped.values()):
        raise ValueError('Missing comparator')
    paired, totals = [], {}
    for label in sorted(required):
        rr = [x for x in rows if x['model'] == label]
        totals[label] = dict(all_cases=len(rr), all_labels=dict(Counter(x['balance_label'] for x in rr)),
                             disturbed_labels=dict(Counter(x['balance_label'] for x in rr if x['force_N'])),
                             zero_labels=dict(Counter(x['balance_label'] for x in rr if not x['force_N'])),
                             fallback_cases=sum(x['fallback_triggered'] for x in rr))
    for key, members in sorted(grouped.items()):
        paired.append(dict(side=key[0], speed=key[1], fraction=key[2], force_N=key[3],
                           labels={k:v['row']['balance_label'] for k,v in members.items()},
                           fallback={k:v['row']['fallback_triggered'] for k,v in members.items()}))
    contrasts = {}
    for label in ['H75_a0', 'H75_a1']:
        gains = [p for p in paired if p['labels'][label]=='NO_FLOOR_CONTACT' and p['labels']['zero_middle']=='FLOOR_CONTACT']
        losses = [p for p in paired if p['labels'][label]=='FLOOR_CONTACT' and p['labels']['zero_middle']=='NO_FLOOR_CONTACT']
        contrasts[label] = dict(gains=gains, losses=losses, net=len(gains)-len(losses))
    midpoint_comparisons, nominal = {}, None
    historical_sources = []
    if args.midpoint_campaign:
        for comparator in ['zero_middle', 'H75_a0', 'H75_a1']:
            disturbed = [p for p in paired if p['force_N']]
            gain = [p for p in disturbed if p['labels']['H75_a0.5']=='NO_FLOOR_CONTACT' and p['labels'][comparator]=='FLOOR_CONTACT']
            loss = [p for p in disturbed if p['labels']['H75_a0.5']=='FLOOR_CONTACT' and p['labels'][comparator]=='NO_FLOOR_CONTACT']
            midpoint_comparisons[comparator] = dict(gains=gain, losses=loss, net=len(gain)-len(loss))
        nominal_data = read(args.nominal_utility, receipts)
        for file, sha in nominal_data['input_sha256'].items():
            if digest(file) != sha:
                # The historical scorer was subsequently extended. Authenticate
                # its exact original bytes; raw evidence still must match live.
                scorer = Path(__file__).with_name('audit_preference_utility.py').resolve()
                if Path(file).resolve() != scorer or not args.nominal_source_commit:
                    raise ValueError('Frozen nominal evidence changed '+file)
                commit = subprocess.check_output(['git', 'rev-parse', args.nominal_source_commit+'^{commit}'], text=True).strip()
                relative = str(scorer.relative_to(base))
                blob = subprocess.check_output(['git', 'show', commit+':'+relative])
                if hashlib.sha256(blob).hexdigest() != sha:
                    raise ValueError('Original scorer does not match recorded evidence')
                historical_sources.append(dict(file=file, source_commit=commit,
                    recorded_sha256=sha, live_sha256=digest(file),
                    authentication='Exact historical Git bytes, not current scorer'))
                continue
            receipts[file] = sha
        nominal = []
        for side in ['left', 'right']:
            for speed in [2.35, 2.75]:
                rr = [r for r in nominal_data['trajectories'] if r['route'].startswith(side) and r['speed']==speed]
                mid = next(r for r in rr if r['policy']=='H75_a0.5')
                for comparator in ['zero_middle', 'H75_a1']:
                    other = next(r for r in rr if r['policy']==comparator)
                    nominal.append(dict(side=side, speed=speed, comparator=comparator,
                        utility_alphas=mid['utility_alphas'],
                        midpoint_minus_comparator=[a-b for a,b in zip(mid['returns'],other['returns'])],
                        midpoint_metrics=mid['metrics'], comparator_metrics=other['metrics']))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'cases').mkdir()
    strata = {}
    for field in ['side', 'requested_speed', 'fraction', 'force_N']:
        strata[field] = [dict(value=value, model=label,
            counts=dict(Counter(r['balance_label'] for r in rows
                        if r[field]==value and r['model']==label)),
            fallback_cases=sum(r['fallback_triggered'] for r in rows
                        if r[field]==value and r['model']==label))
            for value in sorted({r[field] for r in rows}) for label in sorted(required)]
    atomic_json(args.output/'analysis.json', dict(cases=rows, totals=totals, matched=paired, contrasts=contrasts,
                new_lower_ticks=new_ticks, new_training_steps=0, reused_H75_cases=48,
                reused_zero_cases=24 if args.midpoint_campaign else 0,
                midpoint_comparisons=midpoint_comparisons, nominal_utility=nominal, strata=strata, raw_sha256=receipts,
                authenticated_historical_sources=historical_sources,
                analysis_source_sha256=digest(__file__),
                scope='Own genuine prefixes at matched geometry. Existing DEV contexts; not same physical states or new method gains.'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'zero_middle':'#009E73', 'H75_a0':'#0072B2', 'H75_a1':'#D55E00', 'H75_a0.5':'#CC79A7'}
    links = []
    for key, members in sorted(grouped.items()):
        side, speed, fraction, force = key
        fig, axes = plt.subplots(9, 1, figsize=(11, 17), layout='constrained')
        for label, member in sorted(members.items()):
            row, trace, color = member['row'], member['trace'], colors[label]
            t = trace['time']
            xy = np.concatenate([member['prefix_xy'], trace['actual_xy']])
            axes[0].plot(*xy.T, color=color, label=label+' '+row['balance_label'])
            axes[0].scatter(*trace['actual_xy'][0], color=color, marker='^', s=20)
            axes[0].scatter(*trace['actual_xy'][-1], color=color, marker='x' if row['balance_label']=='FLOOR_CONTACT' else 'o')
            for ax, field in zip(axes[1:7], ['actual_forward_speed','path_cross_track','path_heading_error','phi','phi_dot','actual_delta']):
                ax.plot(t, trace[field], color=color, label=label)
            axes[1].plot(t, trace['reference'][:,0], color=color, ls='--', alpha=.6)
            axes[6].plot(t, trace['reference'][:,1], color=color, ls='--', alpha=.6)
            axes[7].plot(t, np.linalg.norm(trace['applied_force_impulse'],axis=1)/.005, color=color, label=label)
            axes[8].plot(t, trace['upper_active'].astype(float), color=color, label=label)
            if row['fallback_start_time_s'] is not None:
                for ax in axes[1:]:ax.axvline(row['fallback_start_time_s'], color=color, ls=':', alpha=.5)
        one = next(iter(members.values()))
        with np.load(one['root']/(one['row']['route']+'_path.npz')) as z:route = z['xy']
        all_xy = np.concatenate([np.concatenate([m['prefix_xy'],m['trace']['actual_xy']]) for m in members.values()])
        lo, hi = all_xy.min(0)-.5, all_xy.max(0)+.5
        visible = np.where(((route>=lo)&(route<=hi)).all(1)[:,None],route,np.nan)
        axes[0].plot(*visible.T, color='gray', ls=':', label='original fixed route')
        axes[0].set(xlabel='world X [m]',ylabel='world Y [m]',aspect='equal')
        axes[1].axhline(speed,color='gray',ls=':')
        for ax,ylabel in zip(axes[1:],['speed/ref [m/s]','signed path [m]','heading [rad]','roll [rad]','roll rate [rad/s]','steer/ref [rad]','force magnitude [N]','follower/middle active']):
            ax.set(xlabel='time since own geometric capture [s]',ylabel=ylabel)
            if force:ax.axvspan(0,.1,color='gray',alpha=.15)
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=6)
        tag=f'{side}_v{speed:g}_phase{fraction:g}_F{force:+g}'
        fig.suptitle(tag+' | own prefixes; zero middle vs H75; contact-only + common hold')
        fig.savefig(args.output/'cases'/(tag+'.png'),dpi=95);plt.close(fig);links.append(tag)
    lines=['# 当前协议：零中层与冻结H75各偏好的匹配对照','',
           ('24条新中间偏好分支＋72条既有端点/零中层分支。' if args.midpoint_campaign else '24条新零中层分支＋48条原H75分支。')+'相同任务几何、各自真实前缀；域退出后统一保持最后参考。',
           '这是完整系统比较，非同状态切换、非公式增量或独立最终测试。零策略仍含冻结local300。','',
           '|策略|非零力完整无触地/16|非零力触地/16|零力无触地/8|全部fallback|',
           '|---|---:|---:|---:|---:|']
    for label,x in totals.items():
        lines.append(f"|{label}|{x['disturbed_labels'].get('NO_FLOOR_CONTACT',0)}|{x['disturbed_labels'].get('FLOOR_CONTACT',0)}|{x['zero_labels'].get('NO_FLOOR_CONTACT',0)}|{x['fallback_cases']}|")
    lines += ['',f'新增准备＋接续共{new_ticks}个5ms底层步；0训练。原H75未重跑，历史交互不重复记新花费。',
              '路径曲线在域外缺失，停止处不可视作恢复；无触地仅指完整4s观察，不是无限时保证或任务到达。',
              '本组不是V3奖励优化对照；接管后不计算任务reward，不给域外虚构奖励。',
              '', '## 对照新增/丢失无触地', '']
    for label,x in contrasts.items():lines.append(f"- {label} 相对零中层：新增{len(x['gains'])}、丢失{len(x['losses'])}，净{x['net']:+d}；逐例与fallback见analysis.json。")
    if args.midpoint_campaign:
        lines += ['', '## 固定α=.5：防摔结局与独立名义效用', '']
        for label, x in midpoint_comparisons.items():
            lines.append(f"- 中点相对{label}：非零力新增{len(x['gains'])}、丢失{len(x['losses'])}；逐例见analysis.json。")
        lines += ['', '|路线/速度|参照|五用户α下中点减参照的名义效用|', '|---|---|---|']
        for row in nominal:
            lines.append(f"|{row['side']}/{row['speed']}|{row['comparator']}|"+', '.join(f'{x:+.5f}' for x in row['midpoint_minus_comparator'])+'|')
        lines += ['', '依次对应用户α=0/.25/.5/.75/1；效用来自已冻结完整10s名义轨迹，物理结局来自4s受扰分支，不合成受扰任务成功率。全部四相关上下文、单冻结模型，不是独立训练重复；无论结果均不再扩α/力/相位。',
                  '名义奖励/有符号分量及原始XY见nominal_preference_transfer_v1/analysis。当前受扰协议在fallback后无任务reward，不造域外分量。']
    lines += ['', '## 全部24个工况（含原始XY与真实终点）', '']
    lines += [f'- [{tag}](cases/{tag}.png)' for tag in links]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(totals=totals,new_lower_ticks=new_ticks,figures=len(links)),ensure_ascii=False))


if __name__ == '__main__':
    main()
