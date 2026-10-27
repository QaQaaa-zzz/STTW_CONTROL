"""Complete paired reward/tracking figures from immutable evaluated traces only.

Separate post-analysis command: never imported by the sealed physical campaign.
No simulation, reselection, smoothing or invented post-contact observations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json, digest
from sttw_control.path_command_reward import configured_cost_keys

ROLES = ['goal_full_feedback', 'goal_zero_feedback', 'goal_steer_feedback',
         'neural_after_only', 'cem', 'random']
COLORS = ['#222222', '#777777', '#D55E00', '#AA4499', '#0072B2', '#009E73']
STYLES = ['-', ':', '-.', '--', '-', '--']


def policy_rewards(raw, scale):
    """Sum real physical substeps within original 20ms bins, retaining gaps."""
    active = np.asarray(raw['active_tick'], bool)
    ticks = np.asarray(raw['physical_tick'])[active].astype(int)
    if not len(ticks) or np.any(np.diff(ticks) != 1):
        raise ValueError('contiguous actual active physical ticks required')
    block = ticks // 4
    n = int(block.max()) + 1
    count = np.bincount(block, minlength=n)
    def summed(values):
        a = np.bincount(block, weights=np.asarray(values)[active], minlength=n)
        a[count == 0] = np.nan
        return a
    scored = summed(raw['scored_tick_reward'])
    running = summed(raw['tick_reward'])
    parts = np.column_stack([summed(-scale * .005 * raw['effective_costs'][:, j])
                             for j in range(raw['effective_costs'].shape[1])])
    np.testing.assert_allclose(parts.sum(axis=1), running, rtol=2e-5, atol=2e-6,
                               equal_nan=True)
    return np.arange(n), scored, parts, scored-running


def diagnose(root, *, confirmation=False,support_permission=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if confirmation and support_permission:raise ValueError('one explicit analysis protocol only')
    root = root.resolve(); source = root/('analysis' if support_permission else 'preservation_timecheck_v1/analysis' if confirmation else 'production/analysis')
    if json.loads((source/'status.json').read_text())['phase'] != 'completed':
        raise ValueError('complete audited final analysis required')
    rows = json.loads((source/'results.json').read_text())
    expected_rows=216 if support_permission else 432 if confirmation else 360
    expected_groups=108 if (confirmation or support_permission) else 60
    roles=['full','support_permission'] if support_permission else ['full','context_choice','geometry_alpha','alpha_only'] if confirmation else ROLES
    colors=['#222222','#0072B2'] if support_permission else ['#222222','#0072B2','#D55E00','#009E73'] if confirmation else COLORS
    styles=['-','--'] if support_permission else ['-','-','--',':'] if confirmation else STYLES
    if len(rows) != expected_rows: raise ValueError('complete declared final traces required')
    cfg = json.loads((root/'common_environment_config.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    keys = configured_cost_keys(cfg)
    if len(keys) != 10: raise ValueError('declared ten-component reward required')
    out = source/'diagnostics'; out.mkdir(exist_ok=False)
    atomic_json(out/'status.json', dict(phase='running', new_physics_steps=0))
    groups = {}
    for r in rows:
        group = groups.setdefault((r['context'], r['alpha'], r['event_id']), {})
        if r['role'] in group: raise ValueError('duplicate method')
        group[r['role']] = r
    if len(groups) != expected_groups or any(set(g) != set(roles) for g in groups.values()):
        raise ValueError('all declared matched method/events required')
    index = []
    for i, ((context, alpha, event), group) in enumerate(groups.items()):
        reward, aa = plt.subplots(4, 3, figsize=(15, 11), layout='constrained')
        tracking, bb = plt.subplots(4, 2, figsize=(15, 12), layout='constrained')
        speed, cc = plt.subplots(3, 2, figsize=(14, 10), layout='constrained')
        path_file = plan['contexts'][context]['path']
        if digest(path_file) != plan['contexts'][context]['path_sha256']:
            raise ValueError('original path changed')
        with np.load(path_file) as path:
            goal = np.array([np.interp(float(path['goal']), path['s'], path['xy'][:, j])
                             for j in range(2)])
        for method_index, (role, color, style) in enumerate(zip(roles, colors, styles)):
            r = group[role]
            if digest(r['raw_file']) != r['raw_sha256']: raise ValueError('raw changed')
            with np.load(r['raw_file']) as z: raw = {k: z[k] for k in z.files}
            x, total, parts, terminal = policy_rewards(raw, cfg['path_reward']['scale'])
            label = role + (' (privileged onset)' if role == 'neural_after_only' else '')
            aa.flat[0].plot(x, total, color=color, ls=style, label=label, lw=.9)
            for j in range(10): aa.flat[j+1].plot(x, parts[:, j], color=color, ls=style, lw=.9)
            aa.flat[11].plot(x, terminal, color=color, ls=style, lw=.9)
            a = raw['active_tick'].astype(bool)
            t = (raw['physical_tick'][a]+1)*.005
            sx = cc.flat[method_index]
            for key, speed_style, name in [('actual_forward_speed', '-', 'Actual body speed'),
                                     ('wheel_speed_proxy', ':', 'Rear-wheel proxy'),
                                     ('requested_speed', '-.', 'Original requested speed')]:
                sx.plot(t, raw[key][a], ls=speed_style, label=name)
            sx.plot(t, raw['reference'][a, 0], ls='--', label='Adjusted speed reference')
            sx.set(title=label, xlabel='Original task time (s)', ylabel='Speed (m/s)')
            sx.grid(alpha=.2)
            values = [raw['actual_forward_speed'][a], raw['reference'][a, 0],
                      raw['wheel_speed_proxy'][a]-raw['actual_forward_speed'][a],
                      raw['actual_delta'][a], raw['reference'][a, 1], raw['phi'][a],
                      raw['path_cross_track'][a], np.linalg.norm(raw['actual_xy'][a]-goal, axis=1)]
            for ax, y in zip(bb.flat, values):
                ax.plot(t, y, color=color, ls=style, label=label, lw=.9)
                ax.scatter(t[-1], y[-1], marker='x' if r['physical_outcome']=='contact' else 'o',
                           s=12, color=color)
            if role in ['goal_full_feedback','full']:
                bb.flat[0].plot(t, raw['requested_speed'][a], color='#555555', ls=':',
                                label='Original requested speed', lw=1.5)
        for ax, title in zip(aa.flat, ['Actual scored reward /20ms', *keys, 'Terminal replacement']):
            ax.set(title=title, xlabel='Original policy step (20ms)', ylabel='Signed reward')
            ax.grid(alpha=.2)
        for ax, title, unit in zip(bb.flat,
                ['Actual body speed', 'Adjusted speed reference', 'Rear-wheel proxy minus body speed',
                 'Actual steering', 'Adjusted steering reference', 'Roll', 'Original path cross-track',
                 'Original world-goal distance'],
                ['m/s', 'm/s', 'm/s', 'rad', 'rad', 'rad', 'm', 'm']):
            ax.set(title=title, xlabel='Original task time (s)', ylabel=unit); ax.grid(alpha=.2)
        bb.flat[7].axhline(1., color='gray', ls=':', lw=.8)
        for extra in list(cc.flat)[len(roles):]:extra.set_visible(False)
        r = group[roles[0]]; start = r['pulse_onset_task_tick']*.005
        if r['force_N']:
            for axes, scale in [(aa, .02), (bb, 1.), (cc, 1.)]:
                for ax in axes.flat:
                    ax.axvspan(start/scale, (start+r['pulse_duration_s'])/scale,
                               color='gray', alpha=.12)
        title = f'{context} / original alpha{alpha} / {event} / force {r["force_N"]:g}N / seen-ancestor DEV\nReal active samples only; true endpoints; no smoothing; all {len(roles)} methods'
        for fig, axes, suffix in [(reward, aa, 'reward'), (tracking, bb, 'tracking'), (speed, cc, 'speed')]:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=7)
            fig.suptitle(title, fontsize=10)
            fig.savefig(out/f'event_{i:02d}_{suffix}.png', dpi=110, facecolor='white')
            plt.close(fig)
        index.append(dict(context=context, alpha=alpha, event_id=event, force_N=r['force_N'],
                          reward=f'event_{i:02d}_reward.png', tracking=f'event_{i:02d}_tracking.png',
                          speed=f'event_{i:02d}_speed.png',
                          raw_sha256={role: group[role]['raw_sha256'] for role in roles}))
        atomic_json(out/'status.json', dict(phase='running', completed_groups=i+1, new_physics_steps=0))
    atomic_json(out/'index.json', index)
    (out/'README.md').write_text(f'# 全部{expected_groups}事件逐步奖励与跟踪诊断\n\n'
        f'{expected_rows}逻辑轨迹原raw SHA复核，共享轨迹不当独立样本；每图同初态/用户alpha/事件全部方法。原10秒、真实终点、扰动阴影；'+
        ('after-only含特权起扰信息。' if 'neural_after_only' in roles else '控制器不读取本次真实未来起扰。')+'原5ms奖励按原20ms时隙求和，10分项和terminal替代单列；'
        '缺测不补零或续线，图不平滑。轮速proxy减真实车体速度另列，不能当真值。'
        '原路径误差和原世界目标距离不随参考修改。无新积分/训练/候选选择，生成不等于目视审阅。'
        '索引和全部源SHA见index.json；PNG为研究诊断，未核期刊输出要求。\n\n' +
        '\n'.join(f'- {r["context"]} α{r["alpha"]} F{r["force_N"]:g}: '
                  f'[reward]({r["reward"]}) / [tracking]({r["tracking"]}) / [speed]({r["speed"]})' for r in index)+'\n')
    atomic_json(out/'status.json', dict(phase='completed', groups=expected_groups, traces=expected_rows,
        figures=3*expected_groups, new_physics_steps=0, visual_review_complete=False,
        source_results_sha256=digest(source/'results.json'), script_sha256=digest(__file__)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path);p.add_argument('--confirmation',action='store_true');p.add_argument('--support-permission',action='store_true');a=p.parse_args()
    diagnose(a.root,confirmation=a.confirmation,support_permission=a.support_permission)
