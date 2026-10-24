"""Prospective frozen-lookup confirmation, retaining unavailable and alias cases."""
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
    args = parser.parse_args()
    root = args.run.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    if status['completed_cases'] + status['unavailable_cases'] != 30:
        raise ValueError('all declared suffix slots must be accounted')
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    pairs, hashes = [], {}
    raw_ticks = 0
    for bank in plan['banks']:
        directory = Path(bank['directory'])
        pairing = json.loads((directory / 'pairing.json').read_text())
        rows = []
        evaluation = directory / 'evaluation'
        if pairing['real_suffix_cases']:
            state = json.loads((evaluation / 'status.json').read_text())
            if state['phase'] != 'completed' or state['completed_cases'] != pairing['real_suffix_cases']:
                raise ValueError('bank suffix incomplete')
            rows = json.loads((evaluation / 'results.json').read_text())['cases']
            manifest = json.loads((evaluation / 'manifest.json').read_text())
            raw_ticks += sum(r['active_ticks'] for r in rows)
        figure, axes = plt.subplots(3, 3, figsize=(18, 14), layout='constrained')
        for ax, pair in zip(axes.flat, pairing['pairs']):
            record = dict(pair, bank_index=bank['bank_index'])
            if pair['availability'] != 'AVAILABLE':
                ax.text(.5, .5, 'UNAVAILABLE: capture did not complete', ha='center', transform=ax.transAxes)
                pairs.append(record)
                continue
            base = rows[pair['baseline_index']]
            lookup = rows[pair['lookup_index']]
            if (base['snapshot_sha256'], base['force_N']) != (lookup['snapshot_sha256'], lookup['force_N']):
                raise ValueError('pair is not the same snapshot and disturbance')
            record.update(baseline_name=base['name'], lookup_name=lookup['name'],
                          exact_alias=pair['baseline_index'] == pair['lookup_index'],
                          baseline_safe=base['balance_label'] == 'NO_FLOOR_CONTACT',
                          lookup_safe=lookup['balance_label'] == 'NO_FLOOR_CONTACT',
                          baseline_joint=base['joint_success'], lookup_joint=lookup['joint_success'],
                          baseline_contact=base['event_contact'], lookup_contact=lookup['event_contact'],
                          physical_gain=bool(base['event_contact'] and lookup['balance_label'] == 'NO_FLOOR_CONTACT'),
                          physical_loss=bool(base['balance_label'] == 'NO_FLOOR_CONTACT' and lookup['event_contact']),
                          joint_gain=bool(lookup['joint_success'] and not base['joint_success']),
                          joint_loss=bool(base['joint_success'] and not lookup['joint_success']),
                          contact_to_joint_success=bool(base['event_contact'] and lookup['joint_success']))
            pairs.append(record)
            detail, aa = plt.subplots(8, 1, figsize=(11, 17), layout='constrained')
            component_fig, cc = plt.subplots(1, 2, figsize=(16, 5), layout='constrained')
            sources = [(base, 'black', 'original')]
            if not record['exact_alias']:
                sources.append((lookup, 'tab:orange', 'frozen ' + lookup['candidate']))
            progress = []
            for slot, (row, color, label) in enumerate(sources):
                file = evaluation / (row['name'] + '.npz')
                hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
                with np.load(file) as data:
                    trace = {k: data[k] for k in data.files}
                if len(trace['time']) != row['active_ticks'] or bool(trace['contact_failure'].any()) != row['event_contact']:
                    raise ValueError('raw outcome/count mismatch')
                if not np.allclose(np.diff(trace['time']), .005, atol=2e-6):
                    raise ValueError('task clock gap')
                after = trace['relative_time'] >= .4 - 1e-5
                valid_actor = trace['upper_active'].astype(bool)
                if trace['prefix_override'][after].any():
                    raise ValueError('override persisted after handback')
                np.testing.assert_array_equal(trace['executed_policy_latent'][after & valid_actor],
                                              trace['original_policy_latent'][after & valid_actor])
                finite_progress = trace['path_progress'][np.isfinite(trace['path_progress'])]
                if len(finite_progress):
                    progress.append(float(finite_progress.max()))
                t = trace['relative_time']
                text = f"{label}; {row['balance_label']}; task={row['task_recovered']}"
                marker = 'x' if row['event_contact'] else 'o' if row['full_observation'] else 's'
                ax.plot(*trace['actual_xy'].T, color=color, label=text)
                ax.scatter(*trace['actual_xy'][-1], color=color, marker=marker)
                for axis, key in zip(aa[:5], ['actual_forward_speed', 'path_cross_track', 'path_heading_error', 'phi', 'actual_delta']):
                    axis.plot(t, trace[key], color=color, label=text)
                aa[5].plot(t, trace['governed'][:, 0], color=color)
                aa[5].plot(t, trace['governed'][:, 1], '--', color=color)
                ix = np.arange(0, len(t), 4)
                reward = np.add.reduceat(np.nan_to_num(trace['scored_tick_reward']), ix)
                valid = np.logical_or.reduceat(np.isfinite(trace['scored_tick_reward']), ix)
                reward[~valid] = np.nan
                signed = -manifest['config']['path_reward']['scale'] * .005 * np.add.reduceat(np.nan_to_num(trace['effective_costs']), ix, axis=0)
                signed[~valid] = np.nan
                if not np.isclose(np.nansum(reward), row['logged_suffix_return_while_upper_active'], atol=1e-5):
                    raise ValueError('reward sum mismatch')
                remainder = reward - signed.sum(axis=1)
                aa[6].plot(t[ix], reward, color=color)
                cumulative = np.nancumsum(reward)
                cumulative[~valid] = np.nan
                aa[7].plot(t[ix], cumulative, color=color)
                for k, name in enumerate(manifest['reward_components']):
                    cc[slot].plot(t[ix], signed[:, k], label=name)
                cc[slot].plot(t[ix], remainder, 'k--', label='failure remainder')
                cc[slot].set_yscale('symlog', linthresh=.02)
                cc[slot].set_title(text, fontsize=8)
                cc[slot].legend(fontsize=6, ncol=3)
                np.savez_compressed(out / f"bank{bank['bank_index']}_{row['name']}_rewards.npz",
                                    relative_time=t[ix], task_time=trace['time'][ix], step_reward=reward,
                                    signed_components=signed, failure_remainder=remainder,
                                    component_names=manifest['reward_components'])
            with np.load(directory / 'path_evidence' / (pair['route'] + '_path.npz')) as path:
                show = path['s'] <= max(progress) + 1
                ax.plot(*path['xy'][show].T, '--', color='gray', label='original path, observed range + 1m')
            title = f"bank{bank['bank_index']}_{pair['route']}_v{pair['speed']:g}_a{pair['alpha']:g}_F{pair['force_N']:+g}"
            ax.set_title(title, fontsize=9)
            ax.set_aspect('equal', adjustable='datalim')
            ax.legend(fontsize=6)
            if record['exact_alias']:
                cc[1].set_visible(False)
            for axis, label in zip(aa, ['speed m/s', 'cross-track m', 'heading rad', 'roll rad', 'steer rad',
                                      'reference: speed solid, steer dashed', 'reward /20ms; symlog', 'cumulative reward; symlog']):
                axis.set_ylabel(label)
                axis.grid(alpha=.2)
                axis.axvspan(0, .1, color='gray', alpha=.1)
                axis.axvline(.4, color='gray', ls=':')
            for axis in aa[6:]:
                axis.set_yscale('symlog', linthresh=.1)
            aa[0].axhline(pair['speed'], color='gray', ls='--')
            aa[0].legend(fontsize=7)
            aa[-1].set_xlabel('seconds after authentic capture; original deadline retained')
            detail.suptitle(title + ('; lookup is exact baseline alias' if record['exact_alias'] else ''))
            detail.savefig(out / (title + '_timeseries.png'))
            component_fig.savefig(out / (title + '_components.png'))
            plt.close(detail)
            plt.close(component_fig)
        figure.savefig(out / f"bank{bank['bank_index']}_all_paths.png")
        plt.close(figure)
    if raw_ticks != status['suffix_lower_ticks']:
        raise ValueError('suffix physics accounting differs')
    summaries = []
    for bank in plan['bank_indices']:
        for zero in (True, False):
            group = [p for p in pairs if p['bank_index'] == bank and (p['force_N'] == 0) == zero]
            known = [p for p in group if p['availability'] == 'AVAILABLE']
            summaries.append(dict(bank_index=bank, force_group='zero' if zero else 'nonzero',
                                  declared_pairs=len(group), available_pairs=len(known), unavailable_pairs=len(group) - len(known),
                                  baseline_safe=sum(p['baseline_safe'] for p in known), lookup_safe=sum(p['lookup_safe'] for p in known),
                                  baseline_joint=sum(p['baseline_joint'] for p in known), lookup_joint=sum(p['lookup_joint'] for p in known),
                                  physical_gain=sum(p['physical_gain'] for p in known), physical_loss=sum(p['physical_loss'] for p in known),
                                  joint_gain=sum(p['joint_gain'] for p in known), joint_loss=sum(p['joint_loss'] for p in known),
                                  contact_to_joint_success=sum(p['contact_to_joint_success'] for p in known)))
    result = dict(paired=pairs, summary=summaries, raw_sha256=hashes, prefix_lower_ticks=status['prefix_lower_ticks'],
                  suffix_lower_ticks=status['suffix_lower_ticks'], total_new_lower_ticks=status['active_lower_ticks'],
                  actual_suffix_cases=status['completed_cases'], unavailable_case_slots=status['unavailable_cases'],
                  design_sha256=hashlib.sha256((root / 'design.json').read_bytes()).hexdigest(),
                  limits='Prospective same-context initial-state confirmation, shared prepared-bank ancestor. Frozen hand-written context lookup, known onset, no training or generic learned selector claim. Exact baseline aliases are not independent repeats.')
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 冻结参考规则的完整初态确认', '',
             '目的：裁决旧bank3发现的局部防摔/任务收益，换bank0/7后是否仍存在，以及是否损害无扰动任务。',
             '方法：原三情境、0/±32N/.1s、.4s干预后交回原alpha/H75、原10s截止；lookup在新结果前冻结。仅比较原策略与lookup，右1.8alpha1的lookup就是原策略，严格复用6条baseline，不重复积分或当独立样本。',
             '两个完整初态的速度/准备瞬态/路径原点共同变化，路径几何不变；不是单变量归因，也不是独立训练种子。不可捕获与新无扰动失败保留，不替换。', '',
             '[bank0全部路径](bank0_all_paths.png) · [bank7全部路径](bank7_all_paths.png)', '',
             '|bank|力组|可用/声明配对|原策略不触地|lookup不触地|原策略联合达标|lookup联合达标|防摔gain/loss|任务gain/loss|原触地→联合成功|',
             '|---:|---|---|---:|---:|---:|---:|---|---|---:|']
    for s in summaries:
        lines.append(f"|{s['bank_index']}|{s['force_group']}|{s['available_pairs']}/{s['declared_pairs']}|{s['baseline_safe']}|{s['lookup_safe']}|{s['baseline_joint']}|{s['lookup_joint']}|{s['physical_gain']}/{s['physical_loss']}|{s['joint_gain']}/{s['joint_loss']}|{s['contact_to_joint_success']}|")
    lines += ['', f"新增prefix={status['prefix_lower_ticks']}，suffix={status['suffix_lower_ticks']}，合计{status['active_lower_ticks']} lower ticks；0训练/拟合。", '',
              '逐配对表在analysis.json；全部实际速度、带符号路径/航向/侧倾、参考和每20ms真实奖励/分项/失败余项图及数组在本目录。域退出后的路径/reward为空，不伪造任务得分。',
              '停止：无论结果如何不加第三bank、不改lookup/动作/时机/力档。若收益未复制或名义损失抵消，则停止该lookup迁移主张；阳性也只能证明这两个新初态上的固定情境规则，不能当RA-L新方法。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
