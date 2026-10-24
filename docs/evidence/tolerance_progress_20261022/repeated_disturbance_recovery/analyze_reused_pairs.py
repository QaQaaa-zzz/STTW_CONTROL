"""Reinterpret existing transient60 as paired-pulse evidence; no new simulation."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(exist_ok=False)
    read = lambda name: json.loads((source/name).read_text())
    plan, results, preparation = read('scan_plan.json'), read('results.json'), read('prepare_plan.json')
    cases = plan['cases']
    if len(cases) != 60 or len(results['cases']) != 60:
        raise ValueError('Original transient60 coverage changed')
    groups, hashes, ticks = defaultdict(list), {}, 0
    for case, result in zip(cases, results['cases']):
        if case['force_N'] != result['force_N']:
            raise ValueError('Case layout mismatch')
        path = source/(result['name']+'.npz')
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as trace:
            active = trace['active_tick'].astype(bool)
            contact = trace['contact_failure'][active].astype(bool)
            error = trace['nonfinite'][active].astype(bool) | trace['lower_fault'][active].astype(bool)
            if error.any():
                raise ValueError('Engineering error is not a fall')
            actual = 'FLOOR_CONTACT' if contact.any() else ('NO_FLOOR_CONTACT' if active.sum() == 800 else 'UNKNOWN')
            if actual != result['balance_label'] or active.sum() != result['active_ticks']:
                raise ValueError('Raw contact accounting mismatch')
            ticks += int(active.sum())
        key = (case['speed_ref_m_s'], case['prime_force_N'], case['capture_s'])
        groups[key].append(dict(second_force_N=case['force_N'], outcome=actual,
                               raw_file=str(path)))
    rows = []
    for (speed, prime, onset), outcomes in sorted(groups.items()):
        zero = [r for r in outcomes if r['second_force_N'] == 0]
        if len(zero) != 1 or zero[0]['outcome'] != 'NO_FLOOR_CONTACT':
            raise ValueError('First-only continuation is not fully safe')
        rows.append(dict(speed_ref_m_s=speed, prime_force_N=prime,
                         onset_to_onset_s=onset,
                         first_release_to_second_onset_s=round(onset-preparation['prime_duration_s'], 8),
                         outcomes=outcomes))
    # This is a descriptive sensitivity check on already examined DEV data.
    # A later nominal phase is matched by the prime=0 controls; clocks differ
    # across intervals, so this is no proof of a causal recovery law.
    receipt = dict(source=str(source), source_files_sha256=hashes, rows=rows,
                   reused_cases=60, reused_active_lower_ticks=ticks,
                   new_physics_ticks=0, new_optimizer_updates=0,
                   scope='Historical lower-only fixed references. Middle history was not advanced; '
                         'these snapshots must not reset the middle Actor. No alpha-method benefit.')
    (output/'analysis.json').write_text(json.dumps(receipt, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    forces = [-48, -24, 0, 24, 48]
    matrix = np.array([[{'NO_FLOOR_CONTACT': 0, 'FLOOR_CONTACT': 1, 'UNKNOWN': 2}[
        next(o['outcome'] for o in r['outcomes'] if o['second_force_N'] == force)]
        for force in forces] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 7), layout='constrained')
    ax.imshow(matrix, cmap=ListedColormap(['#D6EAF8', '#F5B7B1', '#DDDDDD']),
              vmin=0, vmax=2, aspect='auto', interpolation='none')
    ax.set_xticks(range(5), [f'{f:+g}' for f in forces])
    ax.set_yticks(range(len(rows)), [f"v={r['speed_ref_m_s']:g}; first={r['prime_force_N']:+g} N; gap={r['first_release_to_second_onset_s']:g} s" for r in rows])
    for i in range(len(rows)):
        for j in range(5):
            ax.text(j, i, ['S', 'F', '?'][matrix[i, j]], ha='center', va='center')
    ax.set(xlabel='Second pulse body-relative signed force [N]',
           title='Reused transient60: S=complete 4 s without contact; F=contact\n'
                 'First pulse duration 0.1 s; gap measured after its release')
    fig.savefig(output/'paired_pulse_outcomes.png', dpi=160)
    fig.savefig(output/'paired_pulse_outcomes.pdf')
    plt.close(fig)
    lines = ['# A3入口：复用既有双脉冲数据', '',
             '目的：判断“扛住一次推力”与“已经恢复再次抗扰的能力”是否有区别。'
             '全部60例来自既有transient_balance_prediction_v1，0新仿真、0优化。'
             '这是历史DEV重分析，不是新验证或方法收益。', '',
             '![完整60例](paired_pulse_outcomes.png)', '',
             '第一脉冲0.1s，第二脉冲在首推开始后0.15或0.40s开始，'
             '因此首推结束后的间隔为0.05或0.30s；不能误写成释放后0.15/0.40s。'
             '每个第二脉冲观察4s，12条第二次零力均完整无触地。', '',
             '|速度参考 m/s|首推 N|开始至开始 s|释放后间隔 s|二推−48|−24|0|+24|+48|',
             '|---:|---:|---:|---:|---|---|---|---|---|']
    for row, values in zip(rows, matrix):
        lines.append(f"|{row['speed_ref_m_s']:g}|{row['prime_force_N']:+g}|"
                     f"{row['onset_to_onset_s']:g}|{row['first_release_to_second_onset_s']:g}|"
                     +'|'.join(['完整无触地', '触地', '未知'][v] for v in values)+'|')
    lines += ['', '## 当前证据与下一步', '',
              '2.55m/s、首推−24N时，短间隔的二推±48N都触地；长间隔−48N通过，'
              '+48N仍触地。1.65m/s的首推±24N在两时刻仍各有一个±48N方向触地。'
              '因此不能宣称“等0.3秒全部恢复”，也不能假定速度高始终更抗扰。'
              '零首推的匹配时刻全部力档均通过，支持首推历史会改变再次受力结局。', '',
              '接下来先检验原有参考权限能否改善这类方向性恢复过程，再决定是否训练。'
              '比较需保留同一首推历史、二推零力控制、同向/反向、多个间隔和全部失败；'
              '控制器不得读取未来二推时刻/幅值。较大间隔不预设单调，先画实测网格。'
              '当前数据中上层历史未随lower-only运行更新，不能直接作为中层重置。'
              '完整α中层收益必须用新的真实完整历史另证。', '',
              '原始代价仍计入原实验；重分析不重复计费。全部NPZ哈希、逐例和时间定义见analysis.json。'
              '图为一般科研分析用图，未作期刊版式合规声明。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(output)


if __name__ == '__main__':
    main()
