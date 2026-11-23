"""Compare the second published ridge archive and quantify coordinate roundoff sensitivity."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from pcrp_replay_gate import KEYS, LABELS, REPO, compare_case, load_case, read

sys.path.insert(0, str(REPO / 'docs/paper/english'))
from analyze_matched_controls import read_case


def verified_s2(folder, oat):
    contract, lanes = read_case(folder, LABELS)
    for field in ('backend', 'identity', 'physics_dt', 'lower_dt', 'policy_dt'):
        if contract[field] != oat[field]:
            raise ValueError(f'{folder}: mismatched {field}')
    a, b = contract['settings'], oat['settings']
    for field in ('seed', 'initial_state', 'randomization', 'controller_config', 'plant_options',
                  'user_speed_m_s', 'lower_sha256', 'episode_s', 'auto_reset_horizon'):
        if a[field] != b[field]:
            raise ValueError(f'{folder}: mismatched {field}')
    if a['case_labels'][:6] != LABELS or b['case_labels'] != LABELS:
        raise ValueError(f'{folder}: unexpected lane labels')
    for field in ('alphas', 'zero_middle', 'zero_reference_channels'):
        if a[field][:6] != b[field]:
            raise ValueError(f'{folder}: mismatched {field}')
    if (REPO / a['asset']).read_bytes() != (REPO / b['asset']).read_bytes():
        raise ValueError(f'{folder}: different vehicle URDF')
    if any(x['sha256'] != y['sha256'] for x, y in zip(contract['terrain'][:6], oat['terrain'])):
        raise ValueError(f'{folder}: different terrain mesh')
    for field in ('randomization_readback', 'delay_steps'):
        for k, v in oat[field].items():
            np.testing.assert_array_equal(contract[field][k][:6], v, err_msg=f'{folder}: {field}.{k}')
    trajectories = [(trace[:, :2], np.rint(t / contract['physics_dt']).astype(int), row['terminal'])
                    for row, trace, t, *_ in lanes]
    return contract, trajectories


def sensitivity(left, right, origin_left, origin_right):
    xa, ta, _ = left
    xb, tb, _ = right
    n = min(len(ta), len(tb))
    # World-to-local subtraction has already occurred. Its floating point resolution depends
    # on the world coordinate before subtraction and the magnitude of the final local result.
    oa, ob = np.asarray(origin_left[:2]), np.asarray(origin_right[:2])
    scale = np.maximum.reduce([np.abs(xa[:n] + oa), np.abs(xb[:n] + ob),
                               np.abs(xa[:n]), np.abs(xb[:n])]).astype(np.float32)
    resolution = 2 * np.spacing(scale)
    diff = np.abs(xa[:n].astype(float) - xb[:n])
    primary = compare_case(left, right)
    return dict(**primary, c10_equal=(left[2] == 'full_window') == (right[2] == 'full_window'),
                xy_within_two_ulp=len(ta) == len(tb) and bool((diff <= resolution).all()),
                max_two_ulp_allowance_m=float(resolution.max()),
                max_diff_in_ulps=float((2*diff/resolution).max()),
                first_nonclose_two_ulp_tick=next((int(ta[k]) for k in
                                                  np.flatnonzero((diff > resolution).any(axis=1))), None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archives-only', action='store_true')
    args = parser.parse_args()
    root = args.run.resolve()
    runs = root.parent
    stage = root/'step0'
    rows = []
    for key in KEYS:
        folders = {'oat': runs/'h75_contact_oat/position'/key}
        samples = {'oat': load_case(folders['oat'])}
        if 'position_s4.5' in key:
            folders['s2'] = runs/'h75_theory_iteration/layout_test'/key.replace('position_s4.5', 'ridge100').replace('_seed', '_layout1_seed')
            samples['s2'] = verified_s2(folders['s2'], samples['oat'][0])
        if not args.archives_only:
            for name in ('repeat_1', 'repeat_2'):
                folders[name] = stage/name/key
                samples[name] = load_case(folders[name])
        names = list(samples)
        for j, left in enumerate(names):
            for right in names[:j]:
                ca, la = samples[left]
                cb, lb = samples[right]
                for i, lane in enumerate(LABELS):
                    oa, ob = ca['origin'][i], cb['origin'][i]
                    rows.append(dict(case=key, lane=lane, comparison=f'{left}_vs_{right}',
                                     left_dir=str(folders[left]), right_dir=str(folders[right]),
                                     left_code=ca['code_revision'], right_code=cb['code_revision'],
                                     left_slot=ca['batch_source']['lane_start']+i,
                                     right_slot=cb['batch_source']['lane_start']+i,
                                     left_origin_xy_m=str(oa[:2]), right_origin_xy_m=str(ob[:2]),
                                     **sensitivity(la[i], lb[i], oa, ob)))
    report = stage/'packing_sensitivity'
    report.mkdir(parents=True, exist_ok=True)
    label = 'archives' if args.archives_only else 'complete'
    with (report/f'{label}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    xy = ('[第 0 步实际与参考 XY 图和预登记主判断](../README.md)' if not args.archives_only else
          ' · '.join(f'[4.5 m 脊 OAT 原始实际与参考 XY · 组 {group}](' +
                      str(runs/f'h75_contact_oat/position/position_s4.5_v2.6_seed{group}/xy.png') + ')'
                      for group in (401, 402)))
    lines = ['# 第 0 步归档与坐标精度复查', '', xy, '',
             'OAT 为原位置扫描。S2 为论文接触机理图的同一 4.5 m 脊数据，来自另一个批次，仅取同物理组、同策略标签的前六条。两份档案的车道平移、批次排列和代码提交见逐条 CSV。对照逐项核对了地形网格、车辆、H75 与底层、初态、控制、采样频率、物理组参数及运行回读。', '',
             'XY 采用原始地形世界坐标，以实际观测终点为止。2 ULP 列取两个布局中绝对世界坐标和局部坐标幅度的最大值，逐点按 float32 的两倍相邻数间距比较，仅检验差异量级。两个 XY 列均要求完整轨迹长度相同。精确 tick 和终止类型另计。', '',
             '|比较|对数|完整主条件相符|XY 在 2 ULP 内|终止时刻相同|终止类型相同|C10 相同|共同前缀最大 XY 差（m）|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in dict.fromkeys(r['comparison'] for r in rows):
        subset = [r for r in rows if r['comparison'] == name]
        counts = [sum(r[field] for r in subset) for field in
                  ('passed', 'xy_within_two_ulp', 'times_equal', 'terminal_equal', 'c10_equal')]
        lines.append('| ' + ' | '.join([name, str(len(subset)), *map(str, counts),
                                        f"{max(r['max_xy_distance_m'] for r in subset):.6g}"]) + ' |')
    lines += ['', f'[逐条数值与数据来源]({label}.csv)', '']
    (report/f'{label}.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
