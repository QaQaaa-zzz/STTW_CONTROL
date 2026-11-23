"""PCRP E1 feasibility score on measured trajectories, with the published task check."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_matched_controls import TASKS, read_case


def score_trajectories(folder, labels):
    contract, runs = read_case(folder, labels)
    bundle = Path(contract['bundle'])
    config = json.loads((bundle / 'config.json').read_text())
    limits = config['stage_B_baseline']
    with np.load(bundle / 'route.npz') as route:
        route_s, route_yaw = route['s'], route['heading']
    with np.load(folder / 'trajectory.npz') as data:
        yaw = data['pose'][:, :, 2]
    scores = []
    for lane, (row, trace, t, progress, _) in enumerate(runs):
        ratio = None
        if row['full']:
            if len(t) != len(yaw) or not np.isclose(t[-1], 10):
                raise ValueError(f'Incomplete full-window trajectory: {folder}, lane {lane}')
            # Same 200 post-control samples as the frozen final-hold implementation.
            tail = t > 10 - limits['hold_s'] + 1e-5
            heading_error = (np.interp(progress[tail], route_s, route_yaw)
                             - yaw[tail, lane] + np.pi) % (2*np.pi) - np.pi
            ratio = max(float(np.abs(trace[tail, 3]).max()) / limits['final_y_error_m'],
                        float(np.abs(heading_error).max()) / limits['final_path_heading_error_rad'],
                        float(np.abs(trace[tail, 2]-contract['settings']['user_speed_m_s']).max())
                        / limits['final_speed_error_m_s'])
            path_bound, speed_bound = TASKS['C']
            cost = max(row['peak_path_m'] / path_bound, row['speed_rmse_m_s'] / speed_bound, ratio)
        else:
            cost = 10 + (10 - row['time_s'])
        if bool(cost <= 1) != bool(row['C']):
            raise ValueError(f'Oracle and published task C disagree: {folder}, lane {lane}')
        scores.append(dict(**row, lane=lane, V=float(cost), hold_ratio=ratio))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    registration = json.loads((root/'registration.json').read_text())
    source = root.parent / 'pcrp_curvature_governor_20261121/source_cases.json'
    raw = json.loads(source.read_text())
    plots = []
    rows = []
    for layout in ('flat_original', 'gravel010', 'position_s3.9', 'position_s4.2',
                   'position_s4.5', 'position_s4.275'):
        for group in range(401, 405):
            name = f'{layout}_v2.6' + ('_layout0' if layout == 'gravel010' else '') + f'_seed{group}'
            if name not in raw:
                if layout != 'position_s4.275':
                    raise FileNotFoundError(name)
                plots.append((name, None, None, None, None))
                continue
            folder = Path(raw[name])
            case = json.loads((folder / 'contract.json').read_text())
            labels = case['settings']['case_labels'][:6]
            if (case['settings']['seed'] != group or labels != ['fixed0','fixed0.25','fixed0.5','fixed0.75','fixed1','Zero']
                    or case['identity']['files']['actor.npz'] != registration['middle_npz_sha256']
                    or case['settings']['lower_sha256'] != registration['lower_npz_sha256']
                    or case['identity']['files']['route.npz'] != registration['route_npz_sha256']):
                raise ValueError(f'Unmatched archived baseline: {folder}')
            scored = score_trajectories(folder, labels)
            rows.extend(dict(case=name, physical_group=group, **row) for row in scored)
            figure = next((folder/x for x in ['xy.png','xy_all_alpha.png','overview.png'] if (folder/x).is_file()), None)
            seed = min(scored[:5], key=lambda r: r['V'])
            incumbent = min(scored, key=lambda r: r['V'])
            plots.append((name, figure, seed, incumbent, sum(r['C'] for r in scored[:5])))
    output = root/'archive_baseline'
    output.mkdir(exist_ok=True)
    with (output/'scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# E1 历史基线与热启动分数','',
             '各案例的 XY 来自已落盘的 H75 五档和 Zero。V 按附件逐轨迹复算，C 取原论文评测代码。',
             '这里列出的都是历史数据。第 0 步通过后才可启动 E1 的 CEM 搜索。门槛、偏移复验和最终判决见判决实验方案。','',
             '|工况|历史轨迹|可作搜索起点的 α|五档与 Zero 中最低 V|五个固定档通过 C|',
             '|---|---|---|---:|---:|']
    for name, fig, seed, incumbent, successes in plots:
        if seed is None:
            lines.append(f'| {name} | 缺同组原档 | 待测 | 待测 | 待测 |')
            continue
        link = f'[XY]({fig})' if fig else f"[轨迹]({Path(seed['raw'])/'trajectory.npz'})"
        lines.append(f"| {name} | {link} | {seed['method']} | {incumbent['V']:.5f} ({incumbent['method']}) | {successes}/5 |")
    (output/'README.md').write_text('\n'.join(lines)+'\n\n[逐条分数、误差和真实终点](scores.csv)\n')


if __name__ == '__main__':
    main()
