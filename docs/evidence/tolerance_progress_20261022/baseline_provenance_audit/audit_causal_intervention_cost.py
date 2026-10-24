"""Read-only paired common-window costs; no new rollout or fitted model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, output):
    reference = json.loads((root / 'analysis/analysis.json').read_text())
    result = []
    hashes = {}
    for bank in ('bank3', 'bank0'):
        directory = root / bank
        manifest = json.loads((directory / 'manifest.json').read_text())
        rows = json.loads((directory / 'results.json').read_text())['cases']
        dt = manifest['config']['timing']['lower_dt_s']
        scale = manifest['config']['path_reward']['scale']
        for original, causal in zip(rows[::2], rows[1::2]):
            assert original['arm'] == 'original' and causal['arm'] == 'causal'
            for key in ('route', 'speed', 'alpha', 'force_N', 'pulse_onset_task_tick', 'world_direction'):
                assert original[key] == causal[key]
            data = []
            for row in (original, causal):
                path = directory / (row['name'] + '.npz')
                hashes[str(path)] = sha(path)
                assert hashes[str(path)] == reference['raw_sha256'][str(path)]
                with np.load(path) as z:
                    data.append({key: z[key] for key in z.files})
            a, b = data
            n = min(len(a['time']), len(b['time']))
            np.testing.assert_array_equal(a['time'][:n], b['time'][:n])
            # End at the FIRST invalid task tick in either arm. Do not delete
            # invalid ticks and stitch a favorable non-contiguous comparison.
            valid = a['upper_active'][:n] & b['upper_active'][:n]
            valid &= np.isfinite(a['effective_costs'][:n]).all(axis=1)
            valid &= np.isfinite(b['effective_costs'][:n]).all(axis=1)
            invalid = np.flatnonzero(~valid)
            stop = int(invalid[0]) if len(invalid) else n
            assert stop > 0
            measurements = []
            for row, d in zip((original, causal), data):
                common = slice(0, stop)
                costs = -dt * scale * d['effective_costs'][common].sum(axis=0, dtype=np.float64)
                measurements.append(dict(
                    arm=row['arm'], physical_label=row['balance_label'],
                    task_recovered=row['task_recovered'],
                    observed_physical_s=len(d['time']) * dt,
                    task_valid_s=int(d['upper_active'].sum()) * dt,
                    common_path_rmse_m=float(np.sqrt(np.mean(d['path_cross_track'][common] ** 2))),
                    common_speed_rmse_m_s=float(np.sqrt(np.mean((d['actual_forward_speed'][common] - row['speed']) ** 2))),
                    common_peak_roll_rad=float(np.max(np.abs(d['phi'][common]))),
                    common_signed_costs=dict(zip(manifest['reward_components'], costs.tolist())),
                    common_running_reward=float(costs.sum()),
                    override_s=float(d['override_active'].sum() * dt),
                    detection_time_s=row['detection_time_s']))
            # This is the paired change in actual delivered references over the
            # SAME valid prefix, not a counterfactual single-step action dose.
            delta = b['governed'][:stop] - a['governed'][:stop]
            result.append(dict(bank=bank, route=original['route'], speed=original['speed'],
                               alpha=original['alpha'], force_N=original['force_N'],
                               common_valid_s=stop * dt,
                               common_abs_reference_difference_integral=(np.abs(delta).sum(axis=0) * dt).tolist(),
                               measurements=measurements))
    output.mkdir(exist_ok=False)
    payload = dict(pairs=result, raw_sha256=hashes, new_physical_ticks=0, fits=0,
                   scope='Posthoc DEV common-prefix description; excludes failure penalty and invalid-domain tail. No whole-task preference guarantee or policy optimality.')
    (output / 'analysis.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 因果干预的共同有效窗口代价', '',
             '[全部实际XY](../REPORT.md)', '',
             '目的：量化既有防摔/任务收益付出了什么速度与路径代价，避免把终止更早误写成成本更低。只读24条已认证轨迹，新增0物理、0拟合。', '',
             '每对从任务起点比较到任一臂首次任务无效或物理观察结束；不跳过无效段再拼接。共同窗口长度因case而异，不能横向混成全任务收益。运行奖励只合计有符号分项，不含失败尾罚；物理结局与原任务标签独立保留。', '',
             '|bank/路线|力N|共同窗s|干预s|路径RMSE 原→干预 m|速度RMSE 原→干预 m/s|物理 原→干预|',
             '|---|---:|---:|---:|---|---|---|']
    for pair in result:
        a, b = pair['measurements']
        lines.append(f"|{pair['bank']}/{pair['route']}|{pair['force_N']:g}|{pair['common_valid_s']:.3f}|{b['override_s']:.2f}|{a['common_path_rmse_m']:.4f}→{b['common_path_rmse_m']:.4f}|{a['common_speed_rmse_m_s']:.4f}→{b['common_speed_rmse_m_s']:.4f}|{a['physical_label']}→{b['physical_label']}|")
    lines += ['', '这里只能量化alpha0两情境的实际代价，不说明其他alpha下仍保留偏好，也不是最小干预证明。共同前缀未覆盖域退出后的任务代价；该缺口必须随结局一起解释，不能用防摔抹去。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(output / 'REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.output.resolve())
