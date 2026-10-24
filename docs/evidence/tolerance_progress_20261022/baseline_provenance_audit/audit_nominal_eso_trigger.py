"""Frozen development-only causal single-shot trigger replay; no new physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RULE = dict(signal='negative latest-minus-previous normalized ESO equilibrium shift',
            latest_index=318, previous_index=298, mask_indices=[334, 335],
            threshold_normalized=.015, equality_triggers=True,
            trigger='first valid policy decision with abs(signal)>=threshold',
            reset='episode reset only; at most one activation per task',
            invalid_input='no activation; count and retain unavailable decisions',
            nominal_selection='all eight H75 endpoint zero-force R1p1 traces, including failures',
            development_basis='Round threshold between previously inspected captured-suffix max .007476 '
                              'and nonzero force 40ms min .026744. New full-task replay is DEV, not independent calibration.',
            forbidden_inputs=['force value/direction', 'pulse onset', 'capture time', 'future frames', 'eventual outcomes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.worktree.resolve()
    args.output.mkdir(exist_ok=False)
    # Write before opening nominal trajectories; never tune after seeing them.
    (args.output / 'rule_before_replay.json').write_text(json.dumps(RULE, indent=2) + '\n')
    sources = sorted((root / 'runs/low_tight_preference_nominal_v1_attempt2/R1p1/evaluation').glob('H75*_f+0.npz'))
    assert len(sources) == 8
    records, hashes = [], {}
    for source in sources:
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source) as z:
            data = {k: z[k] for k in z.files}
        active = data['active_tick'].astype(bool)
        indices = np.arange(0, len(active), 4)
        indices = indices[active[indices]]
        obs = data['upper_observation'][indices]
        assert obs.shape[1] == 351
        times = data['time'][indices]
        np.testing.assert_allclose(np.diff(times), .02, atol=2e-6, rtol=0)
        # This producer broadcasts a pre-action observation over the four ticks.
        for offset in (1, 2, 3):
            available = indices + offset < len(active)
            compare = indices[available]
            complete = active[compare + offset]
            np.testing.assert_array_equal(data['upper_observation'][compare[complete]],
                                          data['upper_observation'][compare[complete] + offset])
        valid = (np.isfinite(obs).all(axis=1) & (obs[:, 334] > .5) & (obs[:, 335] > .5)
                 & (abs(obs[:, 318]) < 5) & (abs(obs[:, 298]) < 5))
        signal = -(obs[:, 318] - obs[:, 298])
        hit = valid & (np.abs(signal) >= RULE['threshold_normalized'])
        hits = np.flatnonzero(hit)
        first = None
        if len(hits):
            k = int(hits[0]); ix = int(indices[k])
            # No actuation simulated: these are contextual descriptions only.
            first = dict(task_time_s=float(times[k]), direction=int(np.sign(signal[k])),
                         signal=float(signal[k]), logged_roll_rad=float(data['phi'][ix]),
                         logged_speed_m_s=float(data['actual_forward_speed'][ix]),
                         logged_path_error_m=float(data['path_cross_track'][ix]),
                         state_timing='physical log is post first lower tick, observation is pre-action')
        records.append(dict(source=source.name, control_decisions=len(indices),
                            valid_decisions=int(valid.sum()), unavailable_decisions=int((~valid).sum()),
                            max_abs_signal=float(np.abs(signal[valid]).max()),
                            above_threshold_decisions=int(hit.sum()), latched_activations=int(len(hits)>0),
                            first_false_trigger=first,
                            original_physical_failure=bool(data['physical_failure'][active].any()),
                            original_domain_exit=bool(data['domain_exit'][active].any()),
                            last_observed_task_time_s=float(times[-1])))
    result = dict(rule=RULE, nominal_cases=records, source_sha256=hashes,
                  tasks_with_false_trigger=sum(r['latched_activations'] for r in records),
                  above_threshold_decisions=sum(r['above_threshold_decisions'] for r in records),
                  new_physics_ticks=0, fits=0,
                  limitation='DEV causal signal replay, not closed-loop action replay. No predicted outcome '
                             'after a trigger. No independent state/noise robustness or nominal-preservation proof.')
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 完整名义任务中的ESO触发审查', '',
             '目的：检查旧捕获后统计遗漏的起步/入弯/出弯信号，优先排除需要已知起扰窗口的检测方案。',
             '开发规则在打开完整名义NPZ前写入rule_before_replay.json：最新两有效历史帧归一化ESO差负值，绝对值≥0.015首次触发，整回合只触发一次。阈值来自已看过的DEV捕获后间隔；非独立校准。',
             '帧语义核对：preference_tolerance_screen在动作前读取上层观测并广播到4底层tick；PathCommandEnv.policy_step末尾record_frame一次，间隔20ms。逐轨迹核对50Hz时间间隔、四tick广播与两帧有效掩码；没有输入起扰/捕获时刻。', '',
             '|轨迹|有效决策|超阈值决策|单次误触发|首次时刻|原域退出|', '|---|---:|---:|---:|---|---|']
    for r in records:
        first = r['first_false_trigger']
        lines.append(f"|{r['source']}|{r['valid_decisions']}|{r['above_threshold_decisions']}|{r['latched_activations']}|{None if first is None else first['task_time_s']}|{r['original_domain_exit']}|")
    lines += ['', f"共{len(records)}条轨迹，{result['tasks_with_false_trigger']}条触发，{result['above_threshold_decisions']}个决策超过阈值。0新仿真/拟合。", result['limitation'],
              '若触发，保留其方向/时刻/状态，不能推断实际干预后结局；若未触发，也只能说明这些旧轨迹没有调用该规则，不能称新状态上名义无害。']
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
