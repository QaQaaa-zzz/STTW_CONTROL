"""Audit existing original-policy traces; no detector fit or delayed rescue claim."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.worktree.resolve()
    panels = {
        'discovery_bank3': root / 'runs/reference_prefix_return_feasibility_v1/evaluation',
        **{f'confirmation_bank{b}': root / f'runs/reference_prefix_return_confirmation_v1_attempt2/bank{b}/evaluation'
           for b in (0, 7)},
    }
    hashes, records, nominal = {}, [], []
    for panel, directory in panels.items():
        assert json.loads((directory / 'status.json').read_text())['phase'] == 'completed'
        manifest = json.loads((directory / 'manifest.json').read_text())
        config = manifest['config']
        assert config['network']['history_frames'] == 16
        assert config['network']['frame_fields'] == 20
        limits = config['residual_interface']
        for metadata in ('manifest.json', 'results.json', 'status.json'):
            path = directory / metadata
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = json.loads((directory / 'results.json').read_text())['cases']
        for row in rows:
            if row['candidate'] != 'continue_original':
                continue
            path = directory / (row['name'] + '.npz')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as archive:
                data = {key: archive[key] for key in archive.files}
            assert len(data['time']) == row['active_ticks']
            ix = np.arange(0, len(data['time']), 4)
            obs = data['upper_observation'][ix]
            # Sixteen 20-field frames, oldest first; then sixteen validity masks.
            valid = (data['upper_active'][ix].astype(bool)
                     & np.isfinite(obs).all(axis=1)
                     & (obs[:, 334] > .5) & (obs[:, 335] > .5)
                     & (np.abs(obs[:, 298]) < 5) & (np.abs(obs[:, 318]) < 5))
            # ESO equilibrium shift is -disturbance/a4. Keep normalized units:
            # no gamma, force calibration, threshold, or fitted model required.
            innovation = -(obs[:, 318] - obs[:, 298])
            if row['force_N'] == 0:
                nominal.append(dict(panel=panel, state_id=row['state_id'],
                                    valid_control_samples=int(valid.sum()),
                                    max_abs_normalized_innovation=float(np.max(np.abs(innovation[valid]))),
                                    p99_abs_normalized_innovation=float(np.quantile(np.abs(innovation[valid]), .99))))
            measurements = []
            for delay in (.02, .04, .06, .1):
                block = round(delay / .02)
                k = block * 4
                if k >= len(data['time']) or not valid[block]:
                    measurements.append(dict(delay_s=delay, available=False))
                    continue
                assert abs(float(data['relative_time'][k]) - delay) < 2e-6
                # Observation at block start is pre-action; physical logs are
                # post-integration values labelled with tick start time.
                pre = k - 1
                ref = data['governed'][pre]
                speed_bounds = limits['speed_reference_bounds_m_s']
                sb = limits['steer_reference_bound_rad']
                measurements.append(dict(
                    delay_s=delay, available=True,
                    normalized_eso_innovation=float(innovation[block]),
                    direction_matches_force=None if row['force_N'] == 0 else bool(innovation[block] * row['force_N'] > 0),
                    physical_contact_before_decision=bool(data['contact_failure'][:k].any()),
                    roll_at_decision_rad=float(data['phi'][pre]),
                    roll_rate_at_decision_rad_s=float(data['phi_dot'][pre]),
                    actual_steer_at_decision_rad=float(data['actual_delta'][pre]),
                    speed_reference_headroom_down_up_m_s=[float(ref[0] - speed_bounds[0]), float(speed_bounds[1] - ref[0])],
                    steer_reference_headroom_negative_positive_rad=[float(ref[1] + sb), float(sb - ref[1])],
                    reference_clip_channels_previous_tick=data['reference_clip_channels'][pre].astype(bool).tolist(),
                    correction_rate_clipped_previous_tick=bool(data['correction_rate_clipped'][pre]),
                    time_until_observed_contact_s=None if row['event_time_s'] is None else float(row['event_time_s'] - delay),
                ))
            records.append(dict(panel=panel, state_id=row['state_id'], force_N=row['force_N'],
                                event_contact=row['event_contact'], event_time_s=row['event_time_s'],
                                joint_success=row['joint_success'], measurements=measurements))
    assert len(records) == 27 and len(nominal) == 9
    summaries = []
    for panel in panels:
        for delay in (.02, .04, .06, .1):
            selected = [m for r in records if r['panel'] == panel and r['force_N'] != 0
                        for m in r['measurements'] if m['delay_s'] == delay]
            available = [m for m in selected if m['available']]
            summaries.append(dict(panel=panel, delay_s=delay, declared=len(selected),
                                  available=len(available), direction_matches=sum(m['direction_matches_force'] for m in available),
                                  min_abs_normalized_innovation=min(abs(m['normalized_eso_innovation']) for m in available),
                                  contact_before_decision=sum(m['physical_contact_before_decision'] for m in available)))
    result = dict(records=records, nominal_full_valid_suffix=nominal, summaries=summaries,
                  source_sha256=hashes, new_physics_ticks=0, model_fits=0,
                  limits='Known-onset alignment is analysis only; no online detector or threshold adopted. '
                  'Original-policy traces only. Headroom is geometric reference headroom, not reachable safe authority. '
                  'No delayed intervention was executed. Shared-ancestor snapshots, not independent seeds. '
                  'Nominal records begin at midturn capture, not full task start; no false-alarm guarantee.')
    args.output.mkdir(exist_ok=False)
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 因果响应时间窗审计', '',
             '目的：在原策略未被候选干预改变的轨迹中，核对既有ESO历史何时包含力方向信息，以及当时是否已触地。',
             '方法：复用发现bank3和确认bank0/7各9条原策略轨迹；只读27条NPZ。每20ms决策前观测取最新两帧ESO归一化字段差的负值，不使用gamma或能量预算。时序来自真实日志，物理状态用决策前最后完成tick。', '',
             '|面板|延迟秒|可用/声明|方向一致|决策前触地|', '|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"|{s['panel']}|{s['delay_s']}|{s['available']}/{s['declared']}|{s['direction_matches']}|{s['contact_before_decision']}|")
    lines += ['', '0新物理步、0拟合、0阈值选择。全部来源SHA与逐例转角、侧倾、角速度、参考剩余幅度、首次触地时间在analysis.json。',
              '解释边界：信号方向可辨、尚未触地、参考尚有幅度空间，都不能推出延迟干预可救回。在线触发还需处理未知起扰时刻、完整名义轨迹与控制动作诱发的ESO变化。没有用已知力方向挑动作，没有新增训练或修改lookup。']
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
