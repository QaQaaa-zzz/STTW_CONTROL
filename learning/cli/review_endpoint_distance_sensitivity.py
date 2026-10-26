"""Read-only distance sensitivity of the authenticated historical488 review."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def review(root):
    root = Path(root).resolve()
    source = root / 'results.json'
    rows = json.loads(source.read_text())
    validation = json.loads((root / 'calculation_validation.json').read_text())
    if validation['complete_unique_trajectories'] != 488 or len(rows) != 1952:
        raise ValueError('complete historical488 review required')
    cases = defaultdict(list)
    for row in rows:
        cases[row['case_id']].append(row)
    if len(cases) != 488:
        raise ValueError('do not count radius queries as independent trajectories')
    unique = []
    for cid, records in cases.items():
        if sorted(r['goal_radius_m'] for r in records) != [.1, .2, .3, .5]:
            raise ValueError(f'incomplete radius series: {cid}')
        distances = [r['minimum_upright_goal_distance_m'] for r in records]
        if any(d is None or not np.isfinite(d) for d in distances) or len(set(distances)) != 1:
            raise ValueError(f'inconsistent pre-fatal distance: {cid}')
        for r in records:
            if r['arrival_outcome'] == 'unknown' or r['goal_arrived'] != (distances[0] <= r['goal_radius_m']):
                raise ValueError(f'old label not reproduced: {cid}')
        unique.append(records[-1])
    groups = defaultdict(list)
    for r in unique:
        groups[(r['panel'], r['role'], bool(r['force_N']))].append(r)
    summaries = []
    for (panel, role, disturbed), rr in sorted(groups.items()):
        dist = np.array([r['minimum_upright_goal_distance_m'] for r in rr])
        alive = np.array([r['physical_outcome'] == 'survived' for r in rr])
        s = dict(panel=panel, role=role, disturbed=disturbed, n=len(rr),
                 old_strict_success=sum(r['old_strict_task'] is True for r in rr),
                 full_window_survived=int(alive.sum()),
                 minimum_distance_median_m=float(np.median(dist)),
                 minimum_distance_p90_m=float(np.quantile(dist, .9)))
        for radius in [.5, .75, 1.]:
            hit = dist <= radius
            s[f'within_{radius:g}m_before_fatal'] = int(hit.sum())
            s[f'within_{radius:g}m_and_full_window_survived'] = int((hit & alive).sum())
        s['miss_0p5_but_within_1m'] = int(((dist > .5) & (dist <= 1.)).sum())
        summaries.append(s)
    out = root / 'distance_sensitivity_v1'
    out.mkdir(exist_ok=False)
    with (out / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    report = ['# STTW 旧方案：放宽目标距离的复查', '',
        '目的：区分严格末段失败、接近目标、真正未回来以及摔倒。复用已认证488条轨迹的首次触地/故障前最小目标距离，0新物理、0训练。', '',
        '条件：原10秒、原世界坐标目标、原控制器与扰动。0.5米保留；0.75/1米是本次事后敏感性分析，不作独立测试或自动替代主判据。不要求末速、航向或保持。整窗存活单列；旧strict与点圆并非嵌套，旧成功不撤销。', '',
        '|面板|方案|受扰|条数|旧strict|0.5m|0.75m|1m|整窗存活|最近距离中位/m|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        report.append(f"|{s['panel']}|{s['role']}|{s['disturbed']}|{s['n']}|{s['old_strict_success']}|{s['within_0.5m_before_fatal']}|{s['within_0.75m_before_fatal']}|{s['within_1m_before_fatal']}|{s['full_window_survived']}|{s['minimum_distance_median_m']:.3f}|")
    report += ['', '表内各半径是首次触地/故障前命中，不能由命中推断整窗不摔；联合计数在summary.csv。确定性、动作噪声及名义面板分别统计，不混池；同快照alpha/噪声不当独立训练重复。zero只计8条真实轨迹，不复制。', '',
        '## 路径图与证据', '',
        '沿用上级目录原12张xy_*.png（实际/原参考、真实终点），本次不改轨迹、不生成伪造的首次命中时刻或held-tail恢复标签。原raw认证见../receipt.json；本次从派生最小距离重计，不冒称再次重放原始仿真。', '',
        '## 去留原则', '',
        '放松后能到达的旧候选保留为实用候选；收益需与普通续训、速度下限规则并列。半径扩大自然提高命中数，不能据此推出新机制改善。只有在既定任务下发现具体可利用增量，才决定后续必要验证；不因新阈值自动重启旧训练。']
    (out / 'README.md').write_text('\n'.join(report)+'\n')
    receipt = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        trajectories=488, queries_in_source=1952, unique_zero=sum(r['role']=='zero_middle' for r in unique),
        existing_radius_labels_reproduced=1952, new_physics_steps=0, optimizer_updates=0,
        scope='Read-only posthoc radius sensitivity; no new first-arrival timing or controller selection.',
        summaries=summaries)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); args = parser.parse_args()
    print(review(args.root))
