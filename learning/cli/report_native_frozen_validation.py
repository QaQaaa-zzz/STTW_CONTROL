"""Render a complete descriptive report from the sealed 52-case analysis.

No simulation, partial-result inspection, threshold selection or model fitting.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['|' + '|'.join(map(str, row)) + '|' for row in rows])


def render(root):
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'completed' and status['completed_cases'] == 52
    review_path = root / 'analysis/review.json'
    data = json.loads(review_path.read_text())
    records, scores, comparisons = (data[k] for k in ('records', 'scores', 'comparisons'))
    assert len(records) == 52 and len(scores) == 156 and len(comparisons) == 84
    assert len({r['case_id'] for r in records}) == 52
    lookup = {(r['role'], r['route'], r['execution_alpha']): r for r in records}
    own = {(s['case_id'], s['evaluation_alpha']): s for s in scores}
    byrole = defaultdict(list)
    groups = defaultdict(list)
    for row in records:
        byrole[row['role']].append(row)
    for pair in comparisons:
        if pair['candidate'] == pair['control'] and pair['user_alpha'] == pair['control_execution_alpha']:
            continue  # Self comparisons remain in CSV, not claimed as independent evidence.
        key = pair['candidate'], pair['control'], pair['control_execution_alpha'] if pair['candidate'] == pair['control'] else 'matched'
        groups[key].append(pair)
    lines = ['# 固定52条原生功能验证：完整描述性结果', '',
             '本报告仅在完整52条执行结束后生成。全部同源完整初态，四固定路线和三个用户偏好；'
             '没有独立训练重复、final TEST、外力或新训练。数值核验与科学解释仍须结合独立审查，自动报告不替代审查。', '',
             '## 目的与条件', '',
             '检验冻结H75的原生偏好用途能否保留到75°/R2.1及110°/R2.65左右弯。'
             '原用户速度2.6m/s，α=0/.5/1，原10秒闭环与停止规则；初态bank3/seed77001，'
             'H75权重、local300、原奖励、滤波和执行权限不变。具体身份与源SHA见[plan](../plan.json)。', '',
             'H75三执行偏好跨用户重评分，提供固定执行0/.5/1对照；5Hz H75与26候选、1秒预测的固定5Hz普通参考选择比较；'
             '另有原50Hz H75、原静态平衡规则和zero。zero四条只重复评分，不计成十二条物理轨迹。'
             '预测使用完整模拟状态，计算暂停仿真时间，未来力假定零，不是实时或全局最优MPC。', '',
             '## 全部结局', '',
             '到达为原world goal一米圆首次实测进入，发生在native停止之前，不要求末速度、航向或保持。'
             '域退出与native物理停止单列；提前退出且未触地仅表示已观察前缀无触地，其后未知。', '']
    lines.append(table(['方法', '实际轨迹', '1m首达', '完整10s无native停止', '实际触地', 'native物理停止', '域停止', '未触地但提前截断'],
                       [[role, len(rows), sum(r['goal1m_before_native_stop'] for r in rows),
                         sum(r['complete_native_horizon'] for r in rows), sum(r['actual_contact'] for r in rows),
                         sum(r['native_physical_stop'] for r in rows), sum(r['native_domain_stop'] for r in rows),
                         sum(r['contact_observation'] == 'truncated_no_contact' for r in rows)]
                        for role, rows in byrole.items()]))
    lines += ['', '## 原用户费用配对', '',
              '差值=候选−对照，负值表示候选费用较低；原10秒J包含首达后的成本，'
              '停止时保留原终止替代费用。任务三通道为speed/path/primary，其余七项另报。'
              '以下都是相关的描述性配对，不作独立样本显著性或全局优势推断。', '']
    rows = []
    for key, pairs in groups.items():
        deltas = [p['cost_delta'] for p in pairs]
        gains = losses = 0
        for p in pairs:
            a = lookup[p['candidate'], p['route'], p['user_alpha']]
            b = lookup[p['control'], p['route'], p['control_execution_alpha']]
            gains += a['goal1m_before_native_stop'] and not b['goal1m_before_native_stop']
            losses += b['goal1m_before_native_stop'] and not a['goal1m_before_native_stop']
        rows.append([f'{key[0]} vs {key[1]} ({key[2]})', len(pairs),
                     sum(d < 0 for d in deltas), sum(d > 0 for d in deltas),
                     f'{min(deltas):.6f}', f'{max(deltas):.6f}', gains, losses])
    lines.append(table(['候选/对照', '非自比较数', '费用降低', '费用增加', '最小ΔJ', '最大ΔJ', '新增到达', '丢失到达'], rows))
    lines += ['', '### 每条实际轨迹的原执行偏好评分', '']
    lines.append(table(['case', '路线', '方法', '执行α', '首达s', '最小目标距离m', '原J', '任务三项', '结局观察'],
        [[r['case_id'], r['route'], r['role'], r['execution_alpha'],
          '-' if r['first_arrival_posttick_s'] is None else f"{r['first_arrival_posttick_s']:.3f}",
          '-' if r['minimum_goal_distance_m'] is None else f"{r['minimum_goal_distance_m']:.6f}",
          f"{own[r['case_id'], r['state_alpha']]['scored_cost']:.6f}",
          f"{own[r['case_id'], r['state_alpha']]['task_channels']:.6f}", r['contact_observation']]
         for r in records]))
    lines += ['', '### 全部费用反例', '',
              '逐一列出候选原J大于对照的配对；没有删掉与总体倾向相反的路线或偏好。', '']
    losses = [p for p in comparisons if p['cost_delta'] > 0]
    lines.append(table(['候选', '对照', '路线', '用户α', '对照执行α', 'ΔJ', 'Δ任务三项'],
        [[p['candidate'], p['control'], p['route'], p['user_alpha'], p['control_execution_alpha'],
          f"{p['cost_delta']:.6f}", f"{p['task_channels_delta']:.6f}"] for p in losses]))
    lines += ['', '## XY与全部诊断', '',
              '方形为实际终点，菱形为域退出，叉号为实际触地，三角为其他native物理停止。'
              '原参考绘至实际最大路径进度之后2米；轨迹全量展示，不以目标首达截掉后续运动。', '']
    for alpha in ('0', '0.5', '1'):
        lines += [f'![用户alpha={alpha}的四路线与全部方法实际XY](xy_alpha{alpha}.png)', '']
    lines += ['[全部时序与十项费用图](diagnostics/)；[52条结局](records.csv)；'
              '[156条跨评分](scores.csv)；[84条配对含12自比较](comparisons.csv)。', '',
              '## 计费与解释边界', '',
              table(['计数', 'lower ticks/槽'], [[k, data[k]] for k in
                    ('actual_lower_ticks', 'allocated_actual_slots', 'prediction_lower_ticks', 'allocated_prediction_slots')]), '',
              '每lower tick包含25个物理子步；实际执行与候选预测分账，0训练。'
              '计费不自动意味着同计算预算或部署实时性。', '',
              '这些无外力结果可以支持或收缩条件偏好功能，不能证明提高了防摔阈值、解析容扰公式有效、'
              '神经方法优于全部参考治理/MPC，或已经满足Transactions投稿证据。'
              '不按这批结果再选checkpoint、调基线或扩大场景。', '',
              f"来源review.json SHA256：`{hashlib.sha256(review_path.read_bytes()).hexdigest()}`。",
              '独立审查、人类核验和图形目视状态须由实际完成记录提供，此处不预填通过。', '']
    target = root / 'analysis/RESULTS.md'
    assert not target.exists(), 'Preserve prior generated report; choose an explicit revision'
    target.write_text('\n'.join(lines))
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    render(parser.parse_args().root.resolve())
