"""CPU-only descriptive accounting after the fixed-final report completes.

Independent of the pinned runtime. No checkpoint selection, threshold tuning,
trajectory simulation, statistical independence claim, or automatic adoption.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(rows):
    groups = defaultdict(list)
    contexts = set()
    for row in rows:
        key = tuple(row[k] for k in
                    ('panel', 'policy', 'route', 'speed', 'alpha', 'force_N', 'onset_s'))
        if key in contexts:
            raise ValueError('Duplicate evaluation context')
        contexts.add(key)
        if row['balance_label'] not in ('FLOOR_CONTACT', 'COMPLETE_NO_CONTACT', 'UNKNOWN'):
            raise ValueError('Unknown physical outcome label')
        if row['force_N'] != 0:
            groups[(row['panel'], row['policy'], row['alpha'])].append(row)
    result = []
    for (panel, policy, alpha), group in sorted(groups.items()):
        counts = Counter(r['balance_label'] for r in group)
        n = len(group)
        safe, unknown = counts['COMPLETE_NO_CONTACT'], counts['UNKNOWN']
        result.append(dict(panel=panel, policy=policy, alpha=alpha, n=n,
                           contact=counts['FLOOR_CONTACT'], complete_no_contact=safe,
                           unknown=unknown,
                           reached_pulse=sum(r.get('prefix_features') is not None for r in group),
                           task_success=sum(bool(r['task_recovered']) for r in group),
                           observed_nonfall_lower=safe/n,
                           missing_outcome_upper=(safe+unknown)/n))
    return result


def directional_grid_summary(rows):
    """Observed signed grids, conditional on the evaluator's full-state branch.

    Matching observed features is a consistency check, not proof of identical
    hidden state. That provenance comes from the pinned evaluator and snapshots.
    A grid bracket implies a threshold interval only under an additional
    monotonicity assumption; untested forces remain untested.
    """
    fields = ('panel', 'policy', 'route', 'speed', 'alpha', 'onset_s')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        zeros = [r for r in group if r['force_N'] == 0]
        if len(zeros) != 1:
            raise ValueError('Each force grid requires one zero-force branch')
        if len({r['force_N'] for r in group}) != len(group):
            raise ValueError('Duplicate signed force')
        zero = zeros[0]
        for direction in (-1, 1):
            ordered = sorted((r for r in group if direction*r['force_N'] > 0),
                             key=lambda r: abs(r['force_N']))
            if not ordered:
                continue
            observed = [zero]+ordered
            valid = all(r.get('prefix_features') is not None for r in observed)
            if valid and any(r['prefix_features'] != zero['prefix_features'] for r in ordered):
                raise ValueError('Within-policy force branches have different prefix features')
            safe = [abs(r['force_N']) for r in observed
                    if r['balance_label'] == 'COMPLETE_NO_CONTACT']
            contact = [abs(r['force_N']) for r in observed
                       if r['balance_label'] == 'FLOOR_CONTACT']
            if not valid:
                status = 'PREFIX_NOT_REACHED'
            elif zero['balance_label'] != 'COMPLETE_NO_CONTACT':
                status = 'ZERO_FORCE_NOT_COMPLETE_SAFE'
            elif safe and contact and max(safe) > min(contact):
                status = 'NONMONOTONIC_GRID'
            elif any(r['balance_label'] == 'UNKNOWN' for r in observed):
                status = 'UNKNOWN_IN_GRID'
            elif contact:
                status = 'GRID_BRACKET'
            else:
                status = 'RIGHT_CENSORED_GRID'
            result.append(dict(zip(fields, key), direction=direction, status=status,
                               largest_observed_safe_N=max(safe) if safe else None,
                               smallest_observed_contact_N=min(contact) if contact else None,
                               observations=[dict(force_N=r['force_N'],
                                                  outcome=r['balance_label']) for r in observed]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--post', required=True)
    args = parser.parse_args()
    root = Path(args.post)
    status = json.loads((root/'status.json').read_text())
    if status['phase'] != 'completed':
        raise ValueError('Wait for the full fixed-final evaluation and report')
    source = root/'analysis.json'
    data = json.loads(source.read_text())
    plan = json.loads((root/'evaluation_plan.json').read_text())
    if len(data['cases']) != plan['total_new_cases']:
        raise ValueError('Incomplete case coverage')
    summary = summarize(data['cases'])
    output = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  physical=summary, paired=data['paired'],
                  directional_grids=directional_grid_summary(data['cases']),
                  scope='Single-seed DEV; own-policy prefixes; bounds reflect missing '
                        'physical outcomes, not confidence intervals; no automatic adoption')
    target = root/'PHYSICAL_DECISION.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 固定末端物理结果核对', '',
             '全部受扰原始分母保留。下界=完整无触地/N，上界=(完整无触地+UNKNOWN)/N；'
             '这是缺失结局范围，不是置信区间。跨策略匹配任务条件，各自前缀状态可不同。', '',
             '|面板|模型|α|N|已起扰|触地|完整无触地|UNKNOWN|不摔倒比例范围|任务成功|',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for r in summary:
        lines.append(f"|{r['panel']}|{r['policy']}|{r['alpha']:g}|{r['n']}|"
                     f"{r['reached_pulse']}|{r['contact']}|{r['complete_no_contact']}|"
                     f"{r['unknown']}|{r['observed_nonfall_lower']:.1%}–"
                     f"{r['missing_outcome_upper']:.1%}|{r['task_success']}|")
    lines += ['', '配对新增/丢失完整无触地和接触→UNKNOWN见同名JSON的paired。'
              '两边都到达起扰的配对诊断另有分母，不能替代上表原始分母。'
              '高速零力面板不进入受扰表，偏好保留须结合原REPORT与全部五α轨迹检查。'
              '不从该单种子表直接宣称新颖性、统计显著或采用新模型。', '',
              'JSON的directional_grids分别保留正负力档。GRID_BRACKET仅表示已测网格中'
              '先通过后触地；只有额外假设结局随力幅单调，才能解释为阈值范围。'
              'RIGHT_CENSORED_GRID表示已测上限仍通过，不代表精确极限。'
              '非单调、UNKNOWN、未到起扰或零力不安全单列，不能硬拟合单阈值。'
              '同前缀特征一致只作数据核对，完整同状态来源仍须查固定评测代码及快照。']
    (root/'PHYSICAL_DECISION.md').write_text('\n'.join(lines)+'\n')
    print(target)


if __name__ == '__main__':
    main()
