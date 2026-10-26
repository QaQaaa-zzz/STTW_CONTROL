"""Read audited STTW endpoints at fixed, shared goal radii; no simulation."""
import hashlib
import json
from pathlib import Path

RADII = (0.5, 0.75, 1.0)


def review(worktree):
    root = Path(worktree)
    sources = {
        'whole_episode': root/'runs/h75_disturbed_common_interface_v1/analysis_reconstructed_v2/results.json',
        'same_state_pause': root/'runs/same_state_reference_pause_v1/analysis_partial_block_fix_v1/results.json',
    }
    output = {'scope': 'Posthoc DEV sensitivity on already audited complete trajectories; original world goal and 10s window, before contact/fault. No speed/heading/hold veto. No new physics/training; no independent repeats; datasets overlap and must not be pooled.',
              'radii_m': RADII, 'sources': {}, 'panels': {}}
    for name, path in sources.items():
        rows = json.loads(path.read_text())
        assert len(rows) == 360 and len({r['case_id'] for r in rows}) == 360
        assert all(r['goal_arrived'] is not None and r['minimum_upright_goal_distance_m'] is not None for r in rows)
        assert all((r['minimum_upright_goal_distance_m'] <= .5) == r['goal_arrived'] for r in rows)
        output['sources'][name] = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n': len(rows), 'verification': 'audited per-case summary read; original 0.5m labels reproduced exactly; raw physics audit inherited, not re-executed'}
        panel = {}
        roles = sorted({r['role'] for r in rows})
        for split in ('nominal', 'disturbed'):
            rr = [r for r in rows if bool(r['force_N']) == (split == 'disturbed')]
            groups = {role: {(r['context'], r['alpha'], r['event_id']): r for r in rr if r['role'] == role} for role in roles}
            assert len({frozenset(g) for g in groups.values()}) == 1
            baseline = 'goal_full_feedback' if name == 'whole_episode' else 'full_current'
            counts, pairs = {}, {}
            for role, bank in groups.items():
                values = list(bank.values())
                counts[role] = {'n': len(values), 'survived': sum(r['physical_outcome'] == 'survived' for r in values),
                    'arrivals': {str(radius): sum(r['minimum_upright_goal_distance_m'] <= radius for r in values) for radius in RADII},
                    'remaining_at_1m': {label: sum(lo < r['minimum_upright_goal_distance_m'] <= hi for r in values) for label, lo, hi in [('1_to_2m', 1, 2), ('2_to_4m', 2, 4), ('over_4m', 4, float('inf'))]}}
                if role != baseline:
                    pairs[role] = {str(radius): {'gain': sum(r['minimum_upright_goal_distance_m'] <= radius < groups[baseline][k]['minimum_upright_goal_distance_m'] for k, r in bank.items()), 'loss': sum(groups[baseline][k]['minimum_upright_goal_distance_m'] <= radius < r['minimum_upright_goal_distance_m'] for k, r in bank.items())} for radius in RADII}
            panel[split] = {'counts': counts, 'paired_vs_current': pairs}
        output['panels'][name] = panel
    return output


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    out = root/'runs/same_state_reference_pause_v1/relaxed_goal_review_v1'
    out.mkdir(exist_ok=False)
    result = review(root)
    (out/'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 放宽目标到达标准后的旧方案复查', '', result['scope'], '', '目的：检验旧负结果是目标范围过严，还是确实缺乏任务/防摔增量。方法：读取两个完整已审计面板，每例使用第一次触地/故障前的最小原目标距离；同一方案使用相同0.5/0.75/1m半径。0.75和1m为事后敏感性，不覆盖旧标签；两个面板有重叠，不能合计为720独立试验。', '', '|面板|条件|方案|例数|0.5m到达|0.75m到达|1m到达|整窗存活|', '|---|---|---|---:|---:|---:|---:|---:|']
    for name, panel in result['panels'].items():
        for split, data in panel.items():
            for role, c in data['counts'].items():
                a=c['arrivals'];lines.append(f"|{name}|{split}|{role}|{c['n']}|{a['0.5']}|{a['0.75']}|{a['1.0']}|{c['survived']}|")
    lines += ['', '## 原始路径证据', '', '复用已覆盖全部场景/alpha/力档的真实XY图，不改变路径或目标位置。', '', '[整回合三模型全部XY与审计](../../h75_disturbed_common_interface_v1/analysis_reconstructed_v2/README.md)', '', '[同状态三分支全部XY与审计](../analysis_partial_block_fix_v1/README.md)', '', '## 解释边界', '', '到达与全窗不摔倒分开报告。更宽半径下新增到达不代表更强抗扰；反之，没有新增到达的策略也可以有独立的存活收益。放宽任务判据不能取消真实触地或把未知记成功。没有修改原用户alpha、动力学、策略或奖励。后续方案在统一任务半径下配对比较，允许失败，以净提升与任务代价裁决，不要求全胜。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['panels'], ensure_ascii=False))
