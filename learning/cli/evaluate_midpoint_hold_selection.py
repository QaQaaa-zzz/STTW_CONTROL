"""Score the actual online midpoint selector against stored complete branches."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from sttw_control.jump_connection import JumpConnectionSelector

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--records', type=Path, required=True)
p.add_argument('--config', type=Path, default=Path('learning/configs/jump_connection_capability_dev.json'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
records = [json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
config = json.loads(a.config.read_text())
groups = defaultdict(list)
for row in records:
    key = (row['origin_id'], row['map_sha256'], json.dumps(row['parent_candidate'], sort_keys=True),
           json.dumps(row['disturbance_protocol'], sort_keys=True))
    groups[key].append(row)
rows = []
fixed = defaultdict(lambda: dict(complete=0, groups=0))
for key, group in groups.items():
    available = {(r['candidate']['speed_m_s'], r['candidate']['alpha']): r for r in group}
    expected = {(c['speed_m_s'], c['alpha']) for c in config['candidates']}
    if len(group) != len(expected) or set(available) != expected:
        raise ValueError('Complete unique candidate groups required for paired midpoint scoring')
    first = group[0]
    selector = JumpConnectionSelector(records, config, origin_id=first['origin_id'],
        **{k: first[k] for k in ('policy_sha256', 'model_contract_sha256', 'physics_dt', 'zero_lower')})
    decision = selector.choose(first['state'], 'midpoint', first['parent_candidate'])
    chosen = decision['candidate']
    result = available[chosen['speed_m_s'], chosen['alpha']]
    held = available[first['parent_candidate']['speed_m_s'], first['parent_candidate']['alpha']]
    rows.append(dict(origin_id=first['origin_id'], parent_candidate=first['parent_candidate'],
        decision=decision, selected_case_id=result['case_id'], selected_complete=result['Y'],
        selected_time_after_midpoint_s=result['T_after_decision_s'], hold_complete=held['Y'],
        oracle_complete=any(r['Y'] for r in group)))
    for candidate, r in available.items():
        name=f'v{candidate[0]}_a{candidate[1]}'
        fixed[name]['groups'] += 1
        fixed[name]['complete'] += r['Y']
report = dict(role='Stored branch evaluation using deployed selector and retain-current fallback; not online rollout or independent TEST',
    independent_origins=len({r['origin_id'] for r in rows}), midpoint_groups=len(rows),
    selected_complete=sum(r['selected_complete'] for r in rows),
    hold_complete=sum(r['hold_complete'] for r in rows),
    oracle_complete=sum(r['oracle_complete'] for r in rows), fixed_candidates=dict(fixed), rows=rows,
    limits='Six parent references create correlated reachable midpoint states per upstream root. Whole queried root excluded from evidence. All branch outcomes are observed, including fallback; no interpolation or fitted parameters.')
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}, indent=2))
