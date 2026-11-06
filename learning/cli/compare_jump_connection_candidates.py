"""Describe matched candidate coverage; oracle labels are never online performance."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--anchor-speed', type=float, default=2.0)
p.add_argument('--anchor-alpha', type=float, default=0.5)
a = p.parse_args()
m = json.loads(a.manifest.read_text())
groups = defaultdict(list)
for case in m['cases']:
    config = json.loads(Path(case['config']).read_text())
    result_path = Path(config['output']) / 'chain_summary.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    groups[(config['origin_id'], case['distance_m'])].append(dict(
        case_id=case['case_id'], speed=case['speed_m_s'], alpha=case['alpha'], result=result))

candidates = sorted({(c['speed'], c['alpha']) for rows in groups.values() for c in rows})
methods = {f'fixed_v{v}_a{alpha}': lambda c, v=v, alpha=alpha: (c['speed'], c['alpha']) == (v, alpha)
           for v, alpha in candidates}
methods.update(speed_only_oracle=lambda c: c['alpha'] == a.anchor_alpha,
               alpha_only_oracle=lambda c: c['speed'] == a.anchor_speed,
               joint_oracle=lambda c: True)
selections = {name: {} for name in methods}
records = []
for (origin, distance), rows in groups.items():
    complete = len(rows) == len(candidates) and all(c['result'] is not None for c in rows)
    record = dict(origin_id=origin, distance_m=distance, declared_candidates=len(rows),
                  evaluated_candidates=sum(c['result'] is not None for c in rows),
                  matched_group_complete=complete, selected={})
    if complete:
        for name, allowed in methods.items():
            successes = [c for c in rows if allowed(c) and c['result']['complete']]
            chosen = min(successes, key=lambda c: (c['result']['success_time_s'], c['case_id'])) if successes else None
            value = dict(Y=int(chosen is not None), T_s=chosen['result']['success_time_s'] if chosen else None,
                         case_id=chosen['case_id'] if chosen else None)
            record['selected'][name] = value
            selections[name][(origin, distance)] = value
    records.append(record)

scores = {}
for name, chosen in selections.items():
    times = [v['T_s'] for v in chosen.values() if v['Y']]
    by_root = {root: dict(successes=sum(v['Y'] for (r, _), v in chosen.items() if r == root),
                         evaluated_tasks=sum(r == root for r, _ in chosen))
               for root in sorted({r for r, _ in chosen})}
    scores[name] = dict(successes=len(times), evaluated_tasks=len(chosen),
                        success_conditional_mean_T_s=sum(times)/len(times) if times else None,
                        per_root=by_root)
anchor_name = f'fixed_v{a.anchor_speed}_a{a.anchor_alpha}'
anchor = selections[anchor_name]
for name, chosen in selections.items():
    common = [k for k in chosen if chosen[k]['Y'] and anchor[k]['Y']]
    scores[name]['paired_with_anchor'] = dict(
        anchor=anchor_name, common_success_tasks=len(common),
        mean_time_saved_s=sum(anchor[k]['T_s']-chosen[k]['T_s'] for k in common)/len(common) if common else None,
        gained_success_tasks=sum(chosen[k]['Y'] and not anchor[k]['Y'] for k in chosen),
        lost_success_tasks=sum(anchor[k]['Y'] and not chosen[k]['Y'] for k in chosen))
output = dict(role=m['role'], interpretation='Descriptive DEV coverage and hindsight oracle upper bounds; not learned or deployed selection. Distances share upstream roots. Incomplete groups remain pending, not failures.',
              declared_origins=len({r for r, _ in groups}), declared_tasks=len(groups),
              complete_matched_tasks=sum(r['matched_group_complete'] for r in records),
              anchor=dict(speed=a.anchor_speed, alpha=a.anchor_alpha), scores=scores, groups=records)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({name: dict(successes=s['successes'], tasks=s['evaluated_tasks']) for name, s in scores.items()}))
