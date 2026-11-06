"""Describe matched candidate coverage; oracle labels are never online performance."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--summary', type=Path, help='Freeze comparison to evaluated rows in an existing collection summary')
p.add_argument('--anchor-speed', type=float, default=2.0)
p.add_argument('--anchor-alpha', type=float, default=0.5)
a = p.parse_args()
m = json.loads(a.manifest.read_text())
frozen = {r['case_id']: r for r in json.loads(a.summary.read_text())['rows'] if r['status'] == 'evaluated'} if a.summary else None
sequences = any('second_candidate' in c for c in m['cases'])
groups = defaultdict(list)
for case in m['cases']:
    config = json.loads(Path(case['config']).read_text())
    result_path = Path(config['output']) / 'chain_summary.json'
    result = frozen.get(case['case_id']) if frozen is not None else (json.loads(result_path.read_text()) if result_path.exists() else None)
    candidate = case['second_candidate'] if sequences else case
    first = case['first_candidate'] if sequences else candidate
    groups[(config['origin_id'], case['distance_m'])].append(dict(
        case_id=case['case_id'], speed=candidate['speed_m_s'], alpha=candidate['alpha'],
        first_speed=first['speed_m_s'], first_alpha=first['alpha'], result=result))

candidates = sorted({(c['speed'], c['alpha']) for rows in groups.values() for c in rows})
methods = {f'fixed_v{v}_a{alpha}': lambda c, v=v, alpha=alpha: (c['speed'], c['alpha']) == (v, alpha)
           for v, alpha in candidates}
if sequences:
    # Fixed policies are diagonal holds, not a hindsight choice of first stage.
    methods = {f'fixed_v{v}_a{alpha}': lambda c, v=v, alpha=alpha:
               (c['first_speed'], c['first_alpha'], c['speed'], c['alpha']) == (v, alpha, v, alpha)
               for v, alpha in candidates}
    methods.update(hold6_oracle=lambda c: (c['first_speed'], c['first_alpha']) == (c['speed'], c['alpha']),
                   sequence36_oracle=lambda c: True)
else:
    methods.update(speed_only_oracle=lambda c: c['alpha'] == a.anchor_alpha,
               alpha_only_oracle=lambda c: c['speed'] == a.anchor_speed,
               joint_oracle=lambda c: True)
selections = {name: {} for name in methods}
records = []
for (origin, distance), rows in groups.items():
    expected = len(candidates) ** (2 if sequences else 1)
    complete = len(rows) == expected and all(c['result'] is not None for c in rows)
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
    if sequences:
        record['candidate_order'] = [dict(speed_m_s=v, alpha=alpha) for v, alpha in candidates]
        indexed = {(c['first_speed'], c['first_alpha'], c['speed'], c['alpha']): c for c in rows}
        record['matrix'] = [[dict(
            case_id=indexed[(v1, a1, v2, a2)]['case_id'],
            Y=int(indexed[(v1, a1, v2, a2)]['result']['complete'])
              if indexed[(v1, a1, v2, a2)]['result'] is not None else None,
            T_s=indexed[(v1, a1, v2, a2)]['result']['success_time_s']
              if indexed[(v1, a1, v2, a2)]['result'] is not None else None)
            for v2, a2 in candidates] for v1, a1 in candidates]
        record['matrix_axes'] = 'Rows: first candidate; columns: second candidate; null is pending.'
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
if sequences:
    hold = selections['hold6_oracle']
    seq = selections['sequence36_oracle']
    common = [k for k in seq if seq[k]['Y'] and hold[k]['Y']]
    scores['sequence36_oracle']['paired_with_hold6'] = dict(
        complete_root_tasks=len(seq), common_success_tasks=len(common),
        mean_time_saved_s=sum(hold[k]['T_s']-seq[k]['T_s'] for k in common)/len(common) if common else None,
        gained_success_tasks=sum(seq[k]['Y'] and not hold[k]['Y'] for k in seq),
        lost_success_tasks=sum(hold[k]['Y'] and not seq[k]['Y'] for k in seq))
output = dict(role=m['role'], interpretation='Descriptive DEV coverage and hindsight oracle upper bounds; not learned or deployed selection. Distances share upstream roots. Incomplete groups remain pending, not failures.',
              collection_summary=str(a.summary) if a.summary else None,
              declared_origins=len({r for r, _ in groups}), declared_tasks=len(groups),
              complete_matched_tasks=sum(r['matched_group_complete'] for r in records),
              anchor=dict(speed=a.anchor_speed, alpha=a.anchor_alpha), scores=scores, groups=records)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({name: dict(successes=s['successes'], tasks=s['evaluated_tasks']) for name, s in scores.items()}))
