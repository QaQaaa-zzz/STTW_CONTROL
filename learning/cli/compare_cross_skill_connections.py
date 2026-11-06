"""Pair frozen skills on identical approaches and expose cross-selected outcomes."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--summary',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--skills',nargs=2,default=['source','rb_run1_candidate'])
a=p.parse_args()
s=json.loads(a.summary.read_text());left,right=a.skills
groups=defaultdict(dict)
for r in s['rows']:
    if r['skill'] in a.skills:
        spec=json.loads(Path(r['config']).read_text())
        disturbance=json.dumps(spec.get('external_force_pulse'),sort_keys=True)
        key=(r['origin_id'],r['distance_m'],disturbance)
        candidate=(r['speed_m_s'],r['alpha'])
        slot=groups[key].setdefault(candidate,{})
        if r['skill'] in slot:raise ValueError('Duplicate skill/candidate in paired summary')
        slot[r['skill']]=r
pairs=[];selected=[]
for (origin,distance,disturbance),candidates in sorted(groups.items()):
    complete=len(candidates)==6 and all(set(by_skill)==set(a.skills) and all(r['status']=='evaluated' for r in by_skill.values()) for by_skill in candidates.values())
    for (speed,alpha),by_skill in sorted(candidates.items()):
        observed=all(skill in by_skill and by_skill[skill]['status']=='evaluated' for skill in a.skills)
        outcome=None
        if observed:
            yl,yr=(bool(by_skill[skill]['complete']) for skill in a.skills)
            outcome='both_success' if yl and yr else 'only_first' if yl else 'only_second' if yr else 'both_failed'
        pairs.append(dict(origin_id=origin,distance_m=distance,disturbance=json.loads(disturbance),speed_m_s=speed,alpha=alpha,
            status='evaluated' if observed else 'pending',outcome=outcome,
            cases={skill:dict(case_id=r['case_id'],Y=r['complete'],T_s=r.get('success_time_s'),failure_phase=r.get('failure_phase')) for skill,r in by_skill.items()}))
    row=dict(origin_id=origin,distance_m=distance,disturbance=json.loads(disturbance),all_candidates_evaluated=complete,selected_by_skill=None)
    if complete:
        row['selected_by_skill']={}
        for skill in a.skills:
            feasible=[(c,rs[skill]) for c,rs in candidates.items() if rs[skill]['complete']]
            best=min(feasible,key=lambda item:(item[1]['success_time_s'],item[0])) if feasible else None
            chosen=None
            if best:
                candidate=best[0]
                chosen=dict(speed_m_s=candidate[0],alpha=candidate[1],
                    executed_outcomes={target:dict(Y=candidates[candidate][target]['complete'],T_s=candidates[candidate][target]['success_time_s'],case_id=candidates[candidate][target]['case_id']) for target in a.skills})
            row['selected_by_skill'][skill]=dict(status='candidate_selected' if best else 'no_successful_candidate',choice=chosen)
    selected.append(row)
result=dict(role='Paired DEV skill/approach outcomes; selection uses observed outcomes and is hindsight, not online or independent TEST',
    skills=a.skills,declared_candidate_pairs=len(pairs),counts=dict(Counter(r['outcome'] if r['status']=='evaluated' else 'pending' for r in pairs)),
    pairs=pairs,cross_selected_approaches=selected,
    limits='Source and RB share adaptation/interface; update2 is excluded from this two-skill claim. Entire candidate matrices must finish before cross-selection. No successful candidate is reported explicitly, never substituted with a survivor. Disturbances and origin families must be separately paired.')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['counts']))
