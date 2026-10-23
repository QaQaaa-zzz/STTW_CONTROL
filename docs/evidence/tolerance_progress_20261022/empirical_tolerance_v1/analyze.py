"""Finite sampled directional recovery curves; never certify between samples."""
from pathlib import Path
import json,hashlib
R=Path('runs/preference_empirical_tolerance_v1').resolve()
D=json.loads((R/'declaration.json').read_text());S=json.loads((R/'evaluation/status.json').read_text());assert S['phase']=='completed'
rows=json.loads((R/'evaluation/results.json').read_text())['cases'];assert len(rows)==144
key=lambda r:tuple(r[k] for k in ['route','speed','alpha','force_N','onset_s'])
base={key(r):r for r in rows if r['policy']=='parent'};pairs=[]
for r in rows:
 if r['policy']=='parent':continue
 b=base[key(r)];pairs.append(dict(context=key(r),parent=b['task_recovered'],candidate=r['task_recovered'],gain=bool(r['task_recovered'] and not b['task_recovered']),loss=bool(b['task_recovered'] and not r['task_recovered']),parent_roll=b['peak_roll_rad'],candidate_roll=r['peak_roll_rad'],parent_tail_path=b['tail_max_path_m'],candidate_tail_path=r['tail_max_path_m']))
curves=[]
for pol in D['policies']:
 for route in D['routes']:
  for alpha in D['alphas']:
   group=[r for r in rows if r['policy']==pol and r['route']==route['name'] and r['alpha']==alpha];zero=next(r for r in group if r['force_N']==0)
   for sign in [-1,1]:
    ordered=sorted([r for r in group if r['force_N']*sign>0],key=lambda r:abs(r['force_N']));failed=[i for i,r in enumerate(ordered) if not r['task_recovered']];first=failed[0] if failed else None
    curves.append(dict(policy=pol,route=route['name'],alpha=alpha,sign=sign,nominal_task=zero['task_recovered'],samples=[dict(force_N=r['force_N'],task=r['task_recovered'],working=r['performance_success'],domain_exit=r['domain_failure'],peak_roll=r['peak_roll_rad'],tail_path=r['tail_max_path_m']) for r in ordered],first_failed_sample_N=None if first is None else abs(ordered[first]['force_N']),sampled_all_pass_through_N=None if not zero['task_recovered'] else (abs(ordered[-1]['force_N']) if first is None else (0 if first==0 else abs(ordered[first-1]['force_N']))),upper_censored=bool(zero['task_recovered'] and first is None),recovery_after_failed_sample=bool(first is not None and any(r['task_recovered'] for r in ordered[first+1:]))))
out=dict(source_sha256=hashlib.sha256((R/'evaluation/results.json').read_bytes()).hexdigest(),status=S,pairs=pairs,curves=curves,totals={p:dict(task=sum(r['task_recovered'] for r in rows if r['policy']==p),n=sum(r['policy']==p for r in rows),domain=sum(r['domain_failure'] for r in rows if r['policy']==p)) for p in D['policies']},interpretation='Own-policy prefixes; discrete operational recovery, not common-onset causal comparison or continuous safety guarantee.')
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(totals=out['totals'],gains=sum(p['gain'] for p in pairs),losses=sum(p['loss'] for p in pairs),nonmonotonic=sum(c['recovery_after_failed_sample'] for c in curves)),indent=2))
