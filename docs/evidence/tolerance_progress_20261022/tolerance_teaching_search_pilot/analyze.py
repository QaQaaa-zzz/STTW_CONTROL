"""Predeclared search replay. Exhaustive acquisition costs remain charged."""
from pathlib import Path
import json, math, hashlib
import numpy as np
R=Path('runs/tolerance_teaching_search_pilot_v1').resolve()
D=json.loads((R/'declaration.json').read_text());S=json.loads((R/'evaluation/status.json').read_text())
assert S['phase']=='completed'
rows=json.loads((R/'evaluation/results.json').read_text())['cases'];assert len(rows)==120
key=lambda r:tuple(r[k] for k in ('route','speed','alpha','force_N','onset_s'))
lookup={(r['policy'],key(r)):r for r in rows};assert len(lookup)==120
spec=D['search_protocol'];records=[]
for base in rows:
 if base['policy']!='control' or base['force_N']==0:continue
 k=key(base);f=base['prefix_features'];lam=f['lambda_u'];T=min(1.,f['remaining_s'])
 W=f['b_u']**2*(-math.expm1(-2*lam*T))/(2*lam)
 K=spec['gamma']*(-math.expm1(-lam*D['duration_s']))/lam
 risk=abs(f['z_u']+K*base['force_N'])/math.sqrt(spec['effort_budget']*W)
 useful={}
 for arm in ('speed_off','steer_off'):
  a=lookup[arm,k]
  useful[arm]=bool(not base['task_recovered'] and a['task_recovered'] and not a['domain_failure'] and not a['physical_failure'] and a['logged_return']>base['logged_return'])
 records.append(dict(context=k,risk=risk,abs_force=abs(base['force_N']),abs_roll=abs(f['roll_rad']),useful=useful,either=any(useful.values()),control_task=base['task_recovered']))
assert len(records)==32
rng=np.random.default_rng(spec['random_seed']);tie=rng.permutation(32)
orders={'analytic':sorted(range(32),key=lambda i:(-records[i]['risk'],int(tie[i]))),'abs_force':sorted(range(32),key=lambda i:(-records[i]['abs_force'],int(tie[i]))),'abs_roll':sorted(range(32),key=lambda i:(-records[i]['abs_roll'],int(tie[i])))}
perms=[rng.permutation(32) for _ in range(spec['random_permutations'])]
curves=[]
for label in ('either','speed_off'):
 y=np.array([r['either'] if label=='either' else r['useful']['speed_off'] for r in records],int)
 for b in spec['query_budgets']:
  dist=np.array([y[o[:b]].sum() for o in perms]);curves.append(dict(label=label,budget=b,**{name:int(y[o[:b]].sum()) for name,o in orders.items()},uniform_mean=float(dist.mean()),uniform_p05_p95=np.quantile(dist,[.05,.95]).tolist(),uniform_exact_expectation=float(b*y.mean())))
out=dict(declaration_sha256=hashlib.sha256((R/'declaration.json').read_bytes()).hexdigest(),results_sha256=hashlib.sha256((R/'evaluation/results.json').read_bytes()).hexdigest(),records=records,orders=orders,curves=curves,actual_lower_ticks=S['active_lower_ticks'],interpretation='Retrospective fixed-order query replay; no realized acquisition savings. Randomization bands are not confidence intervals over independent states.',tie_rule='seeded random tie order shared by deterministic rankings',totals={p:sum(r['task_recovered'] for r in rows if r['policy']==p) for p in D['policies']})
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['totals','curves','actual_lower_ticks']},indent=2))
