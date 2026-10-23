"""Evaluate frozen acquisition rankings; no refitting on new labels."""
import hashlib,json
from pathlib import Path
import numpy as np
from sttw_control.tolerance_acquisition import query_features,acquisition_score
R=Path(__file__).resolve().parent
D=json.loads((R/'declaration.json').read_text()); spec=D['acquisition_protocol']
S=json.loads((R/'evaluation/status.json').read_text()); assert S['phase']=='completed'
model_path=R/'frozen_models.json'
assert hashlib.sha256(model_path.read_bytes()).hexdigest()==spec['frozen_model_sha256']
models=json.loads(model_path.read_text())['models']
rows=json.loads((R/'evaluation/results.json').read_text())['cases']; assert len(rows)==168
key=lambda r:tuple(r[k] for k in ('route','speed','alpha','force_N','onset_s'))
lookup={(r['policy'],key(r)):r for r in rows}; assert len(lookup)==len(rows)
records=[]; X=[]; H=[]
for base in rows:
 if base['policy']!='control' or base['force_N']==0: continue
 k=key(base); useful={}
 for arm in ('speed_off','steer_off'):
  a=lookup[arm,k]
  assert a['prefix_valid'] and a['pulse_complete']
  useful[arm]=bool(not base['task_recovered'] and a['task_recovered'] and not a['physical_failure'] and not a['domain_failure'] and a['logged_return']>base['logged_return'])
 assert base['prefix_valid'] and base['pulse_complete']
 X.append(query_features(base['prefix_features'],base['force_N'],D['duration_s']))
 H.append(query_features(base['prefix_features'],base['force_N'],D['duration_s'],hybrid=True))
 records.append(dict(context=k,useful=useful,either=any(useful.values()),control_task=base['task_recovered']))
assert len(records)==48
scores={name:acquisition_score(model,X if name=='data_opportunity' else H) for name,model in models.items()}
scores.update(analytic_risk=np.array(H)[:,-1],abs_force=np.abs(np.array(X)[:,-2]))
rng=np.random.default_rng(spec['random_seed']);tie=rng.permutation(48)
orders={name:sorted(range(48),key=lambda i:(-s[i],int(tie[i]))) for name,s in scores.items()}
y=np.array([r['either'] for r in records],int)
perms=[rng.permutation(48) for _ in range(spec['random_permutations'])]
curves=[]
for b in spec['query_budgets']:
 dist=np.array([y[o[:b]].sum() for o in perms])
 curves.append(dict(budget=b,**{n:int(y[o[:b]].sum()) for n,o in orders.items()},uniform_exact_expectation=float(b*y.mean()),uniform_p05_p95=np.quantile(dist,[.05,.95]).tolist()))
summary={p:dict(n=sum(r['policy']==p for r in rows),success=sum(r['task_recovered'] for r in rows if r['policy']==p)) for p in D['policies']}
out=dict(summary=summary,useful_total=int(y.sum()),curves=curves,records=records,scores={n:v.tolist() for n,v in scores.items()},orders=orders,actual_lower_ticks=S['active_lower_ticks'],model_sha256=spec['frozen_model_sha256'],results_sha256=hashlib.sha256((R/'evaluation/results.json').read_bytes()).hexdigest(),limitations='Frozen ranking replay; all acquisition costs charged. 48 force queries share 8 prefix contexts. Permutation bands are not population confidence intervals. No Actor updates.')
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('summary','useful_total','curves','actual_lower_ticks')},indent=2))
