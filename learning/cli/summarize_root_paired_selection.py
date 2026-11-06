"""Describe paired selection differences with whole-origin bootstrap resampling."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comparison',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--anchor',default='fixed_2.0_0.0')
p.add_argument('--seed',type=int,default=20261106)
p.add_argument('--resamples',type=int,default=10000)
a=p.parse_args();data=json.loads(a.comparison.read_text());rows=data['groups'];origins=sorted({r['origin_id'] for r in rows});methods=list(rows[0]['methods'])
rng=np.random.default_rng(a.seed);indices=rng.integers(len(origins),size=(a.resamples,len(origins)));scores={}
for method in methods:
 per_root=[]
 for origin in origins:
  group=[r for r in rows if r['origin_id']==origin]
  paired=[r for r in group if r['methods'][method]['Y'] and r['methods'][a.anchor]['Y']]
  per_root.append(dict(origin_id=origin,tasks=len(group),successes=sum(r['methods'][method]['Y'] for r in group),
   anchor_successes=sum(r['methods'][a.anchor]['Y'] for r in group),
   common_success_tasks=len(paired),time_saved_sum_s=sum(r['methods'][a.anchor]['T_s']-r['methods'][method]['T_s'] for r in paired)))
 counts=np.array([[r['successes'],r['anchor_successes'],r['tasks'],r['common_success_tasks'],r['time_saved_sum_s']] for r in per_root]);sample=counts[indices].sum(1)
 delta=(sample[:,0]-sample[:,1])/sample[:,2];valid=sample[:,3]>0
 scores[method]=dict(paired_success_difference=float((counts[:,0].sum()-counts[:,1].sum())/counts[:,2].sum()),
  empirical_root_bootstrap_95_percentile=np.quantile(delta,[.025,.975]).tolist(),
  common_success_tasks=int(counts[:,3].sum()),mean_time_saved_s=float(counts[:,4].sum()/counts[:,3].sum()) if counts[:,3].sum() else None,
  time_difference_percentile_s=np.quantile(sample[valid,4]/sample[valid,3],[.025,.975]).tolist() if valid.any() else None,
  resamples_without_common_success=int((~valid).sum()),per_root=per_root)
result=dict(role='Descriptive pilot DEV uncertainty, not population or independent TEST confidence',anchor=a.anchor,independent_origins=len(origins),task_groups=len(rows),resamples=a.resamples,seed=a.seed,scores=scores,
 scope='Sample whole origins jointly across methods; keep all distances and upstream failures. Six preselected DEV roots are not an IID population sample. Percentiles describe sensitivity to root composition; not a guarantee or post-hoc significance claim. Common-success time intervals omit and count resamples with no common successes.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:dict(delta=v['paired_success_difference'],range=v['empirical_root_bootstrap_95_percentile']) for k,v in scores.items()},indent=2))
