"""Leave one parameter repeat out for offline alpha selection; no new physics."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
a=p.parse_args()
source=json.loads((a.evaluation/'alpha_selection.json').read_text())
alpha=np.array(source['alpha']);eta=source['eta_per_s'];bound=source['path_bound_m']
rows=[]
for case in source['cases']:
 costs=[];speeds=[];feasible=[]
 for seed in case['per_seed']:
  with np.load(Path(seed['run'])/'trajectory.npz') as raw:
   t=raw['lower_trace'].reshape(-1,6,11)[:,:5]
   if not (t.shape[0]==2000 and t[:,:,6].all() and not raw['failure'][:,:5].any()):raise ValueError('This analysis requires full10s H75 traces')
   ey=np.mean(t[:,:,3].astype(float)**2,axis=0)
   ev=np.mean((t[:,:,2].astype(float)-case['speed_m_s'])**2,axis=0)
  costs.append(ev+eta**2*ey);speeds.append(np.sqrt(ev));feasible.append(np.array(seed['peak_path_m'])<=bound)
 costs=np.array(costs);speeds=np.array(speeds);feasible=np.array(feasible)
 folds=[]
 for held in range(len(costs)):
  train=np.arange(len(costs))!=held
  selected=int(costs[train].mean(0).argmin())
  allowed=np.flatnonzero(feasible[train].all(0))
  constrained=int(allowed[speeds[train].mean(0)[allowed].argmin()]) if len(allowed) else None
  oracle=int(costs[held].argmin())
  folds.append(dict(held_seed=case['per_seed'][held]['seed'],selected_alpha=float(alpha[selected]),held_cost=float(costs[held,selected]),held_speed_rmse=float(speeds[held,selected]),held_path_rmse=float(np.sqrt(max(0.,(costs[held,selected]-speeds[held,selected]**2)/eta**2))),held_oracle_alpha=float(alpha[oracle]),held_oracle_cost=float(costs[held,oracle]),excess_cost=float(costs[held,selected]-costs[held,oracle]),fixed05_cost=float(costs[held,2]),fixed075_cost=float(costs[held,3]),constraint_selected_alpha=None if constrained is None else float(alpha[constrained]),held_constraint_met=None if constrained is None else bool(feasible[held,constrained]),constraint_held_speed_rmse=None if constrained is None else float(speeds[held,constrained])))
 rows.append(dict(terrain=case['terrain'],speed_m_s=case['speed_m_s'],folds=folds))
folds=[f for row in rows for f in row['folds']]
summary=dict(n=len(folds),oracle_matches=sum(f['selected_alpha']==f['held_oracle_alpha'] for f in folds),mean_cost=float(np.mean([f['held_cost'] for f in folds])),fixed05_mean_cost=float(np.mean([f['fixed05_cost'] for f in folds])),fixed075_mean_cost=float(np.mean([f['fixed075_cost'] for f in folds])),mean_excess_cost=float(np.mean([f['excess_cost'] for f in folds])),constraint_selected=sum(f['constraint_selected_alpha'] is not None for f in folds),constraint_met=sum(f['held_constraint_met'] is True for f in folds),constraint_failed=sum(f['held_constraint_met'] is False for f in folds),no_candidate=sum(f['constraint_selected_alpha'] is None for f in folds))
(a.evaluation/'alpha_transfer.json').write_text(json.dumps(dict(scope='Retrospective leave-one-parameter-repeat-out DEV, not an independent TEST, online planner or new trained policy. All H75 metrics use fixed10s; no Zero-dependent or held-fold-dependent horizon. Four repeats select, the fifth scores. Terrain and requested speed are known. Existing eta and path bound unchanged.',eta_per_s=eta,path_bound_m=bound,summary=summary,cases=rows),indent=2))
lines=['# Alpha choice across parameter repeats','','[Actual/reference trajectories](RESULTS.md) · [All fold results](alpha_transfer.json)','','Four parameter repeats select alpha; the remaining repeat scores it. The policy is frozen. All metrics use fixed10s, independent of Zero failure times. This is retrospective DEV resampling of five already inspected parameter repeats, not an independent TEST or proof of an online planner. The constraint uses full-window peak center-path error<=0.40m; no feasible calibration candidate means abstention, not an automatic pass.','',f'Summary: {json.dumps(summary)}','','|Terrain|Speed|Selected cost alpha in seed101–105 folds|Held cost oracle matches|Constraint selections|Constraint pass / selected|','|---|---:|---|---:|---|---:|']
for r in rows:
 f=r['folds'];lines.append(f"|{r['terrain']}|{r['speed_m_s']}|{[x['selected_alpha'] for x in f]}|{sum(x['selected_alpha']==x['held_oracle_alpha'] for x in f)}/5|{[x['constraint_selected_alpha'] for x in f]}|{sum(x['held_constraint_met'] is True for x in f)}/{sum(x['constraint_selected_alpha'] is not None for x in f)}|")
(a.evaluation/'ALPHA_TRANSFER.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
