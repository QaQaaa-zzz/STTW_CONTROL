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
 costs=[];speeds=[];paths=[];feasible=[]
 for seed in case['per_seed']:
  with np.load(Path(seed['run'])/'trajectory.npz') as raw:
   t=raw['lower_trace'].reshape(-1,6,11)[:,:5]
   if not (t.shape[0]==2000 and t[:,:,6].all() and not raw['failure'][:,:5].any()):raise ValueError('This analysis requires full10s H75 traces')
   ey=np.mean(t[:,:,3].astype(float)**2,axis=0)
   ev=np.mean((t[:,:,2].astype(float)-case['speed_m_s'])**2,axis=0)
  paths.append(np.sqrt(ey));costs.append(ev+eta**2*ey);speeds.append(np.sqrt(ev));feasible.append(np.array(seed['peak_path_m'])<=bound)
 costs=np.array(costs);speeds=np.array(speeds);feasible=np.array(feasible)
 folds=[]
 for held in range(len(costs)):
  train=np.arange(len(costs))!=held
  selected=int(costs[train].mean(0).argmin())
  allowed=np.flatnonzero(feasible[train].all(0))
  constrained=int(allowed[speeds[train].mean(0)[allowed].argmin()]) if len(allowed) else None
  oracle=int(costs[held].argmin())
  folds.append(dict(held_seed=case['per_seed'][held]['seed'],selected_alpha=float(alpha[selected]),held_cost=float(costs[held,selected]),held_speed_rmse=float(speeds[held,selected]),held_path_rmse=float(np.sqrt(max(0.,(costs[held,selected]-speeds[held,selected]**2)/eta**2))),held_oracle_alpha=float(alpha[oracle]),held_oracle_cost=float(costs[held,oracle]),excess_cost=float(costs[held,selected]-costs[held,oracle]),all_alpha_cost=costs[held].tolist(),all_alpha_speed_rmse=speeds[held].tolist(),all_alpha_path_rmse=paths[held].tolist(),fixed05_cost=float(costs[held,2]),fixed075_cost=float(costs[held,3]),constraint_selected_alpha=None if constrained is None else float(alpha[constrained]),held_constraint_met=None if constrained is None else bool(feasible[held,constrained]),constraint_held_speed_rmse=None if constrained is None else float(speeds[held,constrained])))
 rows.append(dict(terrain=case['terrain'],speed_m_s=case['speed_m_s'],folds=folds))
folds=[f for row in rows for f in row['folds']]
summary=dict(n=len(folds),oracle_matches=sum(f['selected_alpha']==f['held_oracle_alpha'] for f in folds),mean_cost=float(np.mean([f['held_cost'] for f in folds])),fixed05_mean_cost=float(np.mean([f['fixed05_cost'] for f in folds])),fixed075_mean_cost=float(np.mean([f['fixed075_cost'] for f in folds])),mean_excess_cost=float(np.mean([f['excess_cost'] for f in folds])),constraint_selected=sum(f['constraint_selected_alpha'] is not None for f in folds),constraint_met=sum(f['held_constraint_met'] is True for f in folds),constraint_failed=sum(f['held_constraint_met'] is False for f in folds),no_candidate=sum(f['constraint_selected_alpha'] is None for f in folds))
# Exclude a parameter seed across every condition when calibrating the global fixed baseline.
fixed_candidates=[]
for i,x in enumerate(alpha):
 fixed_candidates.append(dict(alpha=float(x),mean_cost=float(np.mean([f['all_alpha_cost'][i] for f in folds])),mean_speed_rmse=float(np.mean([f['all_alpha_speed_rmse'][i] for f in folds])),mean_path_rmse=float(np.mean([f['all_alpha_path_rmse'][i] for f in folds]))))
global_choices={}
for seed in sorted({f['held_seed'] for f in folds}):
 calibration=[f['all_alpha_cost'] for f in folds if f['held_seed']!=seed]
 global_choices[seed]=int(np.mean(calibration,axis=0).argmin())
for f in folds:
 i=global_choices[f['held_seed']]
 f['calibrated_fixed_alpha']=float(alpha[i]);f['calibrated_fixed_cost']=f['all_alpha_cost'][i]
summary['fixed_candidates']=fixed_candidates
summary['global_fixed_selection_by_held_seed']={str(seed):float(alpha[i]) for seed,i in global_choices.items()}
summary['calibrated_fixed_mean_cost']=float(np.mean([f['calibrated_fixed_cost'] for f in folds]))
summary['conditional_vs_calibrated_fixed_cost_reduction_percent']=100*(1-summary['mean_cost']/summary['calibrated_fixed_mean_cost'])
summary['conditional_better_than_calibrated_fixed_count']=sum(f['held_cost']<f['calibrated_fixed_cost']-1e-12 for f in folds)
summary['conditional_worse_than_calibrated_fixed_count']=sum(f['held_cost']>f['calibrated_fixed_cost']+1e-12 for f in folds)
summary['conditional_mean_speed_rmse']=float(np.mean([f['held_speed_rmse'] for f in folds]))
summary['conditional_mean_path_rmse']=float(np.mean([f['held_path_rmse'] for f in folds]))
(a.evaluation/'alpha_transfer.json').write_text(json.dumps(dict(scope='Retrospective leave-one-parameter-repeat-out DEV, not an independent TEST, online planner or new trained policy. All H75 metrics use fixed10s; no Zero-dependent or held-fold-dependent horizon. Four repeats select, the fifth scores. Terrain and requested speed are known. Existing eta and path bound unchanged.',eta_per_s=eta,path_bound_m=bound,summary=summary,cases=rows),indent=2))
lines=['# Alpha choice across parameter repeats','','[Actual/reference trajectories](RESULTS.md) · [All fold results](alpha_transfer.json)','','Four parameter repeats select alpha; the remaining repeat scores it. The policy is frozen. All metrics use fixed10s, independent of Zero failure times. This is retrospective DEV resampling of five already inspected parameter repeats, not an independent TEST or proof of an online planner. The constraint uses full-window peak center-path error<=0.40m; no feasible calibration candidate means abstention, not an automatic pass.','',f'Summary: {json.dumps(summary)}','','|Terrain|Speed|Selected cost alpha in seed101–105 folds|Held cost oracle matches|Constraint selections|Constraint pass / selected|','|---|---:|---|---:|---|---:|']
for r in rows:
 f=r['folds'];lines.append(f"|{r['terrain']}|{r['speed_m_s']}|{[x['selected_alpha'] for x in f]}|{sum(x['selected_alpha']==x['held_oracle_alpha'] for x in f)}/5|{[x['constraint_selected_alpha'] for x in f]}|{sum(x['held_constraint_met'] is True for x in f)}/{sum(x['constraint_selected_alpha'] is not None for x in f)}|")
lines += ['', '## All fixed-alpha baselines', '', 'A global fixed alpha is calibrated on the other four parameter seeds across all20 conditions, then scored on the excluded seed. Conditions have equal weight. The same seed is excluded across every condition. This is a stronger baseline than picking only .5 or .75 for comparison.', '', '|Fixed alpha|Mean cost|Mean speed RMSE m/s|Mean path RMSE m|', '|---|---:|---:|---:|']
for r in fixed_candidates:
 lines.append(f"|{r['alpha']:g}|{r['mean_cost']:.8f}|{r['mean_speed_rmse']:.6f}|{r['mean_path_rmse']:.6f}|")
lines += ['', f"Global choices by excluded seed: {summary['global_fixed_selection_by_held_seed']}. Conditional selection reduces mean cost by {summary['conditional_vs_calibrated_fixed_cost_reduction_percent']:.2f}% relative to that baseline; it is better in {summary['conditional_better_than_calibrated_fixed_count']}/100 and worse in {summary['conditional_worse_than_calibrated_fixed_count']}/100 (remaining ties). Conditional mean speed/path RMSE are {summary['conditional_mean_speed_rmse']:.6f}m/s and {summary['conditional_mean_path_rmse']:.6f}m. These are means of per-run RMSE, not pooled RMSE. The gain refers to the declared quadratic cost, not simultaneous improvement of both metrics in every condition."]
(a.evaluation/'ALPHA_TRANSFER.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
