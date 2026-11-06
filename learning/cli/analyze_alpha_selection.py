"""Check two declared alpha selection rules on saved paired repeats (no physics)."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--batch-manifest',type=Path,required=True)
p.add_argument('--eta',type=float,default=.533)
p.add_argument('--path-bound',type=float,default=.4)
p.add_argument('--calibration-speed',type=float,default=2.6)
p.add_argument('--calibration-terrains',nargs='+',default=['oblique_2cm','oblique_4cm'])
a=p.parse_args()
rows=json.loads((a.evaluation/'index.json').read_text())
rows += [dict(terrain=c['terrain'],user_speed_m_s=c['user_speed_m_s'],seed=c['seed'],run=c['output']) for c in json.loads(a.batch_manifest.read_text())]
cases={(c['terrain'],c['user_speed_m_s'],c['seed']):c for c in rows}
if len(cases)!=100:raise ValueError('This declared repetition analysis requires all 100 conditions')
alpha=np.array([0.,.25,.5,.75,1.]);groups={}
for key,c in sorted(cases.items()):
 run=Path(c['run']);status=json.loads((run/'status.json').read_text())
 if status['stage']!='complete':raise ValueError(f'Incomplete physics: {run}')
 with np.load(run/'trajectory.npz') as raw:
  trace=raw['lower_trace'].reshape(-1,6,11)
  groups.setdefault(key[:2],[]).append((c,trace,raw['failure'].any(0)))
result=[]
for (terrain,speed),samples in groups.items():
 count=min(int(t[:,:,6].astype(bool).sum(0).min()) for _,t,_ in samples)
 costs=[];speeds=[];peaks=[];survival=[];per_seed=[]
 for c,t,failed in samples:
  ey=np.sqrt(np.mean(t[:count,:5,3]**2,axis=0));ev=np.sqrt(np.mean((t[:count,:5,2]-speed)**2,axis=0))
  j=ev**2+a.eta**2*ey**2
  peak=[float(np.max(np.abs(t[t[:,i,6].astype(bool),i,3]))) for i in range(5)]
  full=(t[:,:5,6].sum(0)==2000)&~failed[:5]
  per_seed.append(dict(seed=c['seed'],path_rmse=ey.tolist(),speed_rmse=ev.tolist(),cost=j.tolist(),peak_path_m=peak,full_observation=full.tolist(),cost_best_alpha=float(alpha[j.argmin()]),run=c['run']))
  costs.append(j);speeds.append(ev);peaks.append(peak);survival.append(full)
 costs=np.array(costs);mean=costs.mean(0);speed_mean=np.mean(speeds,axis=0)
 feasible=np.all(np.array(survival)&(np.array(peaks)<=a.path_bound),axis=0)
 candidates=np.flatnonzero(feasible);selected=int(candidates[speed_mean[candidates].argmin()]) if len(candidates) else None
 gap=costs[:,3]-costs[:,2]
 result.append(dict(terrain=terrain,speed_m_s=speed,n=len(samples),common_window_s=count*.005,
  mean_cost=mean.tolist(),cost_best_alpha=float(alpha[mean.argmin()]),
  per_seed_best_counts=[int(sum(r['cost_best_alpha']==x for r in per_seed)) for x in alpha],
  alpha075_minus05_cost_mean=float(gap.mean()),alpha075_minus05_cost_sd=float(gap.std(ddof=1)),
  alpha075_beats05_count=int((gap<0).sum()),alpha075_vs05_cost_change_percent=float(100*gap.mean()/mean[2]),
  full_window_feasible_alpha=alpha[candidates].tolist(),constraint_selected_alpha=float(alpha[selected]) if selected is not None else None,
  per_seed=per_seed))
output=dict(role='DEV repetition check; offline task-rule selection, not an online alpha planner',eta_per_s=a.eta,path_bound_m=a.path_bound,
 cost_rule='minimize mean across seeds of Ev^2 + eta^2 Ey^2; common window includes all six methods and all seeds in each panel',
 constraint_rule='require every seed full 10s observation without termination and peak |ey| <= bound, then minimize mean common-window speed RMSE; survival is not goal completion',
 alpha=alpha.tolist(),cases=result)
calibration=[r for r in result if r['speed_m_s']==a.calibration_speed and r['terrain'] in a.calibration_terrains]
calibration_cost=np.mean([r['mean_cost'] for r in calibration],axis=0)
common_feasible=set.intersection(*(set(r['full_window_feasible_alpha']) for r in calibration))
speed_mean=np.mean([seed['speed_rmse'] for r in calibration for seed in r['per_seed']],axis=0)
constraint_choice=min(common_feasible,key=lambda x:speed_mean[list(alpha).index(x)]) if common_feasible else None
output['calibration_repetition']=dict(terrains=a.calibration_terrains,speed_m_s=a.calibration_speed,mean_cost=calibration_cost.tolist(),cost_best_alpha=float(alpha[calibration_cost.argmin()]),alpha075_vs05_cost_change_percent=float(100*(calibration_cost[3]/calibration_cost[2]-1)),common_feasible_alpha=sorted(common_feasible),constraint_choice=constraint_choice)
(a.evaluation/'alpha_selection.json').write_text(json.dumps(output,indent=2))
lines=['# Alpha selection: two distinct task rules','','[Actual/reference trajectories and repeated tradeoffs](RESULTS.md) · [All seeds and five-alpha costs](alpha_selection.json)','',
 'Original calibration: eta=.533 selects alpha=.75 among five levels, only about1% below alpha=.5. A0.40m peak path bound permits .75 on2cm and .5 on4cm; requiring both picks .5. These are different rules, not a universal optimal alpha.','',
 'New repeated checks below retain those weights and bound. The quadratic rule averages per-seed costs, not the squared average RMSE. The constraint rule requires all five seeds to complete the observation without termination and obey the full-window path bound; this is not a claim of goal completion. Zero remains in the common-window definition, but is not an alpha candidate. All runs start from the saved2.3m/s warm state. No new training or physics.','',
 '|Terrain|Speed m/s|n|Window s|Mean-cost best alpha|.75 versus .5 cost change %|Seeds preferring .75 over .5|All-seed bound-feasible alpha|Bound-rule choice|',
 '|---|---:|---:|---:|---:|---:|---:|---|---|']
for r in result:
 lines.append(f"|{r['terrain']}|{r['speed_m_s']:g}|{r['n']}|{r['common_window_s']:.3f}|{r['cost_best_alpha']:g}|{r['alpha075_vs05_cost_change_percent']:+.2f}|{r['alpha075_beats05_count']}/{r['n']}|{r['full_window_feasible_alpha']}|{r['constraint_selected_alpha']}|")
lines += ['', '## Calibration-condition repetition', '', f"For {a.calibration_terrains} at {a.calibration_speed:g}m/s, the repeated mean-cost choice is alpha={output['calibration_repetition']['cost_best_alpha']:g}; .75 versus .5 cost change is {output['calibration_repetition']['alpha075_vs05_cost_change_percent']:+.2f}%. Common full-window bound-feasible levels are {sorted(common_feasible)}, giving alpha={constraint_choice} under the speed-priority constrained rule. These are task-conditioned choices, not a universal optimum.", '', 'Negative percentage favors .75; positive favors .5. No feasible candidate is reported as None rather than forcing a recommendation. These are paired physical-parameter repeats of one trained policy, not independent training seeds or a significance test.']
(a.evaluation/'ALPHA_SELECTION.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(panels=len(result),cost_best_counts={str(x):int(sum(r['cost_best_alpha']==x for r in result)) for x in alpha},bound_choice_counts={str(x):int(sum(r['constraint_selected_alpha']==x for r in result)) for x in [*alpha,None]})))
