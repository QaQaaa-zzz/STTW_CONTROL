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
first_case=next(iter(cases.values()))
contract=json.loads((Path(first_case['run'])/'contract.json').read_text())
with np.load(Path(contract['bundle'])/'route.npz') as route:
 goal_s=float(route['turn_end'])+5.
 goal_xy=np.array([np.interp(goal_s,route['s'],route['xy'][:,k]) for k in (0,1)])
result=[]
for (terrain,speed),samples in groups.items():
 count=min(int(t[:,:,6].astype(bool).sum(0).min()) for _,t,_ in samples)
 costs=[];speeds=[];peaks=[];survival=[];per_seed=[]
 for c,t,failed in samples:
  ey=np.sqrt(np.mean(t[:count,:5,3]**2,axis=0));ev=np.sqrt(np.mean((t[:count,:5,2]-speed)**2,axis=0))
  j=ev**2+a.eta**2*ey**2
  peak=[float(np.max(np.abs(t[t[:,i,6].astype(bool),i,3]))) for i in range(5)]
  full=(t[:,:5,6].sum(0)==2000)&~failed[:5]
  task_windows=[]
  for i in range(5):
   ids=np.flatnonzero(t[:,i,6]);v=t[ids,i];arrival=np.flatnonzero(np.linalg.norm(v[:,:2]-goal_xy,axis=1)<=2.)
   end=int(arrival[0])+1 if len(arrival) else len(v)
   task_windows.append(dict(goal_arrived=bool(len(arrival)),arrival_s=float((ids[arrival[0]]+1)*.005) if len(arrival) else None,peak_before_goal_or_end_m=float(np.abs(v[:end,3]).max()),bound_met_before_goal_or_end=bool(np.abs(v[:end,3]).max()<=a.path_bound)))
  per_seed.append(dict(seed=c['seed'],task_windows=task_windows,path_rmse=ey.tolist(),speed_rmse=ev.tolist(),cost=j.tolist(),peak_path_m=peak,full_observation=full.tolist(),cost_best_alpha=float(alpha[j.argmin()]),run=c['run']))
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
  per_alpha_feasible_seed_counts=np.sum(np.array(survival)&(np.array(peaks)<=a.path_bound),axis=0).tolist(),
  peak_path_range_m=np.stack((np.min(peaks,axis=0),np.max(peaks,axis=0)),axis=1).tolist(),
  full_window_feasible_alpha=alpha[candidates].tolist(),constraint_selected_alpha=float(alpha[selected]) if selected is not None else None,
  per_seed=per_seed))
output=dict(role='DEV repetition check; offline task-rule selection, not an online alpha planner',eta_per_s=a.eta,path_bound_m=a.path_bound,
 cost_rule='minimize mean across seeds of Ev^2 + eta^2 Ey^2; common window includes all six methods and all seeds in each panel',
 constraint_rule='require every seed full 10s observation without termination and peak |ey| <= bound, then minimize mean common-window speed RMSE; survival is not goal completion',
 alpha=alpha.tolist(),goal_window_definition=dict(goal_xy=goal_xy.tolist(),goal_after_turn_m=5.,arrival_radius_m=2.,role='Separate retrospective first-arrival window; full10s metrics unchanged'),cases=result)
calibration=[r for r in result if r['speed_m_s']==a.calibration_speed and r['terrain'] in a.calibration_terrains]
calibration_cost=np.mean([r['mean_cost'] for r in calibration],axis=0)
common_feasible=set.intersection(*(set(r['full_window_feasible_alpha']) for r in calibration))
speed_mean=np.mean([seed['speed_rmse'] for r in calibration for seed in r['per_seed']],axis=0)
constraint_choice=min(common_feasible,key=lambda x:speed_mean[list(alpha).index(x)]) if common_feasible else None
output['calibration_repetition']=dict(terrains=a.calibration_terrains,speed_m_s=a.calibration_speed,mean_cost=calibration_cost.tolist(),cost_best_alpha=float(alpha[calibration_cost.argmin()]),alpha075_vs05_cost_change_percent=float(100*(calibration_cost[3]/calibration_cost[2]-1)),common_feasible_alpha=sorted(common_feasible),constraint_choice=constraint_choice)
# Each discrete candidate is a line A + eta^2 B; intersect its pairwise optimality intervals.
calibration_seeds=[seed for r in calibration for seed in r['per_seed']]
speed_squared=np.mean([np.square(seed['speed_rmse']) for seed in calibration_seeds],axis=0)
path_squared=np.mean([np.square(seed['path_rmse']) for seed in calibration_seeds],axis=0)
weight_intervals=[]
for i in range(len(alpha)):
 lo,hi=0.,float('inf')
 for j in range(len(alpha)):
  if i==j:continue
  slope=path_squared[i]-path_squared[j];rhs=speed_squared[j]-speed_squared[i]
  if slope>0:hi=min(hi,rhs/slope)
  elif slope<0:lo=max(lo,rhs/slope)
  elif rhs<0:hi=-1.
 if hi>=lo and hi>=0:
  weight_intervals.append(dict(alpha=float(alpha[i]),eta_min_per_s=float(max(lo,0)**.5),eta_max_per_s=float(hi**.5) if np.isfinite(hi) else None))
output['calibration_repetition']['optimal_eta_intervals']=weight_intervals
lines=['# Alpha selection: two distinct task rules','','[Actual/reference trajectories and repeated tradeoffs](RESULTS.md) · [All seeds and five-alpha costs](alpha_selection.json)','',
 'Original calibration: eta=.533 selects alpha=.75 among five levels, only about1% below alpha=.5. A0.40m peak path bound permits .75 on2cm and .5 on4cm; requiring both picks .5. These are different rules, not a universal optimal alpha.','',
 'New repeated checks below retain those weights and bound. The quadratic rule averages per-seed costs, not the squared average RMSE. The constraint rule requires all five seeds to complete the observation without termination and obey the full-window path bound; this is not a claim of goal completion. Zero remains in the common-window definition, but is not an alpha candidate. All runs start from the saved2.3m/s warm state. No new training or physics.','',
 '|Terrain|Speed m/s|n|Window s|Mean-cost best alpha|.75 versus .5 cost change %|Seeds preferring .75 over .5|All-seed bound-feasible alpha|Bound-rule choice|',
 '|---|---:|---:|---:|---:|---:|---:|---|---|']
for r in result:
 lines.append(f"|{r['terrain']}|{r['speed_m_s']:g}|{r['n']}|{r['common_window_s']:.3f}|{r['cost_best_alpha']:g}|{r['alpha075_vs05_cost_change_percent']:+.2f}|{r['alpha075_beats05_count']}/{r['n']}|{r['full_window_feasible_alpha']}|{r['constraint_selected_alpha']}|")
lines += ['', '## Full-window bound feasibility', '', 'Counts require a full10s observation without termination and peak absolute center-path error within the declared bound. They are not fall counts or automatic proof of goal completion. All five parameter repeats and all five alpha levels are retained.', '', '|Terrain|Speed m/s|Counts for alpha0/.25/.5/.75/1|Denominator per alpha|', '|---|---:|---|---:|']
for r in result:
 lines.append(f"|{r['terrain']}|{r['speed_m_s']:g}|{' / '.join(map(str,r['per_alpha_feasible_seed_counts']))}|{r['n']}|")
lines += ['', '## Calibration-condition repetition', '', f"For {a.calibration_terrains} at {a.calibration_speed:g}m/s, the repeated mean-cost choice is alpha={output['calibration_repetition']['cost_best_alpha']:g}; .75 versus .5 cost change is {output['calibration_repetition']['alpha075_vs05_cost_change_percent']:+.2f}%. Common full-window bound-feasible levels are {sorted(common_feasible)}, giving alpha={constraint_choice} under the speed-priority constrained rule. These are task-conditioned choices, not a universal optimum.", '', 'Negative percentage favors .75; positive favors .5. No feasible candidate is reported as None rather than forcing a recommendation. These are paired physical-parameter repeats of one trained policy, not independent training seeds or a significance test.']
lines += ['', '## First-arrival interpretation check', '', 'Original task goal is turn end+5m with2m arrival radius. Full10s constraint violations may occur AFTER first goal entry. They do not imply inability to reach the goal. In4cm/1.5m/s, all five alphas reach the goal in all five repeats and all satisfy the0.40m bound BEFORE arrival; the previously reported0/5 versus5/5 split concerns continued tracking over the full10s only.', '', '|Terrain|Speed|Goal arrivals by alpha0/.25/.5/.75/1|Pre-arrival/end bound counts|', '|---|---:|---|---|']
for r in result:
 arrivals=[sum(seed['task_windows'][i]['goal_arrived'] for seed in r['per_seed']) for i in range(5)]
 counts=[sum(seed['task_windows'][i]['bound_met_before_goal_or_end'] for seed in r['per_seed']) for i in range(5)]
 lines.append(f"|{r['terrain']}|{r['speed_m_s']:g}|{arrivals}|{counts}|")
lines += ['', '## Arrival-time tradeoff under the original goal definition', '', 'Descriptive mean arrival times over all five parameter repeats. These are observed task outcomes, not a new trained selector. Feasibility requires arrival and the path bound before first entry; the post-arrival tail is excluded only from this separate table. All original full-window analyses remain above.', '', '|Terrain|Speed|Mean arrival seconds: alpha0/.25/.5/.75/1|All-repeat feasible alpha before arrival|Fastest feasible mean alpha|Time saved versus alpha0 s|', '|---|---:|---|---|---:|---:|']
for r in result:
 task=[[seed['task_windows'][i] for seed in r['per_seed']] for i in range(5)]
 means=[float(np.mean([q['arrival_s'] for q in group])) if all(q['goal_arrived'] for q in group) else None for group in task]
 allowed=[i for i,group in enumerate(task) if all(q['goal_arrived'] and q['bound_met_before_goal_or_end'] for q in group)]
 choice=min(allowed,key=lambda i:means[i]) if allowed else None
 r['arrival_summary']=dict(mean_arrival_s=means,feasible_alpha_before_arrival=[float(alpha[i]) for i in allowed],fastest_feasible_alpha=None if choice is None else float(alpha[choice]),saving_vs_alpha0_s=None if choice is None or means[0] is None else means[0]-means[choice])
 times=' / '.join('not all arrived' if t is None else f'{t:.3f}' for t in means)
 lines.append(f"|{r['terrain']}|{r['speed_m_s']:g}|{times}|{r['arrival_summary']['feasible_alpha_before_arrival']}|{r['arrival_summary']['fastest_feasible_alpha']}|{r['arrival_summary']['saving_vs_alpha0_s']}|")
(a.evaluation/'alpha_selection.json').write_text(json.dumps(output,indent=2))
lines += ['', '## Task-weight sensitivity', '', 'For the same calibration repeats, the mean cost is A + eta^2 B. Pairwise line intersections give these discrete optimality intervals; boundary values are ties. These are empirical policy choices, not a fitted continuous alpha law.', '', '|Alpha|Minimum eta (1/s)|Maximum eta (1/s)|', '|---|---:|---:|']
for r in weight_intervals:
 upper='unbounded' if r['eta_max_per_s'] is None else f"{r['eta_max_per_s']:.6f}"
 lines.append(f"|{r['alpha']:g}|{r['eta_min_per_s']:.6f}|{upper}|")
(a.evaluation/'ALPHA_SELECTION.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(panels=len(result),cost_best_counts={str(x):int(sum(r['cost_best_alpha']==x for r in result)) for x in alpha},bound_choice_counts={str(x):int(sum(r['constraint_selected_alpha']==x for r in result)) for x in [*alpha,None]})))
