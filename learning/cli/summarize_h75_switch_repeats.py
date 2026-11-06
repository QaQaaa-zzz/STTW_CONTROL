"""Summarize every frozen parameter repeat of a prescribed H75 switch."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--first',type=Path,required=True);a=p.parse_args()
runs={101:a.first,**{s:a.root/f'seed{s}' for s in range(102,106)}}
cases=['alpha0','alpha025','alpha05','alpha075','alpha1','switch1to05','Zero']
colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7','#555555']
rows=[];pairs=[];data={}
for seed,root in runs.items():
 metrics=json.loads((root/'metrics.json').read_text());by={r['case']:r for r in metrics['cases']}
 for case in cases:
  row=dict(by[case],seed=seed,source=str(root));rows.append(row)
  data[seed,case]=np.load(root/'original'/f'{case}_analysis.npz')
 for base in ['alpha1','alpha05']:
  end=min(by[base]['before_goal_or_termination']['window_end_s'],by['switch1to05']['before_goal_or_termination']['window_end_s'])
  values={}
  for case in [base,'switch1to05']:
   d=data[seed,case];mask=d['time']<=end+1e-9
   values[case]=dict(path_rmse=float(np.sqrt(np.mean(d['original_cross'][mask]**2))),speed_rmse=float(np.sqrt(np.mean((d['speed'][mask]-2.6)**2))),path_peak=float(np.max(abs(d['original_cross'][mask]))))
  pairs.append(dict(seed=seed,baseline=base,window_end_s=end,values=values,delta={k:values['switch1to05'][k]-values[base][k] for k in values[base]}))
fig,axs=plt.subplots(1,5,figsize=(21,5),layout='constrained')
contract=json.loads((a.first/'original/contract.json').read_text());route=np.load(Path(contract['bundle'])/'route.npz')
for ax,(seed,root) in zip(axs,runs.items()):
 keep=route['s']<=11.33;ax.plot(*route['xy'][keep].T,':',color='.4',label='Original route')
 for case,color in zip(cases,colors):
  d=data[seed,case];end=next(r for r in rows if r['seed']==seed and r['case']==case)['before_goal_or_termination']['window_end_s'];xy=d['xy'][d['time']<=end+1e-9];ax.plot(*xy.T,color=color,label=case,lw=1);ax.plot(*xy[-1],marker='x',color=color)
 ax.set(title=f'Parameter seed {seed}',xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
axs[0].legend(fontsize=7);fig.suptitle('Original H75 + Local300 | 6 cm, 2.6 m/s | all replanning | XY until first goal entry or termination\nFive frozen parameter repeats, same initial state; crosses mark observed endpoints; DEV')
fig.savefig(a.root/'xy_repeats.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
for ax,key,unit in zip(axs,['path_rmse','speed_rmse','path_peak'],['m','m/s','m']):
 for x,base in enumerate(['alpha1','alpha05']):
  vals=np.array([r['delta'][key] for r in pairs if r['baseline']==base]);ax.scatter(x+np.linspace(-.08,.08,5),vals,color=colors[5]);ax.errorbar(x,vals.mean(),yerr=vals.std(ddof=1),fmt='D',color='black',capsize=4)
 ax.axhline(0,color='.5',lw=.8);ax.set(xticks=[0,1],xticklabels=['vs fixed 1','vs fixed .5'],ylabel=f'Switch minus baseline {key} ({unit})');ax.grid(axis='y',alpha=.2)
fig.suptitle('Prescribed 1 to .5 switch | paired common pre-arrival windows\nAll five parameter repeats shown; diamond: mean; error bar: sample SD; negative means lower error')
fig.savefig(a.root/'paired_differences.png',dpi=160);plt.close(fig)
summary={}
for base in ['alpha1','alpha05']:
 subset=[r for r in pairs if r['baseline']==base];summary[base]={}
 for key in ['path_rmse','speed_rmse','path_peak']:
  delta=np.array([r['delta'][key] for r in subset]);summary[base][key]=dict(mean_delta=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),lower_count=int((delta<0).sum()),n=len(delta))
(a.root/'summary.json').write_text(json.dumps(dict(role='DEV frozen parameter repeats, not independent training seeds; no exclusions',cases=rows,paired_common_windows=pairs,paired_summary=summary),indent=2)+'\n')
lines=['# Prescribed bend-switch parameter repeats','','![All task-window XY](xy_repeats.png)','','![Paired differences](paired_differences.png)','','Original H75 + Local300, 6cm terrain, requested2.6m/s, same saved2.3m/s initial state. Five fixed alphas, prescribed1to.5 at original progress5.3m, and matched Zero. All methods replan. Seed101 reused;102–105 run as one28-lane GPU scene. No training or switch-point tuning. Error bars are parameter sample SD, not independent-policy confidence intervals.','','## Complete task before first goal entry (or termination)','','|Seed|Case|Goal|Full10s|Arrival s|Path RMSE m|Path peak m|Speed RMSE m/s|','|---|---|---|---|---|---|---|---|']
for r in rows:
 q=r['before_goal_or_termination'];lines.append(f"|{r['seed']}|{r['case']}|{r['goal_arrived']}|{r['survived_full_window']}|{r['first_arrival_s']}|{q['original_path_rmse_m']:.4f}|{q['original_path_peak_m']:.4f}|{q['speed_rmse_m_s']:.4f}|")
lines+=['','## Paired equal-time comparison','','Each switch/baseline pair ends at the earliest first-goal entry or termination of that pair. Zero does not shorten this separate pairwise comparison. Whole-task and all-method common windows remain in each per-seed report. Negative delta means lower error.','','|Baseline|Metric|Mean delta|Sample SD|Lower error repeats|','|---|---|---|---|---|']
for base,metrics in summary.items():
 for key,q in metrics.items():lines.append(f"|{base}|{key}|{q['mean_delta']:.6f}|{q['sample_sd']:.6f}|{q['lower_count']}/5|")
lines+=['','All per-seed XY, rewards and full-window metrics:']
for seed,root in runs.items():lines.append(f'- Seed{seed}: [{root}/RESULTS.md]({root.resolve()}/RESULTS.md)')
lines+=['','Raw paired values and all35 outcomes: [summary.json](summary.json). Fixed-alpha .4m feasibility remains a declared task constraint; improvement does not imply satisfying it. Results include replanning feedback, not an isolated coordination-limiter effect.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))
