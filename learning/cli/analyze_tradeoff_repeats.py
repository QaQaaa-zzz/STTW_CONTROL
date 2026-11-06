"""Paired repeated-parameter tradeoffs from saved trajectories; no physics."""
import argparse,json,csv,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--window-mode',choices=['panel','global','both'],default='both');a=p.parse_args()
if a.window_mode=='both':
 for mode in ('panel','global'):subprocess.run([sys.executable,__file__,'--evaluation',str(a.evaluation),'--window-mode',mode],check=True)
 sys.exit(0)
index=json.loads((a.evaluation/'index.json').read_text())
suffix='' if a.window_mode=='panel' else '_global_window'
replicate_label='paired parameter repeats' if any(r.get('seed') is not None for r in index) else 'nominal single-state preview'
global_count=None
if a.window_mode=='global':
 counts=[]
 for case in index:
  with np.load(Path(case['run'])/'trajectory.npz') as d:counts.append(int(d['lower_trace'].reshape(-1,6,11)[:,:,6].astype(bool).sum(0).min()))
 global_count=min(counts)
speeds=sorted({r['user_speed_m_s'] for r in index});terrains=list(dict.fromkeys(r['terrain'] for r in index))
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500','#333333'];markers=['o','s','^','D','v','x'];labels=['0','.25','.5','.75','1','Zero']
fig,axs=plt.subplots(len(speeds),len(terrains),figsize=(4*len(terrains),3.3*len(speeds)),squeeze=False,layout='constrained');records=[];groups=[];exchanges=[]
for row,speed in enumerate(speeds):
 for col,terrain in enumerate(terrains):
  ax=axs[row,col];cases=[r for r in index if r['terrain']==terrain and r['user_speed_m_s']==speed];samples=[]
  for case in cases:
   with np.load(Path(case['run'])/'trajectory.npz') as d:
    tr=d['lower_trace'].reshape(-1,6,11);valid=tr[:,:,6].astype(bool);lengths=valid.sum(0)
    samples.append((case,tr.copy(),lengths,d['failure'].any(0).copy()))
  if not samples:ax.set_visible(False);continue
  # Identical duration for every seed and method in this panel; failures retained.
  count=global_count if global_count is not None else min(int(lengths.min()) for _,_,lengths,_ in samples);window=count*.005;values=[]
  for case,tr,lengths,failed in samples:
   xy=np.sqrt(np.mean(tr[:count,:,3]**2,axis=0));sv=np.sqrt(np.mean((tr[:count,:,2]-speed)**2,axis=0));values.append(np.stack((xy,sv),axis=-1))
   for lane,label in enumerate(labels):records.append(dict(terrain=terrain,speed_m_s=speed,seed=case.get('seed'),alpha=label,common_window_s=window,path_rmse_m=float(xy[lane]),speed_rmse_m_s=float(sv[lane]),failed=bool(failed[lane]),observed_s=float(lengths[lane]*.005),run=case['run']))
  values=np.stack(values);mean=values.mean(0);std=values.std(0,ddof=1) if len(values)>1 else None
  # Paired adjacent-alpha changes preserve folds and seed disagreement.
  delta=np.diff(values[:,:5,:],axis=1)
  for edge in range(4):
   path_delta=delta[:,edge,0];speed_gain=-delta[:,edge,1]
   dx=float(path_delta.mean());gain=float(speed_gain.mean())
   exchanges.append(dict(terrain=terrain,speed_m_s=speed,n=len(samples),
    alpha_from=labels[edge],alpha_to=labels[edge+1],common_window_s=window,
    path_cost_mean_m=dx,speed_gain_mean_m_s=gain,
    path_cost_sample_sd_m=float(path_delta.std(ddof=1)) if len(samples)>1 else None,
    speed_gain_sample_sd_m_s=float(speed_gain.std(ddof=1)) if len(samples)>1 else None,
    mean_exchange_per_s=gain/dx if dx>0 else None,
    conventional_tradeoff_count=int(((path_delta>0)&(speed_gain>0)).sum()),
    both_errors_improve_count=int(((path_delta<0)&(speed_gain>0)).sum()),
    both_errors_worsen_count=int(((path_delta>0)&(speed_gain<0)).sum()),
    reverse_tradeoff_count=int(((path_delta<0)&(speed_gain<0)).sum())))
  ax.plot(mean[:5,0],mean[:5,1],color='.5',lw=1)
  for lane in range(5):
   ax.scatter(values[:,lane,0],values[:,lane,1],color=colors[lane],s=12,alpha=.35)
   ax.errorbar(*mean[lane],xerr=std[lane,0] if std is not None else None,yerr=std[lane,1] if std is not None else None,fmt=markers[lane],color=colors[lane],capsize=2,label='alpha '+labels[lane])
  ax.set(title=f'{terrain} | {speed:g} m/s\nn={len(samples)}; shared 0–{window:.2f}s',xlabel='Path RMSE (m)',ylabel='Speed RMSE (m/s)');ax.grid(alpha=.2)
  groups.append(dict(terrain=terrain,speed_m_s=speed,n=len(samples),window_mode=a.window_mode,common_window_s=window,labels=labels,mean=mean.tolist(),sample_sd=std.tolist() if std is not None else None,failure_counts=np.sum([x[3] for x in samples],axis=0).tolist()))
axs[0,0].legend(fontsize=7)
fig.suptitle(a.window_mode+' common window | Original H75 + selected Local300 | '+replicate_label+', DEV\nPoints: seed results; bars: ±1 sample SD (omitted for n=1); lines connect alpha means, not a fitted law. Zero retained in tables and shared-window selection.')
fig.savefig(a.evaluation/('tradeoff_repeated_mean'+suffix+'.png'),dpi=160);plt.close(fig)
with (a.evaluation/('tradeoff_repeated_raw'+suffix+'.csv')).open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(a.evaluation/('tradeoff_repeated_summary'+suffix+'.json')).write_text(json.dumps(groups,indent=2)+'\n')
(a.evaluation/('tradeoff_marginal_exchange'+suffix+'.json')).write_text(json.dumps(exchanges,indent=2)+'\n')
print(json.dumps(dict(completed_runs=len(index),panels=len(groups),replicates='paired parameter seeds; SD not confidence interval')))
