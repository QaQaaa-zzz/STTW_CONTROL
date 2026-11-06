"""Paired repeated-parameter tradeoffs from saved trajectories; no physics."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args()
index=json.loads((a.evaluation/'index.json').read_text())
speeds=sorted({r['user_speed_m_s'] for r in index});terrains=list(dict.fromkeys(r['terrain'] for r in index))
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500','#333333'];markers=['o','s','^','D','v','x'];labels=['0','.25','.5','.75','1','Zero']
fig,axs=plt.subplots(len(speeds),len(terrains),figsize=(4*len(terrains),3.3*len(speeds)),squeeze=False,layout='constrained');records=[];groups=[]
for row,speed in enumerate(speeds):
 for col,terrain in enumerate(terrains):
  ax=axs[row,col];cases=[r for r in index if r['terrain']==terrain and r['user_speed_m_s']==speed];samples=[]
  for case in cases:
   with np.load(Path(case['run'])/'trajectory.npz') as d:
    tr=d['lower_trace'].reshape(-1,6,11);valid=tr[:,:,6].astype(bool);lengths=valid.sum(0)
    samples.append((case,tr.copy(),lengths,d['failure'].any(0).copy()))
  if not samples:ax.set_visible(False);continue
  # Identical duration for every seed and method in this panel; failures retained.
  count=min(int(lengths.min()) for _,_,lengths,_ in samples);window=count*.005;values=[]
  for case,tr,lengths,failed in samples:
   xy=np.sqrt(np.mean(tr[:count,:,3]**2,axis=0));sv=np.sqrt(np.mean((tr[:count,:,2]-speed)**2,axis=0));values.append(np.stack((xy,sv),axis=-1))
   for lane,label in enumerate(labels):records.append(dict(terrain=terrain,speed_m_s=speed,seed=case['seed'],alpha=label,common_window_s=window,path_rmse_m=float(xy[lane]),speed_rmse_m_s=float(sv[lane]),failed=bool(failed[lane]),observed_s=float(lengths[lane]*.005),run=case['run']))
  values=np.stack(values);mean=values.mean(0);std=values.std(0,ddof=1) if len(values)>1 else np.zeros_like(mean)
  ax.plot(mean[:5,0],mean[:5,1],color='.5',lw=1)
  for lane in range(5):
   ax.scatter(values[:,lane,0],values[:,lane,1],color=colors[lane],s=12,alpha=.35)
   ax.errorbar(*mean[lane],xerr=std[lane,0],yerr=std[lane,1],fmt=markers[lane],color=colors[lane],capsize=2,label='alpha '+labels[lane])
  ax.set(title=f'{terrain} | {speed:g} m/s\nn={len(samples)}; shared 0–{window:.2f}s',xlabel='Path RMSE (m)',ylabel='Speed RMSE (m/s)');ax.grid(alpha=.2)
  groups.append(dict(terrain=terrain,speed_m_s=speed,n=len(samples),common_window_s=window,labels=labels,mean=mean.tolist(),sample_sd=std.tolist(),failure_counts=np.sum([x[3] for x in samples],axis=0).tolist()))
axs[0,0].legend(fontsize=7)
fig.suptitle('Original H75 + selected Local300 | paired parameter repeats, DEV\nPoints: seed results; bars: ±1 sample SD; lines connect alpha means, not a fitted law. Zero retained in tables and shared-window selection.')
fig.savefig(a.evaluation/'tradeoff_repeated_mean.png',dpi=160);plt.close(fig)
with (a.evaluation/'tradeoff_repeated_raw.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(a.evaluation/'tradeoff_repeated_summary.json').write_text(json.dumps(groups,indent=2)+'\n')
print(json.dumps(dict(completed_runs=len(index),panels=len(groups),replicates='paired parameter seeds; SD not confidence interval')))
