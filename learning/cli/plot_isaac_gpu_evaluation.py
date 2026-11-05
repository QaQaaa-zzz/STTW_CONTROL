"""Plot saved fixed-alpha GPU trajectories without rerunning physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=np.load(a.run/'trajectory.npz');c=json.loads((a.run/'contract.json').read_text());route=np.load(Path(c['bundle'])/'route.npz');labels=['alpha 0','alpha .25','alpha .5','alpha .75','alpha 1','Zero'];colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','black'];summary=[]
fig,axs=plt.subplots(2,2,figsize=(12,9));xy,per,cumulative,speed=axs.ravel();ref=route['s']<=14;xy.plot(*route['xy'][ref].T,':',color='gray',label='Original reference')
for lane,(label,color) in enumerate(zip(labels,colors)):
    trace=r['lower_trace'][:,:,lane].reshape(-1,11);mask=trace[:,6].astype(bool);indices=np.flatnonzero(mask);steps=np.flatnonzero(r['valid'][:,lane]);position=trace[indices,:2];t=(indices+1)*.005;rewards=r['reward'][steps,lane]
    xy.plot(*position.T,color=color,label=label);xy.scatter(*position[-1],marker='x',color=color)
    per.plot(steps+1,rewards,color=color,label=label);cumulative.plot(steps+1,np.cumsum(rewards),color=color);speed.plot(t,trace[indices,2]-2.3,color=color)
    summary.append(dict(lane=lane,label=label,observed_lower_steps=len(indices),failed=bool(r['failure'][:,lane].any()),last_progress_m=float(r['progress'][indices[-1],lane]),speed_rmse=float(np.sqrt(np.mean((trace[indices,2]-2.3)**2))),cross_rmse=float(np.sqrt(np.mean(trace[indices,3]**2))),peak_roll_rad=float(trace[indices,4].max()),rate_limited_lower_steps=int(r['rate_clipped'][indices,lane].sum())))
xy.set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY; crosses mark observed endpoints');xy.axis('equal');xy.legend(fontsize=8);per.set(xlabel='Policy step',ylabel='Reward',title='Per-step reward (includes terminal penalty)');cumulative.set(xlabel='Policy step',ylabel='Cumulative reward');speed.set(xlabel='Time (s)',ylabel='True speed error (m/s)')
fig.suptitle('Original H75, bend bump, fixed alpha, seed 1 — exploratory DEV\nZero = LOCAL300 + ECBC/ESO; weighted rewards are not cross-alpha performance ranks');fig.tight_layout();fig.savefig(a.run/'overview.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,6));ax.plot(*route['xy'][ref].T,':',color='gray',label='Original reference')
for lane,(label,color) in enumerate(zip(labels,colors)):
 trace=r['lower_trace'][:,:,lane].reshape(-1,11);v=trace[trace[:,6].astype(bool)];ax.plot(v[:,0],v[:,1],color=color,label=label);ax.scatter(*v[-1,:2],marker='x',color=color)
ax.set(xlabel='World x (m)',ylabel='World y (m)',title='Bend bump: actual/reference XY, seed 1');ax.axis('equal');ax.legend();fig.tight_layout();fig.savefig(a.run/'xy.png',dpi=170);plt.close(fig)
(a.run/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
