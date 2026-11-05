"""Plot saved fixed-alpha GPU trajectories without rerunning physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--title',default='H75 fixed-alpha GPU evaluation');a=p.parse_args();r=np.load(a.run/'trajectory.npz');c=json.loads((a.run/'contract.json').read_text());route=np.load(Path(c['bundle'])/'route.npz');labels=['alpha 0','alpha .25','alpha .5','alpha .75','alpha 1','Zero'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500','#333333'];styles=['-']*5+['--'];summary=[]
fig,axs=plt.subplots(2,2,figsize=(12,9));xy,per,cumulative,speed=axs.ravel();last_progress=max(float(r['progress'][np.flatnonzero(r['lower_trace'][:,:,i,6].reshape(-1))[-1],i]) for i in range(6));ref=route['s']<=last_progress+1;xy.plot(*route['xy'][ref].T,':',color='gray',label='Original reference')
for lane,(label,color) in enumerate(zip(labels,colors)):
    trace=r['lower_trace'][:,:,lane].reshape(-1,11);mask=trace[:,6].astype(bool);indices=np.flatnonzero(mask);steps=np.flatnonzero(r['valid'][:,lane]);position=trace[indices,:2];t=indices*.005+trace[indices,7]*.0002;rewards=r['reward'][steps,lane]
    xy.plot(*position.T,color=color,label=label,linestyle=styles[lane]);xy.scatter(*position[-1],marker='x',color=color)
    per.plot(steps+1,rewards,color=color,label=label,linestyle=styles[lane]);cumulative.plot(steps+1,np.cumsum(rewards),color=color);speed.plot(t,trace[indices,2]-2.3,color=color)
    summary.append(dict(observation_end_s=float(t[-1]),lane=lane,label=label,observed_lower_steps=len(indices),failed=bool(r['failure'][:,lane].any()),last_progress_m=float(r['progress'][indices[-1],lane]),speed_rmse=float(np.sqrt(np.mean((trace[indices,2]-2.3)**2))),cross_rmse=float(np.sqrt(np.mean(trace[indices,3]**2))),peak_roll_rad=float(trace[indices,4].max()),rate_limited_lower_steps=int(r['rate_clipped'][indices,lane].sum())))
xy.set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY; crosses mark observed endpoints');xy.axis('equal');xy.legend(fontsize=8);per.set(xlabel='Policy step',ylabel='Reward',title='Per-step reward (includes terminal penalty)');cumulative.set(xlabel='Policy step',ylabel='Cumulative reward');speed.set(xlabel='Time (s)',ylabel='True speed error (m/s)')
fig.suptitle(a.title+' — exploratory DEV, one initial state\nZero = LOCAL300 + ECBC/ESO; weighted rewards are not cross-alpha performance ranks');fig.tight_layout();fig.savefig(a.run/'overview.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,6));ax.plot(*route['xy'][ref].T,':',color='gray',label='Original reference')
for lane,(label,color) in enumerate(zip(labels,colors)):
 trace=r['lower_trace'][:,:,lane].reshape(-1,11);v=trace[trace[:,6].astype(bool)];ax.plot(v[:,0],v[:,1],color=color,label=label,linestyle=styles[lane]);ax.scatter(*v[-1,:2],marker='x',color=color)
ax.set(xlabel='World x (m)',ylabel='World y (m)',title=a.title+': actual/reference XY');ax.axis('equal');ax.legend();fig.tight_layout();fig.savefig(a.run/'xy.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,8),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
for ax in axs:
 ax.plot(*route['xy'][ref].T,':',color='.55',label='Fixed original path')
 for lane,(label,color) in enumerate(zip(labels,colors)):
  trace=r['lower_trace'][:,:,lane].reshape(-1,11);v=trace[trace[:,6].astype(bool)];ax.plot(v[:,0],v[:,1],color=color,label=label,linestyle=styles[lane]);ax.plot(*v[-1,:2],marker='x',color=color)
 ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='box')
turn=np.flatnonzero(abs(route['curvature'])>1e-5);bend=route['xy'][turn];axs[1].set(xlim=(bend[:,0].min()-.5,bend[:,0].max()+.7),ylim=(bend[:,1].min()-.3,bend[:,1].max()+1),title='Bend detail');axs[0].set_title('Full trajectory; × actual endpoint');axs[1].legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9);fig.suptitle(a.title+' | unchanged H75 u77 + selected original Local300');fig.savefig(a.run/'xy_all_alpha.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for lane,(label,color,ax) in enumerate(zip(labels,colors,axs.flat)):
 trace=r['lower_trace'][:,:,lane].reshape(-1,11);ids=np.flatnonzero(trace[:,6]);t=ids*.005+trace[ids,7]*.0002
 ax.plot(t,trace[ids,2],color=color,label='True forward speed');ax.plot(t,trace[ids,10],'--',color='.4',label='Rear axle × 0.1 m');ax.axhline(2.3,color='black',linestyle=':',label='User request');ax.plot(t,trace[ids,8],'-.',color=color,label='Executed speed reference');ax.set(title=label,xlabel='Time (s)',ylabel='Speed (m/s)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle(a.title+' — true / wheel proxy / requested / governed speeds');fig.savefig(a.run/'speed_measurements.png',dpi=150);plt.close(fig)
(a.run/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

(a.run/'figure_manifest.json').write_text(json.dumps(dict(source='trajectory.npz',outputs=['xy.png','overview.png','xy_all_alpha.png','speed_measurements.png'],role='exploratory DEV, one initial seed, no uncertainty estimate',filtering='active_lower mask only, no smoothing; each lane stops at observed failure',metric_window='individual observed duration; unequal durations do not support direct success rankings',reward='native alpha-weighted rewards; Zero scored alpha=0 only, not yet rescored at other alphas',checkpoint=c['identity']['files']['actor.npz'],matplotlib=matplotlib.__version__),indent=2))
