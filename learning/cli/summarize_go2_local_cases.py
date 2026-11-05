"""Compare selected saved candidates on fixed nominal cases (DEV, no simulation)."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--reference-label',default='MuJoCo Local300');p.add_argument('--config',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--zero',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot-top',type=int,default=0);a=p.parse_args()
s=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
colors=['#0072B2','#E69F00','#009E73','#CC79A7'];styles=['-','--','-.',':'];results=[];traces=[]
for folder in a.runs:
 identity=json.loads((folder/'identity.json').read_text());d=np.load(folder/'trajectory.npz');cases=[]
 for i,name in enumerate(s['cases']):
  n=int(d['valid'][:,i].sum());ev=d['speed'][:n,i]-d['reference'][:n,i,0];ed=d['measurement'][:n,i,2]-d['reference'][:n,i,1];failed=bool(d['failed'][:,i].any());survived=not failed and n==2000
  tolerance=.01 if name.startswith('post_return_small') else .02
  hold=survived and bool((np.abs(ev[-100:])<=.05).all() and (np.abs(ed[-100:])<=tolerance).all())
  old=np.load(Path(s['mujoco_reference'])/(name+'.npz'));xy=d['pose'][:n,i,:2]-old['reference_xy'][:n]
  cases.append(dict(case=name,survived=survived,final_tracking_hold=hold,end_s=float(d['time'][n-1,i]+d['stop_tick'][n-1,i]*.0002),speed_rmse=float(np.sqrt(np.mean(ev**2))),steer_rmse=float(np.sqrt(np.mean(ed**2))),xy_reference_rmse=float(np.sqrt(np.mean(np.sum(xy**2,axis=-1)))),peak_roll=float(d['peak_roll'][:n,i].max())))
 score=dict(survived=sum(c['survived'] for c in cases),tracking_hold=sum(c['final_tracking_hold'] for c in cases),observed_seconds=sum(c['end_s'] for c in cases),tracking=float(np.mean([c['speed_rmse']/.05+c['steer_rmse']/.02 for c in cases])))
 results.append(dict(**identity,run=str(folder.resolve()),cases=cases,score=score));traces.append(d)
rank=sorted(results,key=lambda r:(-r['score']['survived'],-r['score']['tracking_hold'],-r['score']['observed_seconds'],r['score']['tracking']))
plot_rows=list(zip(traces,results))
if a.plot_top:
 chosen={row['label'] for row in rank[:a.plot_top]}
 plot_rows=[(d,r) for d,r in plot_rows if r['label'] in chosen]
zero=np.load(a.zero/'trajectory.npz');fig,axes=plt.subplots(2,4,figsize=(19,10));axes.flat[-1].axis('off')
for i,name in enumerate(s['cases']):
 ax=axes.flat[i];old=np.load(Path(s['mujoco_reference'])/(name+'.npz'));ax.plot(*old['reference_xy'].T,':',color='.5',label='Original reference');ax.plot(*old['actual_xy'].T,'--',color='black',lw=1.4,label=a.reference_label)
 nz=int(zero['valid'][:,i].sum());ax.plot(*zero['pose'][:nz,i,:2].T,color='.6',lw=1,label='Lab ECBC+ESO')
 for (d,result),color,style in zip(plot_rows,colors,styles):
  n=int(d['valid'][:,i].sum());xy=d['pose'][:n,i,:2];ax.plot(*xy.T,style,color=color,lw=1.4,label=(result['label'] if 'controller_parameters' in result else 'Update '+str(result['completed_updates'])));ax.scatter(*xy[-1],color=color,marker='x',s=18)
 ax.set(title=name,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=10)
fig.suptitle(('ECBC parameter screen, plotted top '+str(a.plot_top) if a.plot_top else ('ECBC parameter candidates' if all('controller_parameters' in r for r in results) else 'Four candidates after simplified-distribution training'))+' | fixed nominal DEV\nSame saved initial state and issued references; crosses mark actual observation endpoints')
fig.tight_layout();fig.savefig(a.output/'all_candidates_xy.png',dpi=160);plt.close(fig)
result=dict(role='fixed-case DEV comparison, not held-out generalization',best=rank[0],ranked=rank,
 selection='survival count, final .5s speed/steer tracking count, observed duration, normalized tracking RMSE',
 adoption='not qualified if any case lacks survival or tracking hold; best only relative to tested candidates',
 limitation='Each error uses its actual observation window; early failure is ranked before error. XY uses original issued-reference trajectory, not a reward target.')
(a.output/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps([dict(label=r['label'],update=r['completed_updates'],**r['score']) for r in rank],indent=2))
