"""Raw XY and reward diagnostics for every declared batched ditch candidate."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
cfg=json.loads((a.collection/'config.json').read_text());protocol=json.loads((a.collection/'protocol.json').read_text());summary=json.loads((a.collection/'evaluation/summary.json').read_text());data=np.load(a.collection/'evaluation/trajectories.npz');root=data['root_xyzw'];valid=data['valid'];reward=data['reward'];routes=np.load(cfg['ground_handoff']['reference_path'])['xy'];roads=np.load(cfg['task_reference_path'])['xy'];names=protocol['candidates'] if 'candidates' in protocol else protocol['roles'];colors=['#0072B2','#E69F00','#009E73','#CC79A7'];styles=['-','--','-.',':'];cases=summary['cases'];n=len(names);success=np.array([c['success'] for c in cases]).reshape(-1,n)
if 'method_assignment' in protocol:
 colors=[colors[m] for m in protocol['method_assignment']];styles=[styles[j//2%4] for j in range(n)]
planning=a.collection/'evaluation/planning.json'
route_ids=json.loads(planning.read_text())['selected'] if planning.exists() else cfg['ground_handoff']['reference_route_ids']
fig,axes=plt.subplots(n,1,figsize=(14,2.5*n),layout='constrained')
for j,ax in enumerate(axes):
 for ix,rid in enumerate(sorted(set(route_ids[j::n]))):
  ax.plot(roads[rid,:,0],roads[rid,:,1],color='black',ls=':',lw=1,label='Original road center' if ix==0 else None)
  ax.plot(routes[rid,:,0],routes[rid,:,1],color='black',ls='--',lw=1,label='Issued path' if ix==0 else None)
 for k in range(j,len(cases),n):
  xy=root[valid[:,k],k,:2];passed=cases[k]['success'];ax.plot(xy[:,0],xy[:,1],color=colors[j],alpha=.8 if passed else .22,lw=1.6 if passed else .7);ax.scatter(*xy[-1],marker='o' if passed else 'x',s=12,color=colors[j])
 ax.axvspan(cfg['ditch_terrain']['front_x_m'],cfg['ditch_terrain']['front_x_m']+cfg['ditch_terrain']['width_m'],color='.7',alpha=.4)
 ax.set(title=f"{names[j]}: {success[:,j].sum()}/{len(success)} complete (all data roles pooled for overview only)",xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
axes[0].legend(loc='upper right');fig.suptitle('heading125 + frozen H75, v=2 m/s, alpha=0.5; markers are actual termination; no smoothing')
fig.savefig(a.output/'xy_all_candidates.png',dpi=160,facecolor='white');plt.close(fig)
index=[]
for i in range(len(success)):
 fig,axes=plt.subplots(4,1,figsize=(12,10),layout='constrained')
 for j in range(n):
  k=i*n+j;m=valid[:,k];x=root[m,k];r=reward[m,k];steps=np.arange(1,len(x)+1);label=f"{names[j]} ({'goal' if cases[k]['success'] else 'end '+str(cases[k]['end_code'])})"
  axes[0].plot(x[:,0],x[:,1],color=colors[j],ls=styles[j],label=label);axes[0].scatter(*x[-1,:2],marker='o' if cases[k]['success'] else 'x',color=colors[j])
  rid=route_ids[k];axes[0].plot(routes[rid,:,0],routes[rid,:,1],color=colors[j],ls=':',alpha=.4,lw=.7)
  axes[1].plot(steps,r,color=colors[j],ls=styles[j]);axes[2].plot(steps,r.cumsum(),color=colors[j],ls=styles[j]);axes[3].plot(steps,x[:,7],color=colors[j],ls=styles[j])
 axes[0].set(xlabel='World x (m)',ylabel='World y (m)');axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(ncol=2,fontsize=8)
 axes[1].set(ylabel='Per-step reward',xlabel='Policy step (20 ms)');axes[2].set(ylabel='Cumulative reward',xlabel='Policy step (20 ms)');axes[3].set(ylabel='World vx (m/s)',xlabel='Policy step (20 ms)')
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle(f"Root {i} | {protocol['cases'][i*n]['role'] if 'cases' in protocol else 'Online paired execution'} | heading125/H75 | all declared methods; true endpoints")
 name=f'root_{i:03d}.png';fig.savefig(a.output/name,dpi=130,facecolor='white');plt.close(fig);index.append(f'- Root {i}: [{name}]({name})')
(a.output/'README.md').write_text('# 完整轨迹与奖励索引\n\n'+f'原始数据：`{a.collection}/evaluation/trajectories.npz`。全部{len(cases)}条，不筛除失败；50Hz原始数据，无平滑。XY等比例；圆点为合格目标，叉为真实终点。候选各自原道路与发布参考不同，不把其中之一叫纯ECBC基线。\n\n'+ '\n'.join(index)+'\n')
print(a.output/'xy_all_candidates.png')
