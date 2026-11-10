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
moving_path=a.collection/'evaluation/moving_planning.json'
switches={r['lane']:r for r in json.loads(moving_path.read_text())['records'] if r['status']=='published'} if moving_path.exists() else {}
initial_routes=list(route_ids);route_ids=list(route_ids)
for lane,record in switches.items():route_ids[lane]=record['selected']
if 'cases' in protocol and 'logical_case' in protocol['cases'][0]:
 order=np.argsort([c['logical_case'] for c in protocol['cases']]);root=root[:,order];valid=valid[:,order];reward=reward[:,order]
 cases=[cases[i] for i in order];protocol['cases']=[protocol['cases'][i] for i in order]
 route_ids=[route_ids[i] for i in order];initial_routes=[initial_routes[i] for i in order];switches={j:switches[i] for j,i in enumerate(order) if i in switches}
 success=np.array([c['success'] for c in cases]).reshape(-1,n)

def reference_segments(lane,bank):
 if lane not in switches:return [bank[route_ids[lane]]]
 record=switches[lane];x=record['published_root'][0];before=bank[initial_routes[lane]];after=bank[record['selected']]
 return [before[before[:,0]<=x],after[after[:,0]>=x]]

fig,axes=plt.subplots(n,1,figsize=(14,2.5*n),layout='constrained')
for j,ax in enumerate(axes):
 for index,k in enumerate(range(j,len(cases),n)):
  for seg,points in enumerate(reference_segments(k,roads)):ax.plot(points[:,0],points[:,1],color='black',ls=':',lw=.7,alpha=.5,label='Original road center' if index==0 and seg==0 else None)
  for seg,points in enumerate(reference_segments(k,routes)):ax.plot(points[:,0],points[:,1],color='black',ls='--',lw=.7,alpha=.5,label='Issued path' if index==0 and seg==0 else None)
 for k in range(j,len(cases),n):
  xy=root[valid[:,k],k,:2];passed=cases[k]['success'];ax.plot(xy[:,0],xy[:,1],color=colors[j],alpha=.8 if passed else .22,lw=1.6 if passed else .7);ax.scatter(*xy[-1],marker='o' if passed else 'x',s=12,color=colors[j])
 ax.axvspan(cfg['ditch_terrain']['front_x_m'],cfg['ditch_terrain']['front_x_m']+cfg['ditch_terrain']['width_m'],color='.7',alpha=.4)
 ax.set(title=f"{names[j]}: {success[:,j].sum()}/{len(success)} complete (all data roles pooled for overview only)",xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
axes[0].legend(loc='upper right');fig.suptitle('heading125 + frozen H75, v=2 m/s, alpha=0.5; markers are actual termination; no smoothing')
fig.savefig(a.output/'xy_all_candidates.png',dpi=160,facecolor='white');plt.close(fig)
if 'land_bridge_y_interval_m' in cfg['ditch_terrain']:
 from matplotlib.patches import Rectangle
 spec=cfg['ditch_terrain'];front=spec['front_x_m'];width=spec['width_m'];lo,hi=spec['land_bridge_y_interval_m']
 fig,ax=plt.subplots(figsize=(14,5),layout='constrained')
 ax.add_patch(Rectangle((-5,-4),30,8,facecolor='#eeeeee',edgecolor='.5'))
 ax.add_patch(Rectangle((front,-4),width,8,facecolor='#777777',label='Open trench'))
 ax.add_patch(Rectangle((front,lo),width,hi-lo,facecolor='#99cc99',label='Continuous side ground'))
 goal=cfg['ground_handoff']['goal'];ax.add_patch(Rectangle((goal['x_min_m'],-goal['abs_y_max_m']),goal['x_max_m']-goal['x_min_m'],2*goal['abs_y_max_m'],fill=False,edgecolor='black',lw=2,label='Common goal'))
 for rid in sorted(set(route_ids)):
  k=route_ids.index(rid);name=protocol['route_names'][rid] if 'route_names' in protocol else f'route {rid}';ax.plot(routes[rid,:,0],routes[rid,:,1],ls='--',lw=2,color=colors[k%n],label=name+' reference')
 for k,case in enumerate(cases):
  xy=root[valid[:,k],k,:2];ax.plot(xy[:,0],xy[:,1],color=colors[k%n],alpha=.45,lw=.8);ax.scatter(*xy[-1],color=colors[k%n],marker='o' if case['success'] else 'x',s=14)
 ax.set(xlim=(-.5,20),ylim=(-4.2,4.2),xlabel='World x (m)',ylabel='World y (m)',title='One physical map / one goal; all measured trajectories; '+('online decisions' if planning.exists() or moving_path.exists() else 'enumerated candidates'))
 ax.set_aspect('equal');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);fig.savefig(a.output/'map_and_actual_xy.png',dpi=160,facecolor='white');plt.close(fig)
index=[]
for i in range(len(success)):

 fig,axes=plt.subplots(4,1,figsize=(12,10),layout='constrained')
 for j in range(n):
  k=i*n+j;m=valid[:,k];x=root[m,k];r=reward[m,k];steps=np.arange(1,len(x)+1);label=f"{names[j]} ({'goal' if cases[k]['success'] else 'end '+str(cases[k]['end_code'])})"
  axes[0].plot(x[:,0],x[:,1],color=colors[j],ls=styles[j],label=label);axes[0].scatter(*x[-1,:2],marker='o' if cases[k]['success'] else 'x',color=colors[j])
  for points in reference_segments(k,routes):axes[0].plot(points[:,0],points[:,1],color=colors[j],ls=':',alpha=.4,lw=.7)
  if k in switches:axes[0].scatter(*switches[k]['published_root'][:2],marker='D',color=colors[j],s=20)
  axes[1].plot(steps,r,color=colors[j],ls=styles[j]);axes[2].plot(steps,r.cumsum(),color=colors[j],ls=styles[j]);axes[3].plot(steps,x[:,7],color=colors[j],ls=styles[j])
 axes[0].set(xlabel='World x (m)',ylabel='World y (m)');axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(ncol=2,fontsize=8)
 axes[1].set(ylabel='Per-step reward',xlabel='Policy step (20 ms)');axes[2].set(ylabel='Cumulative reward',xlabel='Policy step (20 ms)');axes[3].set(ylabel='World vx (m/s)',xlabel='Policy step (20 ms)')
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle(f"Root {i} | {protocol['cases'][i*n]['role'] if 'cases' in protocol else 'Online paired execution'} | heading125/H75 | all declared methods; true endpoints")
 name=f'root_{i:03d}.png';fig.savefig(a.output/name,dpi=130,facecolor='white');plt.close(fig);index.append(f'- Root {i}: [{name}]({name})')
(a.output/'README.md').write_text('# 完整轨迹与奖励索引\n\n'+f'原始数据：`{a.collection}/evaluation/trajectories.npz`。全部{len(cases)}条，不筛除失败；50Hz原始数据，无平滑。XY等比例；圆点为合格目标，叉为真实终点。候选各自原道路与发布参考不同，不把其中之一叫纯ECBC基线。\n\n'+ '\n'.join(index)+'\n')
print(a.output/'xy_all_candidates.png')
