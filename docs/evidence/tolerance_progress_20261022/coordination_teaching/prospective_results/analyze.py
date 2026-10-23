"""Frozen prospective panel: paired outcomes and unique nominal preferences."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;plan=json.loads((R/'declaration.json').read_text())
if not (R/'numeric_results.json').exists():raise RuntimeError('all numerical panels must finish')
key=lambda r:tuple(r[k] for k in ('route','speed','alpha','force_N','onset_s'))
nominal=[];paired=[];traces={};sources=[]
for panel in plan['panels']:
 folder=R/panel;decl=json.loads((folder/'declaration.json').read_text());assert hashlib.sha256((folder/'declaration.json').read_bytes()).hexdigest()==plan['panel_sha256'][panel]
 run=folder/'evaluation';assert json.loads((run/'status.json').read_text())['phase']=='completed';rows=json.loads((run/'results.json').read_text())['cases'];baseline={key(r):r for r in rows if r['policy']=='parent'}
 c=decl['config']['path'];start=c['straight_prefix_m'];end=start+abs(c['turn_angle_rad'])*c['turn_radius_m']+c['curvature_transition_length_m']
 for r in rows:
  b=baseline[key(r)]
  if r['policy']!='parent':paired.append(dict(panel=panel,policy=r['policy'],context=key(r),task=r['task_recovered'],parent_task=b['task_recovered'],gain=bool(r['task_recovered'] and not b['task_recovered']),loss=bool(b['task_recovered'] and not r['task_recovered']),domain=r['domain_failure'],parent_domain=b['domain_failure'],peak_roll=r['peak_roll_rad'],parent_peak_roll=b['peak_roll_rad'],tail_path=r['tail_max_path_m'],parent_tail_path=b['tail_max_path_m'],working=r['performance_success'],parent_working=b['performance_success']))
  if r['force_N'] or r['onset_s']!=min(decl['onsets_s']):continue
  p=run/(r['name']+'.npz')
  with np.load(p) as z:
   d={k:z[k] for k in z.files};mask=(d['path_progress']>=start)&(d['path_progress']<=end)
   assert mask.any();nominal.append(dict(r,turn_speed_rmse=float(np.sqrt(np.mean((d['actual_forward_speed'][mask]-r['speed'])**2))),turn_path_rmse=float(np.sqrt(np.mean(d['path_cross_track'][mask]**2)))))
  k=(r['policy'],r['route'],r['speed'],r['alpha']);traces[k]=d;sources.append(dict(key=k,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert len(nominal)==len(traces)==128
summaries=[]
for alias in plan['policies']:
 group=[r for r in nominal if r['policy']==alias];turns=[]
 for route,speed in sorted({(r['route'],r['speed']) for r in group}):
  g=sorted([r for r in group if (r['route'],r['speed'])==(route,speed)],key=lambda r:r['alpha']);v=[r['turn_speed_rmse'] for r in g];p=[r['turn_path_rmse'] for r in g];turns.append(dict(route=route,speed=speed,alphas=[r['alpha'] for r in g],speed_rmse=v,path_rmse=p,speed_monotone=all(a>b for a,b in zip(v,v[1:])),path_endpoint_gain=p[-1]-p[0]))
 summaries.append(dict(policy=alias,nominal_task=sum(r['task_recovered'] for r in group),nominal_working=sum(r['performance_success'] for r in group),turns=turns))
out=R/'analysis';out.mkdir(exist_ok=False);(out/'summary.json').write_text(json.dumps(dict(nominal_summary=summaries,nominal_cases=nominal,paired=paired,sources=sources),indent=2)+'\n')
contexts=sorted({(r['route'],r['speed']) for r in nominal});alphas=sorted({r['alpha'] for r in nominal})
for field in ('actual_xy','scored_tick_reward','cumulative'):
 fig,axes=plt.subplots(8,4,figsize=(22,28),layout='constrained')
 for i,(route,speed) in enumerate(contexts):
  for j,alpha in enumerate(alphas):
   ax=axes[i,j]
   for alias in plan['policies']:
    d=traces[alias,route,speed,alpha]
    if field=='actual_xy':ax.plot(*d[field].T,label=alias);ax.plot(*d[field][-1],marker='x',ms=3)
    else:ax.plot(d['time'],np.cumsum(d['scored_tick_reward']) if field=='cumulative' else d[field],label=alias)
   if field=='actual_xy':ax.plot(*d['reference_xy'].T,'k--',label='reference');ax.set_aspect('equal',adjustable='datalim')
   ax.set_title(f'{route} {speed:g} a={alpha:g}',fontsize=8);ax.legend(fontsize=5);ax.grid(alpha=.2)
 fig.savefig(out/(field+'.png'),dpi=100);plt.close(fig)
print('nominal',[(r['policy'],r['nominal_task'],r['nominal_working'],sum(t['speed_monotone'] for t in r['turns'])) for r in summaries])
for panel in plan['panels']:
 for alias in list(plan['policies'])[1:]:
  g=[r for r in paired if r['panel']==panel and r['policy']==alias];print(panel,alias,'gains',sum(r['gain'] for r in g),'losses',sum(r['loss'] for r in g))
