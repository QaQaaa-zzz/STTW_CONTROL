from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
if not (R/'numeric_results.json').exists():raise RuntimeError('numerical panels incomplete')
old=json.loads((R.parent/'preference_coordination_teaching_standard_dev_v1/nominal_review/summary.json').read_text());rows=[];paths={}
for r in old['cases']:
 if r['policy'] not in ('parent','taught'):continue
 alias='seed918501' if r['policy']=='taught' else 'parent';row=dict(r,policy=alias);rows.append(row)
 source=next(x for x in old['sources'] if tuple(x['key'])==(r['policy'],r['route'],r['speed'],r['alpha']));p=Path(source['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256'];paths[alias,r['route'],r['speed'],r['alpha']]=p
for panel in ('moderate','straight'):
 run=R/panel/'evaluation';d=json.loads((R/panel/'declaration.json').read_text());assert json.loads((run/'status.json').read_text())['phase']=='completed';c=d['config']['path'];start=c['straight_prefix_m'];end=start+abs(c['turn_angle_rad'])*c['turn_radius_m']+c['curvature_transition_length_m']
 for row in json.loads((run/'results.json').read_text())['cases']:
  if row['force_N'] or row['onset_s']!=min(d['onsets_s']):continue
  p=run/(row['name']+'.npz')
  with np.load(p) as z:
   mask=np.ones(len(z['time']),bool) if row['route'].startswith('straight') else (z['path_progress']>=start)&(z['path_progress']<=end)
   row=dict(row,turn_speed_rmse=float(np.sqrt(np.mean((z['actual_forward_speed'][mask]-row['speed'])**2))),turn_path_rmse=float(np.sqrt(np.mean(z['path_cross_track'][mask]**2))))
  rows.append(row);paths[row['policy'],row['route'],row['speed'],row['alpha']]=p
assert len(rows)==len(paths)==120
summaries=[];aliases=('parent','seed918501','seed918502','seed918503')
for alias in aliases:
 g=[r for r in rows if r['policy']==alias];turns=[]
 for route,speed in sorted({(r['route'],r['speed']) for r in g if not r['route'].startswith('straight')}):
  group=sorted([r for r in g if (r['route'],r['speed'])==(route,speed)],key=lambda r:r['alpha']);v=[r['turn_speed_rmse'] for r in group];p=[r['turn_path_rmse'] for r in group];turns.append(dict(route=route,speed=speed,speed_rmse=v,path_rmse=p,speed_monotone=all(a>b for a,b in zip(v,v[1:])),path_endpoint_gain=p[-1]-p[0]))
 summaries.append(dict(policy=alias,task=sum(r['task_recovered'] for r in g),working=sum(r['performance_success'] for r in g),turns=turns))
out=R/'nominal_review';out.mkdir(exist_ok=False);sources=[dict(key=k,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()];(out/'summary.json').write_text(json.dumps(dict(summary=summaries,cases=rows,sources=sources),indent=2)+'\n')
contexts=sorted({(r['route'],r['speed']) for r in rows});alphas=sorted({r['alpha'] for r in rows})
for field in ('actual_xy','scored_tick_reward','cumulative'):
 fig,axes=plt.subplots(6,5,figsize=(22,20),layout='constrained')
 for i,(route,speed) in enumerate(contexts):
  for j,alpha in enumerate(alphas):
   ax=axes[i,j]
   for alias in aliases:
    with np.load(paths[alias,route,speed,alpha]) as z:
     if field=='actual_xy':ax.plot(*z[field].T,label=alias);ref=z['reference_xy']
     else:ax.plot(z['time'],np.cumsum(z['scored_tick_reward']) if field=='cumulative' else z[field],label=alias)
   if field=='actual_xy':ax.plot(*ref.T,'k--',label='reference');ax.set_aspect('equal',adjustable='datalim')
   ax.set_title(f'{route} {speed:g} a={alpha:g}',fontsize=8);ax.legend(fontsize=5);ax.grid(alpha=.2)
 fig.savefig(out/(field+'.png'),dpi=100);plt.close(fig)
for r in summaries:print(r['policy'],r['task'],r['working'],sum(t['speed_monotone'] for t in r['turns']))
