"""Offline all-alpha nominal comparison with authenticated cached traces."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'learning/src'))
from sttw_control.preference_best import atomic_json,digest
from sttw_control.preference_tail_evaluation import cached_policy_panel

def main():
 cache=json.loads((ROOT/'cached_controls.json').read_text());plan=json.loads((ROOT/'declaration.json').read_text())
 rows=[];traces={};source=[]
 for panel in ('moderate','straight'):
  run=ROOT/panel/'evaluation';decl=json.loads((ROOT/panel/'declaration.json').read_text())
  status=json.loads((run/'status.json').read_text())
  if status['phase']!='completed':raise RuntimeError('incomplete panel')
  sets={'coverage':(json.loads((run/'results.json').read_text())['cases'],run)}
  for alias in ('parent','oldfull10'):
   r=cache[panel][alias]['receipt'];authenticated=cached_policy_panel(r['run'],decl,plan['controls'][alias],r['source_policy'],alias)
   if authenticated['receipt']['results_sha256']!=r['results_sha256']:raise RuntimeError('cache changed')
   sets[alias]=(authenticated['rows'],Path(r['run']))
  cfg=decl['config'];prefix=cfg['path']['straight_prefix_m'];end=prefix+abs(cfg['path']['turn_angle_rad'])*cfg['path']['turn_radius_m']+cfg['path']['curvature_transition_length_m']
  for alias,(group,location) in sets.items():
   for row in group:
    if row['force_N']!=0 or row['onset_s']!=min(decl['onsets_s']):continue
    p=location/(row['name']+'.npz')
    with np.load(p) as z:d={k:z[k] for k in z.files}
    mask=np.ones(len(d['time']),bool) if row['route'].startswith('straight') else (d['path_progress']>=prefix)&(d['path_progress']<=end)
    if not mask.any():raise RuntimeError('missing turn samples')
    r=dict(row,turn_speed_rmse=float(np.sqrt(np.mean((d['actual_forward_speed'][mask]-row['speed'])**2))),turn_path_rmse=float(np.sqrt(np.mean(d['path_cross_track'][mask]**2))))
    rows.append(r);key=(alias,row['route'],row['speed'],row['alpha']);traces[key]=d;source.append(dict(key=key,path=str(p),sha256=digest(p)))
 if len(rows)!=90 or len(traces)!=90:raise RuntimeError('expected 30 unique nominal cases per policy')
 summary=[]
 for alias in ('parent','oldfull10','coverage'):
  group=[r for r in rows if r['policy']==alias];turns=[]
  for route,speed in sorted({(r['route'],r['speed']) for r in group if not r['route'].startswith('straight')}):
   g=sorted([r for r in group if (r['route'],r['speed'])==(route,speed)],key=lambda r:r['alpha'])
   v=[r['turn_speed_rmse'] for r in g];p=[r['turn_path_rmse'] for r in g]
   turns.append(dict(route=route,speed=speed,alphas=[r['alpha'] for r in g],speed_rmse=v,path_rmse=p,speed_monotone=all(a>b for a,b in zip(v,v[1:])),speed_endpoint_gain=v[0]-v[-1],path_endpoint_gain=p[-1]-p[0]))
  summary.append(dict(policy=alias,cases=len(group),task=sum(r['task_recovered'] for r in group),working=sum(r['performance_success'] for r in group),turns=turns))
 out=ROOT/'nominal_review';out.mkdir(exist_ok=False);atomic_json(out/'summary.json',dict(summary=summary,cases=rows,sources=source,interpretation='unique zero-force contexts; all five alphas, own closed loops; fixed-alpha DEV, not online switching'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 contexts=sorted({(r['route'],r['speed']) for r in rows});alphas=sorted({r['alpha'] for r in rows})
 for field,title in [('actual_xy','Actual XY [m]'),('scored_tick_reward','Reward per 5ms'),('cumulative','Cumulative reward')]:
  fig,axes=plt.subplots(len(contexts),len(alphas),figsize=(22,3.3*len(contexts)),squeeze=False,layout='constrained')
  for i,(route,speed) in enumerate(contexts):
   for j,alpha in enumerate(alphas):
    ax=axes[i,j]
    for alias in ('parent','oldfull10','coverage'):
     d=traces[(alias,route,speed,alpha)]
     if field=='actual_xy':ax.plot(*d[field].T,label=alias)
     else:ax.plot(d['time'],np.cumsum(d['scored_tick_reward']) if field=='cumulative' else d[field],label=alias)
    if field=='actual_xy':ax.plot(*d['reference_xy'].T,'k--',label='reference');ax.set_aspect('equal',adjustable='datalim')
    ax.set_title(f'{route} {speed:g}m/s a={alpha:g}',fontsize=8);ax.grid(alpha=.2);ax.legend(fontsize=6)
  fig.suptitle(title+' | all unique nominal contexts');fig.savefig(out/(field+'.png'),dpi=105);plt.close(fig)
 print(json.dumps(summary))
if __name__=='__main__':main()
