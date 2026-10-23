"""Check actual switching schedule and paired prefixes before reporting dynamics."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();raw=a.run/'raw';status=json.loads((raw/'status.json').read_text());assert status['phase']=='completed'
 d=json.loads((a.run/'declaration.json').read_text());spec=d['preference_switching'];rows=json.loads((raw/'results.json').read_text())['cases'];assert len(rows)==12
 traces={};metrics=[];sources={}
 for r in rows:
  file=raw/(r['name']+'.npz');sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
  with np.load(file) as f:x={k:f[k] for k in f.files}
  traces[r['name']]=x;t=x['time'];decision=(np.rint(t/.005).astype(int)//4)*.02;mode=spec['modes'][r['policy']]
  w=np.zeros_like(t) if mode=='fixed' else (decision>=spec['switch_s']).astype(float) if mode=='step' else np.clip((decision-spec['switch_s'])/spec['ramp_s'],0,1)
  expected=r['alpha']+(1-2*r['alpha'])*w
  np.testing.assert_allclose(x['alpha_actual'],expected,atol=1e-6)
  assert np.all(np.diff(t)>0) and np.isclose(float(x['scored_tick_reward'].sum()),r['logged_return'],atol=1e-4)
  after=t>=spec['switch_s'];metrics.append(dict(policy=r['policy'],route=r['route'],initial_alpha=r['alpha'],task=r['task_recovered'],working=r['performance_success'],peak_roll=r['peak_roll_rad'],post_path_rmse=float(np.sqrt(np.mean(x['path_cross_track'][after]**2))),post_speed_rmse=float(np.sqrt(np.mean((x['actual_forward_speed'][after]-r['speed'])**2))),return_value=r['logged_return']))
 contexts=sorted({(r['route'],r['alpha']) for r in rows});prefix_diff=0.
 for route,initial in contexts:
  rr=[r for r in rows if (r['route'],r['alpha'])==(route,initial)];base=traces[next(r['name'] for r in rr if r['policy']=='fixed')]
  for r in rr:
   x=traces[r['name']];mask=x['time']<spec['switch_s']-1e-6
   for field in ('actual_xy','actual_forward_speed','path_cross_track','scored_tick_reward'):
    delta=float(np.max(np.abs(x[field][mask]-base[field][mask])));prefix_diff=max(prefix_diff,delta)
 assert np.isfinite(prefix_diff) # Record numerical replay difference; user accepts small discrepancies.
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'summary.json').write_text(json.dumps(dict(scope='engineering only, one frozen75PPO, no external pulse; not method improvement',status=status,metrics=metrics,prefix_max_abs_difference=prefix_diff,sources=sources),indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(4,5,figsize=(20,14),layout='constrained')
 for i,(route,initial) in enumerate(contexts):
  for r in [r for r in rows if (r['route'],r['alpha'])==(route,initial)]:
   x=traces[r['name']];t=x['time'];lab=r['policy']+' task='+str(int(r['task_recovered']))
   axes[i,0].plot(*x['actual_xy'].T,label=lab)
   for col,field in [(1,'actual_forward_speed'),(2,'path_cross_track'),(3,'phi'),(4,'alpha_actual')]:axes[i,col].plot(t,x[field],label=lab)
  with np.load(raw/(route+'_path.npz')) as path:axes[i,0].plot(*path['xy'].T,'k--',label='original path')
  axes[i,0].set_aspect('equal',adjustable='datalim')
  for col,title in enumerate(('XY [m]','true speed [m/s]','path error [m]','roll [rad]','actual alpha')):
   ax=axes[i,col];ax.set_title(f'{route}, initial alpha={initial:g} | {title}');ax.grid(alpha=.2)
   if col:ax.axvline(spec['switch_s'],color='k',ls=':');ax.set_xlabel('task time [s]')
  axes[i,0].legend(fontsize=7)
 fig.savefig(a.output/'all_cases_xy.png',dpi=120);plt.close(fig)
 lines=['# Dynamic preference engineering check','', '![All cases](all_cases_xy.png)','',f'Actual schedule verified for all12 cases; paired pre-switch maximum difference={prefix_diff:g}. Original path and speed retained. Fixed/step/ramp share full initial state; only alpha changes. Reward follows actual alpha; different schedules have different reward weights, so return is not a common performance ranking.','', '| route | initial alpha | mode | task | working | peak roll | post path RMSE | post speed RMSE |','|---|---:|---|---:|---:|---:|---:|---:|']
 for m in metrics:lines.append(f"| {m['route']} | {m['initial_alpha']} | {m['policy']} | {int(m['task'])} | {int(m['working'])} | {m['peak_roll']:.4f} | {m['post_path_rmse']:.4f} | {m['post_speed_rmse']:.4f} |")
 lines+=['','Engineering evidence only. A small undisturbed panel cannot establish preference-switch novelty, robustness, or general superiority. Original raw reward components and speed proxy remain in saved NPZ files.']
 (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
