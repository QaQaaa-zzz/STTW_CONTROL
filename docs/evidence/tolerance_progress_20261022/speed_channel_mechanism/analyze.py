"""Paired intervention-window mechanics from already acquired TRAIN traces."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
records=[]
for name in ['tolerance_teaching_search_pilot_v1','tolerance_recovery_opportunity_v1']:
 root=ROOT/'runs'/name;d=json.loads((root/'declaration.json').read_text());spec=d['recovery_intervention']
 rows=json.loads((root/'evaluation/results.json').read_text())['cases']
 key=lambda r:tuple(r[k] for k in ['route','speed','alpha','force_N','onset_s'])
 lookup={(r['policy'],key(r)):r for r in rows}
 for base in rows:
  if base['policy']!='control' or base['force_N']==0:continue
  other=lookup['speed_off',key(base)];metrics={};sources={}
  for arm,row in [('control',base),('speed_off',other)]:
   path=root/'evaluation'/(row['name']+'.npz');sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
   with np.load(path) as z:
    mask=(z['time']>=spec['start_s']-1e-5)&(z['time']<spec['end_s']-1e-5)
    metrics[arm]=dict(ticks=int(mask.sum()),target_speed_offset=float(np.mean(z['target_offset'][mask,0])),governed_speed=float(np.mean(z['governed'][mask,0])),actual_speed=float(np.mean(z['actual_forward_speed'][mask])),path_rms=float(np.sqrt(np.mean(z['path_cross_track'][mask]**2))),roll_peak=float(np.max(z['peak_roll'][mask])))
  kind='gain' if not base['task_recovered'] and other['task_recovered'] else 'loss' if base['task_recovered'] and not other['task_recovered'] else 'same'
  records.append(dict(run=name,context=key(base),kind=kind,useful=kind=='gain' and other['logged_return']>base['logged_return'] and not other['physical_failure'] and not other['domain_failure'],metrics=metrics,sources=sources,full_window=all(m['ticks']==round((spec['end_s']-spec['start_s'])/.005) for m in metrics.values())))
summary={}
for kind in ['gain','loss','same']:
 rr=[r for r in records if r['kind']==kind];complete=[r for r in rr if r['full_window']]
 summary[kind]=dict(n=len(rr),complete_window_n=len(complete),control_negative_target=sum(r['metrics']['control']['target_speed_offset']<0 for r in complete),actual_speed_increases=sum(r['metrics']['speed_off']['actual_speed']>r['metrics']['control']['actual_speed'] for r in complete),mean_delta_actual_speed=float(np.mean([r['metrics']['speed_off']['actual_speed']-r['metrics']['control']['actual_speed'] for r in complete])) if complete else None)
(OUT/'analysis.json').write_text(json.dumps(dict(records=records,summary=summary,limitations='Posthoc TRAIN mechanistic description, not independent validation. Full-window summaries explicitly exclude early terminal traces; all cases retained.'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for kind in ['gain','loss']:
 r=next(x for x in records if x['kind']==kind and x['full_window']);fig,axes=plt.subplots(2,3,figsize=(14,8))
 for path in r['sources']:
  label='speed_off' if Path(path).name.startswith('speed_off_') else 'control'
  with np.load(path) as z:
   t=z['time'];xy=z['actual_xy'];axes[0,0].plot(xy[:,0],xy[:,1],label=label)
   axes[0,1].plot(t,z['actual_forward_speed'],label=label)
   axes[0,2].plot(t,z['target_offset'][:,0],label=label)
   axes[1,0].plot(t,z['path_cross_track'],label=label)
   axes[1,1].plot(t,z['phi'],label=label)
   axes[1,2].plot(t,np.cumsum(z['scored_tick_reward']),label=label)
   if label=='control':
    axes[0,0].plot(z['reference_xy'][:,0],z['reference_xy'][:,1],'k--',label='original reference')
    axes[0,1].axhline(r['context'][1],color='k',ls='--',lw=.8)
 axes[0,0].set_aspect('equal',adjustable='box');axes[0,0].set_xlabel('x (m)');axes[0,0].set_ylabel('y (m)')
 titles=['Actual XY','Actual forward speed (m/s)','Target speed correction (m/s)','Cross-track error (m)','Roll (rad)','Cumulative original reward']
 for ax,title in zip(axes.flat,titles):ax.set_title(title);ax.legend(fontsize=7)
 d=json.loads((ROOT/'runs'/r['run']/'declaration.json').read_text())
 for ax in list(axes.flat)[1:]:
  ax.set_xlabel('time (s)');ax.axvspan(d['recovery_intervention']['start_s'],d['recovery_intervention']['end_s'],alpha=.08,color='grey')
 fig.suptitle(f"First complete-window {kind} in source order: {r['context']}\n{r['run']}; grey = intervention; original task unchanged")
 fig.tight_layout();fig.savefig(OUT/f'{kind}_example.png',dpi=130);plt.close(fig)
