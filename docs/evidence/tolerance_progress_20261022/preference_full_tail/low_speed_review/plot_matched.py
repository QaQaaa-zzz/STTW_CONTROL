"""Offline matched target-subset traces; no additional simulation."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'matched_results.json').read_text())['cases']
cache=json.loads((ROOT/'cached_parent.json').read_text())['receipt'];locations={'parent':Path(cache['run']),'full10':ROOT/'evaluation'}
key=lambda r:tuple(r[k] for k in ('route','speed','alpha','force_N','onset_s'))
index={(r['policy'],key(r)):r for r in rows}
contexts=sorted({key(r) for r in rows if r['speed']==2.35 and r['alpha'] in (0.,.125) and r['force_N']!=0})
assert len(contexts)==8
out=ROOT/'matched_curves';out.mkdir(exist_ok=False)
for field,title in [('actual_xy','Actual XY [m]'),('path_cross_track','Cross-track error [m]'),('path_heading_error','Heading error [rad]'),('actual_forward_speed','Forward speed [m/s]'),('scored_tick_reward','Reward per 5ms'),('cumulative','Cumulative reward')]:
 fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
 for ax,context in zip(axes.flat,contexts):
  route,speed,alpha,force,onset=context
  for policy in ('parent','full10'):
   row=index[policy,context];z=np.load(locations[policy]/(row['name']+'.npz'))
   if field=='actual_xy':
    ax.plot(*z[field].T,label=policy);ax.plot(*z[field][-1],marker='x')
   else:ax.plot(z['time'],np.cumsum(z['scored_tick_reward']) if field=='cumulative' else z[field],label=policy)
  if field=='actual_xy':
   ax.plot(*z['reference_xy'].T,'k--',label='reference');ax.set_aspect('equal',adjustable='datalim')
  else:
   ax.axvspan(onset,onset+.1,color='gray',alpha=.25);ax.set_xlabel('original task time [s]')
   if field=='actual_forward_speed':ax.axhline(speed,color='k',ls='--')
  ax.set_title(f'{route} a={alpha:g} F={force:g}N',fontsize=9);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle(title+' | prespecified low-speed target subset, actual endpoints')
 fig.savefig(out/(field+'.png'),dpi=110);plt.close(fig)
print('six paired trace figures saved')
