"""Saved-trace diagnostic; executed commands are not latent policy actions."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
cache=json.loads((R/'cached_controls.json').read_text())['low_speed']
rows=json.loads((R/'low_speed/evaluation/results.json').read_text())['cases']+cache['parent']['rows']+cache['oldfull10']['rows']
keys=json.loads((R/'low_speed/evaluation/manifest.json').read_text())['reward_components']
rows=[r for r in rows if r['speed']==2.35 and r['alpha'] in (0,.125) and r['force_N']!=0]
contexts=sorted({(r['route'],r['alpha'],r['force_N']) for r in rows})
fig,axs=plt.subplots(8,3,figsize=(18,26),layout='constrained');records=[]
for r in rows:
 loc=R/'low_speed/evaluation' if r['policy']=='coverage' else Path(cache[r['policy']]['receipt']['run'])
 with np.load(loc/(r['name']+'.npz')) as z:
  t=z['time'];offset=z['governed']-z['nominal'];i=contexts.index((r['route'],r['alpha'],r['force_N']))
  for j in (0,1):axs[i,j].plot(t,offset[:,j],label=r['policy'])
  axs[i,2].plot(t,np.cumsum(z['scored_tick_reward']),label=r['policy'])
  for window,mask in [('post_pulse',t>=2.1),('tail',t>=9)]:
   e=z['effective_costs'][mask];total=e.sum()
   records.append(dict(policy=r['policy'],route=r['route'],alpha=r['alpha'],force=r['force_N'],window=window,task=r['task_recovered'],mean_offset=offset[mask].mean(0).tolist(),peak_abs_offset=np.max(abs(offset[mask]),axis=0).tolist(),cost_share={k:float(e[:,j].sum()/total) for j,k in enumerate(keys)},return_sum=float(z['scored_tick_reward'][mask].sum())))
for i,c in enumerate(contexts):
 for j,label in enumerate(('Executed speed offset [m/s]','Executed steering offset [rad]','Cumulative logged reward')):
  a=axs[i,j];a.set_title(f'{c} | {label}',fontsize=8);a.legend(fontsize=7);a.grid(alpha=.2);a.axvspan(2,2.1,color='gray',alpha=.2)
fig.savefig(R/'matched_curves/command_reward_diagnostic.png',dpi=100);plt.close(fig)
(R/'command_reward_diagnostic.json').write_text(json.dumps(dict(records=records,limitations='No latent action or clipping flags saved in standard screens; executed offset is governed minus nominal. Reward shares are descriptive, not causal attribution. Reward comparisons only within same alpha/context.'),indent=2)+'\n')
for a in ('parent','oldfull10','coverage'):
 g=[r for r in records if r['policy']==a and r['window']=='tail' and not r['task']]
 print(a,'mean tail cost shares', {k:round(float(np.mean([r['cost_share'][k] for r in g])),3) for k in keys})
