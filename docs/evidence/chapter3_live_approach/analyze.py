"""Read actual common-start runs; preserve all declared candidates."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_isaac_live_approach');out=Path(__file__).parent
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');rows=[];arrays={}
for name,label,color in [('evaluation','v2.0 alpha0.5','#0072B2'),('speed18','v1.8 alpha0.5','#D55E00'),('alpha0','v2.0 alpha0','#009E73')]:
 d=np.load(base/name/'heading125/trajectory.npz');t=d['time'];r=d['root'][:,0];rew=d['reward'][:,0];entry=np.load(base/name/'approach_entry.npz');s=json.loads((base/name/'summary.json').read_text())['cases'][0]
 hand=json.loads((base/name/'handoff.json').read_text()) if (base/name/'handoff.json').exists() else None
 rows.append(dict(case=name,request=label,entry_time_s=float(entry['time_s']),entry_root=entry['root'][0].tolist(),handoff_time_s=hand['time_s'] if hand else None,duration_s=s['duration_s'],end_code=s['terminal_code'],final_xyz=s['final_xyz'],stage=s['verified_stage'],reward=s['reward_total'],continued=s['ground_continued_to_horizon']))
 axes[0,0].plot(r[:,0],r[:,1],color=color,label=label);axes[0,0].plot(*r[-1,:2],'x',color=color)
 axes[0,1].plot(t,r[:,7],color=color,label=label);axes[1,0].plot(t,rew,color=color,label=label);axes[1,1].plot(t,np.cumsum(rew),color=color,label=label)
 arrays.update({name+'_time':t,name+'_root':r,name+'_reward':rew})
axes[0,0].axhline(0,color='k',ls=':',label='fixed y=0 reference');axes[0,0].axvspan(3.3,3.6,color='0.8');axes[0,0].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY');axes[0,0].set_aspect('equal',adjustable='box')
for ax,title,y in [(axes[0,1],'Actual world-forward speed','vx (m/s)'),(axes[1,0],'Original diagnostic reward','reward/step'),(axes[1,1],'Original cumulative reward','reward')]:ax.set(xlabel='Time (s)',ylabel=y,title=title)
for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,0].legend(loc='lower left',bbox_to_anchor=(0,1.02),fontsize=7)
fig.suptitle('Isaac: live H75 approach -> heading125 -> H75 | same reset, nominal DEV, no training')
fig.savefig(out/'xy_and_rewards.png',dpi=150)
(out/'summary.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(out/'plot_data.npz',**arrays)
print(json.dumps(rows,indent=2))
