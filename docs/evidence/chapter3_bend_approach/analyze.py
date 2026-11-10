from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_bend_approach');out=Path(__file__).parent
fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained');rows=[];arrays={}
for i,route in enumerate(['straight','left','right']):
 ref=np.load(base/(route+'.npz'));xy=ref['xy'];mask=(xy[:,0]>=1.5)&(xy[:,0]<=16)
 axes[i,0].plot(*xy[mask].T,'k:',label='Fixed reference');axes[i,0].axvspan(6.3,6.6,color='0.85');axes[i,0].set_aspect('equal',adjustable='box');axes[i,0].set(xlim=(1,16),ylim=(-2.5,2.5),yticks=[-2,0,2])
 arrays[route+'_reference']=xy
 for suffix,color,style in [('v20','#0072B2','-'),('v18','#D55E00','--')]:
  name=route+'_'+suffix;p=base/name;s=json.loads((p/'summary.json').read_text())['cases'][0];d=np.load(p/'heading125/trajectory.npz');r=d['root'][:,0];t=d['time'];rew=d['reward'][:,0];entry=np.load(p/'approach_entry.npz');hand=json.loads((p/'handoff.json').read_text()) if (p/'handoff.json').exists() else None
  label='2.0 m/s' if suffix=='v20' else '1.8 m/s'
  for ax,x,y in [(axes[i,0],r[:,0],r[:,1]),(axes[i,1],t,r[:,7]),(axes[i,2],t,rew),(axes[i,3],t,np.cumsum(rew))]:
   ax.plot(x,y,color=color,ls=style,label=label);ax.plot(x[-1],y[-1],marker='o' if s['ground_continued_to_horizon'] else 'x',color=color)
  row=dict(case=name,duration_s=s['duration_s'],end_code=s['terminal_code'],final_xyz=s['final_xyz'],handoff_s=hand['time_s'] if hand else None,continued=s['ground_continued_to_horizon'],entry_time_s=float(entry['time_s']),entry_root=entry['root'][0].tolist(),reward_total=s['reward_total'],max_abs_y_m=float(np.abs(r[:,1]).max()));rows.append(row)
  arrays.update({name+'_time':t,name+'_root':r,name+'_reward':rew})
 for j,(title,xlabel,ylabel) in enumerate([('Actual XY','World x (m)','World y (m)'),('Forward speed','Time (s)','World vx (m/s)'),('Diagnostic reward','Time (s)','Reward / step'),('Cumulative reward','Time (s)','Reward')]):
  ax=axes[i,j];ax.set(title=route+' | '+title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Isaac heading125 + H75 | alpha 0.5 | seed 20261109 | nominal DEV | o: horizon, x: failure\nSame reset; 30cm flat ditch; original support reward remains diagnostic; no training')
fig.savefig(out/'xy_and_rewards.png',dpi=160)
np.savez_compressed(out/'plot_data.npz',**arrays);(out/'summary.json').write_text(json.dumps(rows,indent=2))
print([(r['case'],r['handoff_s'],r['end_code']) for r in rows])
