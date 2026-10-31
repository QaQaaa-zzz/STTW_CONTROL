"""Plot all frozen paired TEST cases without smoothing or outcome exclusions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');a=p.parse_args();root=Path(a.run)
results=json.loads((root/'results.json').read_text());names=list(results);out=root/'analysis';out.mkdir(exist_ok=True)
lines=['# Frozen lower-controller physical-domain TEST','', 'Local speed/steer command task; dashed XY reference is command integration, not a geometric path objective. No smoothing. Each trace ends at physical failure or 10 seconds. Initial-state excitation ends before this observation window.','']
config=json.loads((root/'config.json').read_text());spec=json.loads(Path(config['controller_config']).read_text())
summary={};paired=[];audits=[]
def reconstruct(d):
 r=spec['reward'];h=lambda x:np.where(abs(x)<=1,x*x,2*abs(x)-1)
 action=d['action'];previous=np.concatenate([np.zeros((1,2)),action[:-1]])
 costs=dict(speed=r['speed_weight']*h((d['actual_forward_speed']-d['limited_command'][:,0])/r['speed_huber_scale']),steer=r['steer_weight']*h((d['actual_delta']-d['limited_command'][:,1])/r['steer_huber_scale']),roll=r['roll_weight']*h(np.maximum(d['peak_roll']-r['roll_start'],0)/r['roll_scale']),working_roll=r['working_roll_weight']*h(np.maximum(d['peak_roll']-r['working_roll_start'],0)/r['working_roll_scale']),roll_rate=r['roll_rate_weight']*h(np.maximum(abs(d['phi_dot'])-r['roll_rate_start'],0)/r['roll_rate_scale']),action=r['action_weight']*np.sum(action**2,axis=1),action_change=r['action_change_weight']*np.sum((action-previous)**2,axis=1))
 failed=d['physical_failure'];terms={k:np.where(failed,0,-r['scale']*spec['dt']*np.minimum(v,r['caps'][k])) for k,v in costs.items()}
 remaining=2000-np.arange(len(action));gamma=spec['gamma'];tail=remaining if gamma==1 else -np.expm1(remaining*np.log(gamma))/(1-gamma)
 terms['failure']=np.where(failed,-r['failure_extra']-r['scale']*spec['dt']*r['failure_cost_rate_bound']*tail,0)
 for k,v in terms.items():np.testing.assert_allclose(v,d['reward_parts_'+k],rtol=3e-5,atol=2e-6)
 np.testing.assert_allclose(sum(terms.values()),d['reward'],rtol=3e-5,atol=2e-6)
 return terms

for name,rows in results.items():
 summary[name]=dict(cases=len(rows),survived=sum(not x['failed'] for x in rows),mean_case_speed_rmse=float(np.mean([x['speed_rmse'] for x in rows])),mean_case_steer_rmse_rad=float(np.mean([x['steer_rmse'] for x in rows])))
 lines.append(f'- {name}: `{json.dumps(summary[name])}`')
lines+=['','## All paired trajectories and rewards','']
for i in range(len(results[names[0]])):
 ds=[dict(np.load(root/name/f'case_{i:02d}.npz')) for name in names];n=min(len(d['time']) for d in ds)
 np.testing.assert_allclose(ds[0]['limited_command'][:n],ds[1]['limited_command'][:n],atol=1e-7,rtol=0)
 for name,d in zip(names,ds):
  parts=reconstruct(d);audits.append(dict(case=i,model=name,max_reward_difference=float(np.max(abs(sum(parts.values())-d['reward'])))))
  for regime,mask in [('all',np.ones(n,bool)),('low_requested',ds[0]['limited_command'][:n,0]<=2.2),('high_requested',ds[0]['limited_command'][:n,0]>=2.3)]:
   if mask.any():
    ev=d['actual_forward_speed'][:n]-d['limited_command'][:n,0];ed=d['actual_delta'][:n]-d['limited_command'][:n,1]
    paired.append(dict(case=i,domain=results[name][i]['domain'],model=name,regime=regime,steps=int(mask.sum()),speed_rmse=float(np.sqrt(np.mean(ev[mask]**2))),steer_rmse=float(np.sqrt(np.mean(ed[mask]**2)))))
 fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');ax=axs.ravel()
 for name,d,color,style in zip(names,ds,['#0072B2','#D55E00'],['-','--']):
  t=d['time']+.005;xy=d['actual_xy'];ax[0].plot(*xy.T,color=color,ls=style,label=name);ax[0].scatter(*xy[-1],color=color,marker='x' if d['physical_failure'][-1] else 's')
  ax[1].plot(t,d['reward'],color=color,ls=style,label=name);ax[2].plot(t,np.cumsum(d['reward']),color=color,ls=style)
  ax[3].plot(t,d['actual_forward_speed'],color=color,ls=style);ax[4].plot(t,np.rad2deg(d['actual_delta']),color=color,ls=style);ax[5].plot(t,np.rad2deg(d['phi']),color=color,ls=style)
 d=max(ds,key=lambda x:len(x['time']));t=d['time']+.005
 ax[0].plot(*d['reference_xy'].T,':',color='black',label='command integration');ax[0].set(xlabel='World X (m)',ylabel='World Y (m)');ax[0].set_aspect('equal',adjustable='datalim');ax[0].legend()
 ax[3].plot(t,d['limited_command'][:,0],':',color='black');ax[4].plot(t,np.rad2deg(d['limited_command'][:,1]),':',color='black')
 for a,label in zip(ax[1:],['Step reward','Cumulative reward','Forward speed (m/s)','Steer (deg)','Roll (deg)']):a.set(xlabel='Task time (s)',ylabel=label);a.grid(alpha=.2)
 fig.suptitle(f'TEST domain {results[names[0]][i]["domain"]}, state {i}; local300 vs frozen lower200; command seed 930502')
 f=f'case_{i:02d}.png';fig.savefig(out/f,dpi=150);plt.close(fig);lines.append(f'![Case {i}]({f})')
 fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained')
 for row,(name,d) in enumerate(zip(names,ds)):
  t=d['time']+.005;parts=reconstruct(d)
  for k,v in parts.items():axs[row,0].plot(t,v,label=k);axs[row,1].plot(t,np.cumsum(v),label=k)
  axs[row,2].plot(t,d['limited_command'][:,0],':',color='black',label='requested')
  axs[row,2].plot(t,d['actual_forward_speed'],label='true forward')
  axs[row,2].plot(t,d['wheel_speed_proxy'],'--',label='rear axle x 0.1m')
  for col,label in enumerate(['Step reward components','Cumulative components','Speed (m/s)']):
   axs[row,col].set(xlabel='Task time (s)',ylabel=label,title=name);axs[row,col].legend(fontsize=7);axs[row,col].grid(alpha=.2)
  if d['physical_failure'][-1]:
   for ax in axs[row]:ax.axvline(t[-1],color='black',ls='--')
 fig.suptitle(f'TEST state {i}; signed reward components and speed estimates; independent NumPy reward audit')
 f=f'case_{i:02d}_diagnostics.png';fig.savefig(out/f,dpi=130);plt.close(fig);lines.append(f'[Reward components and speed estimates]({f})')

(out/'paired_regimes.json').write_text(json.dumps(paired,indent=2)+'\n')
(out/'reward_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))
