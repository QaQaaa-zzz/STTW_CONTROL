"""Analyze saved paired PhysX trajectories without rerunning simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
reward_config=json.loads((a.bundle/'config.json').read_text());route=np.load(a.bundle/'route.npz');alphas=[0,.25,.5,.75,1];colors=plt.get_cmap('tab10').colors;all_rows=[]
for terrain in ['slope','rough_slope']:
 folder=a.run/terrain;base=json.loads((folder/'baseline.json').read_text());fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained');reference=route['s']<=12
 axes[0,0].plot(*route['xy'][reference].T,'k:',label='Original reference')
 for ai,alpha in enumerate(alphas):
  series={}
  for method in ['baseline','parent','final']:
   name=method+'_a'+str(alpha);receipt=json.loads((folder/('baseline.json' if method=='baseline' else name+'.json')).read_text());d=np.load(folder/(name+'.rewards.npz'));raw=np.load(folder/('baseline.npz' if method=='baseline' else name+'.npz'))
   for key in ['asset_sha256','controller_config','drive_speed','ground','dt_s','initial_states','initial_dofs','shape_properties','lower_actor','terrain_mesh','physx_config']:
    if receipt[key]!=base[key]:raise ValueError('pair mismatch '+key)
   for key in ['config.json','route.npz']:
    if receipt['path_contract']['bundle']['files'][key]!=base['path_contract']['bundle']['files'][key]:raise ValueError('path configuration mismatch')
   np.testing.assert_allclose(d['reward'].sum(),d['tick_reward'].sum(),rtol=1e-5,atol=1e-4)
   series[method]=(receipt,d,raw)
   label=method+' alpha='+str(alpha);style='--' if method=='parent' else '-';color='black' if method=='baseline' else colors[ai]
   if method!='baseline' or ai==0:
    axes[0,0].plot(*d['xy'].T,style,color=color,label=label if method!='baseline' else 'Zero middle + LOCAL300');axes[0,0].plot(*d['xy'][-1],marker='x',color=color)
    t=np.minimum(np.arange(1,len(d['speed'])+1)*.005,receipt['physics_steps']*.0002)
    for ax,key in [(axes[1,0],'speed'),(axes[1,1],'cross'),(axes[1,2],'peak_roll')]:ax.plot(t,d[key],style,color=color)
   if method!='baseline':
    steps=np.arange(1,len(d['reward'])+1);axes[0,1].plot(steps,d['reward'],style,color=color);axes[0,2].plot(steps,np.cumsum(d['reward']),style,color=color)
  n=min(series[m][0]['physics_steps']//25 for m in ['parent','final'])
  for method,(r,d,raw) in series.items():
   all_rows.append(dict(terrain=terrain,alpha=alpha,method=method,termination=r['termination'],seconds=r['physics_steps']*.0002,reward=float(d['reward'].sum()),common_parent_final_seconds=n*.005 if method!='baseline' else None,speed_rmse_common=float(np.sqrt(np.mean((d['speed'][:n]-r['drive_speed'])**2))) if method!='baseline' else None,cross_rmse_common=float(np.sqrt(np.mean(d['cross'][:n]**2))) if method!='baseline' else None,max_abs_cross=float(np.max(np.abs(d['cross']))),max_abs_roll=float(d['peak_roll'].max())))
  # Per-alpha diagnostics include matched baseline reward under this alpha.
  detail,axs=plt.subplots(3,3,figsize=(15,11),layout='constrained')
  axs[0,0].plot(*route['xy'][reference].T,'k:',label='Original reference')
  for method,(r,d,raw) in series.items():
   color={'baseline':'black','parent':'tab:blue','final':'tab:orange'}[method];t=np.minimum(np.arange(1,len(d['speed'])+1)*.005,r['physics_steps']*.0002);steps=np.arange(1,len(d['reward'])+1);label=method+' / '+r['termination']
   axs[0,0].plot(*d['xy'].T,color=color,label=label);axs[0,0].plot(*d['xy'][-1],marker='x',color=color)
   axs[0,1].plot(steps,d['reward'],color=color);axs[0,2].plot(steps,np.cumsum(d['reward']),color=color)
   axs[1,0].plot(t,d['speed'],color=color,label=method+' true');control=raw['control_trace'];axs[1,0].plot(control[:,0],control[:,6]*.1,color=color,linestyle=':',label=method+' wheel estimate')
   axs[1,1].plot(t,d['cross'],color=color);axs[1,2].plot(t,d['heading_error'],color=color);axs[2,0].plot(t,d['peak_roll'],color=color)
   names=list(d['component_names']);components=-reward_config['path_reward']['scale']*reward_config['timing']['lower_dt_s']*d['effective_components']
   for key,style in [('speed','-'),('path','--'),('primary',':')]:
    value=components[:,names.index(key)];axs[2,1].plot(t,value,color=color,linestyle=style,label=method+' '+key);axs[2,2].plot(t,np.cumsum(value),color=color,linestyle=style)
  labels=['XY [m]','Reward / policy step','Cumulative reward','Speed [m/s]; dotted wheel estimate','Signed cross-track [m]','Heading error [rad]','Peak roll magnitude [rad]','Lower-step reward terms (failure penalty separate)','Cumulative terms (failure penalty separate)']
  for ax,title in zip(axs.flat,labels):ax.set_title(title,fontsize=9);ax.grid(alpha=.2)
  axs[0,0].set_aspect('equal',adjustable='datalim');axs[0,0].legend(fontsize=6);axs[1,0].axhline(base['drive_speed'],color='grey',linestyle='--');axs[2,1].legend(fontsize=5,ncol=2)
  detail.suptitle(terrain+' | alpha='+str(alpha)+' | parent policy_000 / final policy_002 / frozen lower\nSeed 20261105; 5s DEV; terrain begins at world X=2m; actual endpoints; no imposed force pulse')
  detail.savefig(folder/('detail_a'+str(alpha)+'.png'),dpi=140);plt.close(detail)
 axes[0,0].set_aspect('equal',adjustable='datalim');axes[0,0].legend(fontsize=6,ncol=2);axes[1,0].axhline(base['drive_speed'],color='grey',linestyle=':')
 for ax,label in zip(axes.flat,['Actual/reference XY [m]','Reward / policy step (50Hz)','Cumulative reward','True forward speed [m/s]','Signed cross-track [m]','Peak roll magnitude [rad]']):ax.set_title(label);ax.grid(alpha=.2)
 axes[0,0].set(xlabel='World X [m]',ylabel='World Y [m]')
 for ax in axes[1]:ax.set_xlabel('Time [s]')
 fig.suptitle(terrain+' | parent dashed / final solid | five alphas | seed 20261105\nFrozen lower black; alpha-rescored baseline rewards in detailed panels; terrain starts at world X=2m; endpoints x')
 fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
(a.run/'metrics.json').write_text(json.dumps(all_rows,indent=2))
lines=['# Isaac H75 terrain paired DEV','', 'Parent policy_000 vs final policy_002; zero middle + LOCAL300 + ECBC/ESO baseline. Single initial state, deterministic means, 5s. Budget end is not task success. No independent TEST claim.','']
for terrain in ['slope','rough_slope']:
 lines+=['## '+terrain,'','![XY and rewards]('+terrain+'/comparison.png)', '', '| alpha | parent stop | final stop | parent/final cross RMSE m | parent/final speed RMSE m/s | common seconds |','|---|---|---|---|---|---|']
 for alpha in alphas:
  old=next(r for r in all_rows if r['terrain']==terrain and r['alpha']==alpha and r['method']=='parent');new=next(r for r in all_rows if r['terrain']==terrain and r['alpha']==alpha and r['method']=='final')
  lines.append('|%s|%s|%s|%.4f / %.4f|%.4f / %.4f|%.3f|'%(alpha,old['termination'],new['termination'],old['cross_rmse_common'],new['cross_rmse_common'],old['speed_rmse_common'],new['speed_rmse_common'],old['common_parent_final_seconds']))
 lines+=['']+['[alpha %s detailed paired rewards/errors](%s/detail_a%s.png)'%(alpha,terrain,alpha) for alpha in alphas]+['']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(cells=len(all_rows),physical_episodes=22)))
