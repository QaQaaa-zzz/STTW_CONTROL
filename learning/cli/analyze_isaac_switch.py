"""Analyze saved spatial alpha-switch/coordination trials and contact sensors."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--title',default='Original H75 u77');a=p.parse_args();run=a.run;raw=np.load(run/'trajectory.npz');contract=json.loads((run/'contract.json').read_text());settings=contract['settings'];cfg=json.loads((Path(contract['bundle'])/'config.json').read_text());route=np.load(Path(contract['bundle'])/'route.npz');c=cfg['path_reward'];b=cfg['stage_B_baseline'];labels=settings['case_labels'];n=len(labels);tr=raw['lower_trace'].reshape(-1,n,11);out=run.parent/'analysis';out.mkdir(exist_ok=True)
last=raw['offsets'].reshape(-1,4,n,2)[:,-1];prev=np.concatenate([np.zeros_like(last[:1]),last[:-1]]);rate=(last-prev)/.02;ratecost=np.minimum(c['offset_rate_weight']*((rate/np.array(c['offset_rate_scales']))**2).sum(-1),c['component_caps']['offset_rate']);reward=raw['reward_normal'].reshape(-1,4,n)-c['scale']*.005*ratecost[:,None,:];reward=np.where(raw['valid'][:,None,:],reward,0)
for step,lane in np.argwhere(raw['failure']):
 reward[step,:,lane]=0;ids=np.flatnonzero(tr.reshape(-1,4,n,11)[step,:,lane,6]);reward[step,ids[-1] if len(ids) else 0,lane]=raw['reward'][step,lane]
closure=float(abs(reward.sum(1)-raw['reward']).max());assert closure<1e-4,closure;reward=reward.reshape(-1,n);metrics=[];lanes=[]
terrain_path=Path(next(iter(settings['terrains'].values())));terrain=np.load(terrain_path/'terrain.npz');meta=json.loads((terrain_path/'identity.json').read_text());v=terrain['vertices'];side=int(np.sqrt(len(v)));height=v[:,2].reshape(side,side);spacing=meta['horizontal_scale_m'];lo=meta['bounds_m'][0]
for i,label in enumerate(labels):
 ids=np.flatnonzero(tr[:,i,6]);t=ids*.005+tr[ids,i,7]*.0002;pose=raw['pose'][ids,i];progress=raw['progress'][ids,i];eh=(np.interp(progress,route['s'],route['heading'])-pose[:,2]+np.pi)%(2*np.pi)-np.pi;switch_ids=np.flatnonzero(raw['switch_tick'][:,i]>=0);switch_tick=int(raw['switch_tick'][switch_ids[0],i]) if len(switch_ids) else -1;switch_s=switch_tick*.005 if switch_tick>=0 else None
 d=dict(time=t,xy=pose[:,:2],speed=tr[ids,i,2],cross=tr[ids,i,3],roll=tr[ids,i,4],target=raw['target'][ids,i],offsets=raw['offsets'][ids,i],filtered=raw['filtered'][ids,i],eta=raw['channel_progress'][ids,i],reward=reward[ids,i],alpha=raw['alpha'][ids,i]);lanes.append(d)
 goal=float(route['goal']);goalxy=np.array([np.interp(goal,route['s'],route['xy'][:,k]) for k in (0,1)]);yaw=np.interp(goal,route['s'],route['heading']);signed=(d['xy']-goalxy)@np.array([np.cos(yaw),np.sin(yaw)]);failed=bool(raw['failure'][:,i].any());tail=t>len(tr)*.005-b['hold_s']+1e-5
 row=dict(case=label,switch_s=switch_s,switch_tick=switch_tick,failed=failed,observed_seconds=float(t[-1]),goal_crossed=bool(np.any((signed>=0)&(progress>=goal))),final_hold=bool(tail.sum()==round(b['hold_s']/.005) and np.all(abs(d['cross'][tail])<=b['final_y_error_m']) and np.all(abs(eh[tail])<=b['final_path_heading_error_rad']) and np.all(abs(d['speed'][tail]-2.3)<=b['final_speed_error_m_s']) and not failed),working_roll_exceeded=bool(np.any(d['roll']>b['working_roll_rad'])),speed_rmse_m_s=float(np.sqrt(np.mean((d['speed']-2.3)**2))),path_rmse_m=float(np.sqrt(np.mean(d['cross']**2))),peak_roll_rad=float(d['roll'].max()),reward=float(d['reward'].sum()))
 points=raw['wheel_contact_points'][:,:,i].reshape(-1,2,raw['wheel_contact_points'].shape[-2],3)-np.array(contract['origin'][i]);forces=raw['wheel_contact_forces'][:,:,i].reshape(points.shape);physical_active=(np.repeat(tr[:,i,6].astype(bool),25)&(np.tile(np.arange(25),len(tr))<np.repeat(tr[:,i,7],25)));contact_t=(np.arange(len(points))+1)*.0002
 for wheel,w in [('front',0),('rear',1)]:
  xy=points[:,w,:,:2];finite=np.isfinite(xy).all(-1);q=np.rint((np.nan_to_num(xy)-lo)/spacing).astype(int);inside=finite&(q>=0).all(-1)&(q<side).all(-1);q=np.clip(q,0,side-1);support=height[q[...,0],q[...,1]]>0;contact=np.linalg.norm(forces[:,w],axis=-1)>1e-3;raised=(inside&support&contact&(points[:,w,:,2]>.0025)).any(-1)&physical_active;observed=contact_t[raised]
  row[wheel+'_raised_contact_start_s']=float(observed[0]) if len(observed) else None;row[wheel+'_raised_contact_end_s']=float(observed[-1]) if len(observed) else None;row[wheel+'_peak_contact_force_N']=float(np.linalg.norm(forces[physical_active,w],axis=-1).max());d[wheel+'_raised_contact']=raised
 row['switch_after_rear_clearance']=switch_s is not None and row['rear_raised_contact_end_s'] is not None and switch_s>row['rear_raised_contact_end_s'];row['switch_before_turn_end']=bool(len(switch_ids) and raw['progress'][switch_ids[0],i]<float(route['turn_end']));metrics.append(row);np.savez_compressed(out/(label+'.npz'),**d)
colors=['#0072B2','#CC79A7','#009E73'];modes=['Independent','Common progress','Steer priority'];comparisons=[]
fig,axs=plt.subplots(2,2,figsize=(13,12),layout='constrained',gridspec_kw={'width_ratios':[1,1.7]})
for group in range(n//3):
 for ax in axs[group]:
  ref=route['s']<=max(raw['progress'][:,group*3:(group+1)*3].max(),float(route['turn_end']))+1;ax.plot(*route['xy'][ref].T,':',color='.5',label='Original fixed path')
  for j in range(3):
   i=group*3+j;d=lanes[i];ax.plot(*d['xy'].T,color=colors[j],label=modes[j]);ax.plot(*d['xy'][-1],marker='x',color=colors[j]);event=metrics[i]['switch_tick']
   if event>=0 and event<len(d['xy']):ax.plot(*d['xy'][event],marker='o',fillstyle='none',color=colors[j])
  ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal')
 axs[group,0].set_title(f"alpha {settings['alphas'][group*3]} → {settings['alpha_switch']['target_alpha'][group*3]} | full XY")
 bend=route['xy'][abs(route['curvature'])>1e-5];axs[group,1].set(xlim=(bend[:,0].min()-.5,bend[:,0].max()+.7),ylim=(bend[:,1].min()-.3,bend[:,1].max()+2),title='Bend / switch detail');axs[group,1].legend(fontsize=9)
 end=min(len(lanes[group*3+j]['time']) for j in range(3));start=max(metrics[group*3+j]['switch_tick'] for j in range(3));prefix=max(0,min(metrics[group*3+j]['switch_tick'] for j in range(3)));prefix_errors={}
 for key in ['pose','observation','filtered','offsets','measurement']:
  reference=raw[key][:prefix,group*3];prefix_errors[key]=max(float(abs(raw[key][:prefix,group*3+j]-reference).max()) if prefix else 0 for j in (1,2))
 comp=dict(direction=labels[group*3].rsplit('_',1)[0],prefix_compared_ticks=prefix,prefix_max_abs_differences=prefix_errors,post_switch_common_window_s=[max(start,0)*.005,end*.005],methods=[])
 for j in range(3):
  d=lanes[group*3+j];mask=slice(max(start,0),end);comp['methods'].append(dict(mode=modes[j],speed_rmse_m_s=float(np.sqrt(np.mean((d['speed'][mask]-2.3)**2))),path_rmse_m=float(np.sqrt(np.mean(d['cross'][mask]**2))),peak_roll_rad=float(d['roll'][mask].max()),steer_limited_more_ticks=int(np.sum(d['eta'][mask,1]<d['eta'][mask,0])),speed_limited_more_ticks=int(np.sum(d['eta'][mask,0]<d['eta'][mask,1]))))
 comparisons.append(comp)
fig.suptitle(a.title+' + selected Local300 | 4 cm bump | spatial alpha switch at s ≥ 5.3 m\nOpen circles: switch; crosses: real endpoints; shared nominal prefix, recorded differences reported');fig.savefig(out/'xy.png',dpi=150);plt.close(fig)
for group in range(n//3):
 fig,axs=plt.subplots(3,3,figsize=(16,11),layout='constrained')
 for j in range(3):
  i=group*3+j;d=lanes[i];m=metrics[i];series=[(d['speed'],'True forward speed (m/s)'),(d['cross'],'Signed cross-track (m)'),(d['roll'],'Peak roll (rad)'),(d['target'][:,0],'Speed target correction (m/s)'),(d['target'][:,1],'Steer target correction (rad)'),(d['offsets'][:,0],'Executed speed correction (m/s)'),(d['offsets'][:,1],'Executed steer correction (rad)'),(d['reward'],'Reward per 5 ms'),(np.cumsum(d['reward']),'Cumulative reward')]
  for ax,(y,title) in zip(axs.flat,series):
   ax.plot(d['time'],y,color=colors[j],label=modes[j]);ax.set(title=title,xlabel='Time (s)');ax.grid(alpha=.2)
   if m['switch_s'] is not None:ax.axvline(m['switch_s'],color=colors[j],linestyle=':',alpha=.6)
   if j==0 and m['front_raised_contact_start_s'] is not None and m['rear_raised_contact_end_s'] is not None:ax.axvspan(m['front_raised_contact_start_s'],m['rear_raised_contact_end_s'],color='.5',alpha=.1)
 for ax in axs.flat:ax.legend(fontsize=7)
 fig.suptitle(comparisons[group]['direction']+' | vertical lines: actual switch; gray: first method measured raised contact interval');fig.savefig(out/f'direction{group}_responses.png',dpi=150);plt.close(fig)
report=dict(reward_replay_max_error=closure,cases=metrics,comparisons=comparisons,contact_definition='official mean contact point per wheel/filter, force norm>1e-3N, point z>2.5mm and XY above positive grid sample; sampled each0.2ms. First/last qualifying mean-point contact, not every individual manifold contact.',scope='nominal one-route one-initial-state DEV; three execution modes per switch direction; not independent seeds or generalization')
(out/'metrics.json').write_text(json.dumps(report,indent=2));lines=['# Spatial alpha-switch and reference-execution DEV','','![XY](xy.png)','',a.title+' and selected orange Local300 lower. Same original fixed path,4cm bump,full warm reset,2.3m/s,5000/200/50Hz. Nominal evaluation without DR; no optimizer updates. Independent prefix until first50Hz crossing s≥5.3m; only alpha and execution mode change; filter/history/ESO are preserved.','','Contact phases use actual official mean contact points and filter forces sampled each0.2ms; raised-contact definition: force>1e-3N,z>2.5mm and XY above positive grid sample. This is a declared measurement convention; not every individual manifold point. Prefix differences are reported in metrics.json, not assumed zero.','','| Case | Failed | Goal / hold / work exceed | Switch s | Rear raised contact end s | After clearance / still turning |','|---|---|---|---:|---:|---|']
for m in metrics:lines.append(f"|{m['case']}|{m['failed']}|{m['goal_crossed']} / {m['final_hold']} / {m['working_roll_exceeded']}|{m['switch_s']}|{m['rear_raised_contact_end_s']}|{m['switch_after_rear_clearance']} / {m['switch_before_turn_end']}|")
lines+=['',f'Per-lower reward replay closes to {closure:.3g}; terminal penalties retained. [0→1 responses](direction0_responses.png), [1→0 responses](direction1_responses.png). No benefit conclusion from survival or reward alone.']
lines+=['','## Interpretation and limitations','', 'The original H75 configuration declares alpha_conditioning.constant_per_episode=true. A mid-episode alpha switch is a new history/state distribution. This supports a distribution-shift hypothesis, not a proven sole cause of any oscillation.','', '| Direction | Mode | Common-window path RMSE m | Speed RMSE m/s | Ticks with steer more limited |','|---|---|---:|---:|---:|']
for comparison in comparisons:
 for method in comparison['methods']:lines.append(f"|{comparison['direction']}|{method['mode']}|{method['path_rmse_m']:.5f}|{method['speed_rmse_m_s']:.5f}|{method['steer_limited_more_ticks']}|")
lines+=['', 'Prefix state/history/filter differences are nonzero (see metrics.json). These are nominal-prefix comparisons; small differences between execution modes cannot establish a causal improvement. When the speed channel limits progress more, steer priority equals the original independent rule at the same state. A candidate without measured activation is not credited with a benefit. No execution candidate is adopted as an improvement from this panel.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report,indent=2))
