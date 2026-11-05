"""Plot saved matched H75 traces; no simulation or model update."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sttw_control.path_command_reward import costs
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
cfg=json.loads((r/'bundle/config.json').read_text());c=cfg['path_reward'];lab=np.load(r/'lab/trajectory.npz');route=np.load(r/'bundle/route.npz');out=r/'comparison';out.mkdir(exist_ok=True)
labels=['alpha0','alpha025','alpha05','alpha075','alpha1','zero'];names=['α=0','α=.25','α=.5','α=.75','α=1','Zero'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500','#333333']
trace=lab['lower_trace'].reshape(-1,6,11);last=lab['offsets'].reshape(-1,4,6,2)[:,-1];prev=np.concatenate([np.zeros_like(last[:1]),last[:-1]])
rate=(last-prev)/.02;cost=np.minimum(c['offset_rate_weight']*((rate/np.array(c['offset_rate_scales']))**2).sum(-1),c['component_caps']['offset_rate']);lab_reward=lab['reward_normal'].reshape(-1,4,6)-c['scale']*.005*cost[:,None,:]
for step,lane in np.argwhere(lab['failure']):
 lab_reward[step,:,lane]=0;indices=np.flatnonzero(trace.reshape(-1,4,6,11)[step,:,lane,6]);lab_reward[step,indices[-1] if len(indices) else 0,lane]=lab['reward'][step,lane]
closure=float(np.max(np.abs(lab_reward.sum(1)-lab['reward'])))
if closure>1e-4:raise ValueError(f'lower reward replay discrepancy {closure}')
lab_reward=lab_reward.reshape(-1,6);datasets={};metrics=[]
for i,label in enumerate(labels):
 m=dict(np.load(r/'mujoco'/(label+'.npz')));active=trace[:,i,6].astype(bool);n=int(active.sum());t=np.arange(n)*.005;pose=lab['pose'][:n,i];meas=lab['measurement'][:n,i]
 l=dict(time=t,actual_xy=pose[:,:2],actual_forward_speed=trace[:n,i,2],wheel_speed_proxy=trace[:n,i,10],path_cross_track=trace[:n,i,3],phi=meas[:,0],phi_dot=meas[:,1],actual_delta=meas[:,2],path_progress=lab['progress'][:n,i],scored_tick_reward=lab_reward[:n,i],offsets=lab['offsets'][:n,i],target_offset=lab['target'][:n,i],filtered_offset=lab['filtered'][:n,i],governed=trace[:n,i,8:10],nominal=lab['nominal'][:n,i])
 for key in lab.files:
  if key.startswith(('raw_','effective_')):l[key.replace('raw_','raw_cost_',1).replace('effective_','effective_cost_',1)]=lab[key][:n,i]
 l['effective_cost_offset_rate']=np.repeat(cost[:,i],4)[:n];l['raw_cost_offset_rate']=np.repeat(c['offset_rate_weight']*((rate[:,i]/np.array(c['offset_rate_scales']))**2).sum(-1),4)[:n]
 heading=np.interp(l['path_progress'],route['s'],route['heading']);l['path_heading_error']=(heading-pose[:,2]+np.pi)%(2*np.pi)-np.pi;l['peak_roll']=trace[:n,i,4]
 datasets[label]=(m,l);np.savez_compressed(out/(label+'_lab_replayed.npz'),**l)
 common=min(len(m['time']),n);distance=np.linalg.norm(m['actual_xy'][:common]-l['actual_xy'][:common],axis=1)
 metrics.append(dict(condition=label,mujoco_seconds=len(m['time'])*.005,lab_seconds=n*.005,xy_rmse_m=float(np.sqrt(np.mean(distance**2))),xy_endpoint_separation_m=float(distance[-1]),mujoco_cross_track_rmse_m=float(np.sqrt(np.mean(m['path_cross_track']**2))),lab_cross_track_rmse_m=float(np.sqrt(np.mean(l['path_cross_track']**2))),mujoco_speed_rmse=float(np.sqrt(np.mean((m['actual_forward_speed']-2.3)**2))),lab_speed_rmse=float(np.sqrt(np.mean((l['actual_forward_speed']-2.3)**2))),mujoco_reward=float(m['scored_tick_reward'].sum()),lab_reward=float(l['scored_tick_reward'].sum())))
 for backend,d in [('mujoco',m),('lab',l)]:
  b=cfg['stage_B_baseline'];tail=d['time']+.005>10-b['hold_s']+1e-5;goal=float(route['goal']);goal_xy=np.array([np.interp(goal,route['s'],route['xy'][:,k]) for k in (0,1)]);yaw=np.interp(goal,route['s'],route['heading']);signed=(d['actual_xy']-goal_xy)@np.array([np.cos(yaw),np.sin(yaw)])
  metrics[-1][backend+'_goal_crossed']=bool(np.any((signed>=0)&(d['path_progress']>=goal)))
  metrics[-1][backend+'_final_hold']=bool(tail.sum()==round(b['hold_s']/.005) and np.all(abs(d['path_cross_track'][tail])<=b['final_y_error_m']) and np.all(abs(d['path_heading_error'][tail])<=b['final_path_heading_error_rad']) and np.all(abs(d['actual_forward_speed'][tail]-2.3)<=b['final_speed_error_m_s']))
  metrics[-1][backend+'_working_roll_exceeded']=bool(np.any(d['peak_roll']>b['working_roll_rad']))
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
def save(fig,name):fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
for ax,label,name in zip(axs.flat,labels,names):
 m,l=datasets[label];ax.plot(route['xy'][:,0],route['xy'][:,1],':',color='.5',label='Fixed original path')
 for d,col,style,title in [(m,'#0072B2','--','MuJoCo'),(l,'#D55E00','-','Isaac Lab')]:
  xy=d['actual_xy'];ax.plot(*xy.T,style,color=col,label=title);ax.plot(*xy[-1],marker='x',color=col)
 ax.set(title=name,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='box');ax.set_xlim(7,15);ax.set_ylim(-1,19);ax.legend(fontsize=8)
fig.suptitle('Unchanged H75 u77 + selected original Local300 | standard plane | friction 0.8 both engines\nSame full warm initial state; one case per condition; crosses mark actual endpoints; no disturbance')
save(fig,'xy_paired')
fig,axs=plt.subplots(1,2,figsize=(12,8),layout='constrained')
for backend,ax in enumerate(axs):
 for label,name,col in zip(labels,names,colors):
  xy=datasets[label][backend]['actual_xy'];ax.plot(*xy.T,color=col,label=name);ax.plot(*xy[-1],marker='x',color=col)
 ax.plot(*route['xy'].T,':',color='.5',label='Original path');ax.set(xlim=(7,15),ylim=(-1,19),xlabel='World x (m)',ylabel='World y (m)',title=['MuJoCo','Isaac Lab'][backend]);ax.set_aspect('equal');ax.legend()
save(fig,'xy_all_alpha')
fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
for backend,ax in enumerate(axs):
 for label,name,col in zip(labels,names,colors):
  xy=datasets[label][backend]['actual_xy'];ax.plot(*xy.T,color=col,label=name)
 ax.plot(*route['xy'].T,':',color='.5',label='Original path');ax.set(xlim=(9.5,14),ylim=(-.5,4.5),xlabel='World x (m)',ylabel='World y (m)',title=['MuJoCo bend detail','Isaac Lab bend detail'][backend]);ax.set_aspect('equal');ax.legend(fontsize=8)
save(fig,'xy_bend_detail')
for backend,model in enumerate(['MuJoCo','Isaac Lab']):
 fig,axs=plt.subplots(1,2,figsize=(12,8),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
 for ax in axs:
  ax.plot(*route['xy'].T,':',color='.55',linewidth=2,label='Fixed original path')
  for label,name,col in zip(labels,names,colors):
   xy=datasets[label][backend]['actual_xy'];ax.plot(*xy.T,color=col,linestyle='--' if label=='zero' else '-',linewidth=2,label=name);ax.plot(*xy[-1],marker='x',color=col)
  ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='box')
 axs[0].set(xlim=(7.5,14),ylim=(-.5,19.5),title='Full 10 s trajectory; × actual endpoint')
 axs[1].set(xlim=(10,13.7),ylim=(-.3,4.2),title='Bend detail — same colors and raw coordinates')
 axs[1].legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=10)
 fig.suptitle(model+' | original H75 u77 + selected Local300\nFive alpha levels overlaid; standard plane, no disturbance, one initial state')
 save(fig,'xy_'+['mujoco','isaac_lab'][backend]+'_all_alpha')
zero_rewards={}
for alpha,label in zip([0,.25,.5,.75,1],labels):
 zero_rewards[label]=[]
 for backend in (0,1):
  d=datasets['zero'][backend];curvature=np.interp(d['path_progress'],route['s'],route['curvature']);cc=json.loads(Path(json.loads((r/'lab_config.json').read_text())['controller_config']).read_text())
  # Same ECBC kinematic equilibrium ratio used by the runtime reward's chi.
  from sttw_control.controller import ControllerConfig,_system
  ratio=float(_system(jnp.float32(2.3),jnp.zeros(3),ControllerConfig(**cc))[3]);chi=np.clip((abs(np.arctan(cc['wheelbase']*curvature/np.cos(cc['caster']))/ratio)-.18)/.12,0,1)
  args=dict(alpha=jnp.full(len(chi),alpha),chi=jnp.asarray(chi),ev=jnp.asarray(d['actual_forward_speed']-2.3),ey=jnp.asarray(d['path_cross_track']),eh=jnp.asarray(d['path_heading_error']),peak_roll=jnp.asarray(d['peak_roll']),roll_rate=jnp.asarray(d['phi_dot']),speed=jnp.asarray(d['actual_forward_speed']),offsets=jnp.zeros((len(chi),2)),offset_rate=jnp.zeros((len(chi),2)))
  scored=jax.vmap(lambda **x:costs(**x,config=cfg))(**args);zero_rewards[label].append(np.asarray(scored['reward']))
for i,label in enumerate(labels[:5]):
 for backend,name in enumerate(['mujoco','lab']):metrics[i][name+'_matched_zero_reward']=float(zero_rewards[label][backend].sum())
for backend,model in enumerate(['MuJoCo','Isaac Lab']):
 fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 for label,name,col in zip(labels,names,colors):
  d=datasets[label][backend]
  for ax,key in zip(axs.flat,['actual_forward_speed','path_cross_track','phi','path_heading_error']):ax.plot(d['time'],d[key],color=col,linestyle='--' if label=='zero' else '-',label=name)
 for ax,title in zip(axs.flat,['True speed (m/s), request 2.3','Signed cross-track (m)','Roll (rad)','Path heading error (rad)']):ax.set(title=title,xlabel='Time since handover (s)');ax.legend(fontsize=8)
 fig.suptitle(model+' — same model, five alpha colors + matched Zero; no disturbance');save(fig,['mujoco','lab'][backend]+'_all_alpha_responses')
for label,name in zip(labels,names):
 m,l=datasets[label];fig,axs=plt.subplots(3,3,figsize=(17,10),layout='constrained')
 panels=[('actual_forward_speed','True forward speed (m/s)'),('path_cross_track','Signed cross-track (m)'),('phi','Roll (rad)'),('scored_tick_reward','Reward per 5 ms'),('cumulative','Cumulative reward'),('wheel_speed_proxy','Rear axle × 0.1 (m/s)'),('offsets','Executed offsets (m/s, rad)'),('target_offset','Mapped H75 targets (m/s, rad)'),('filtered_offset','Filtered targets (m/s, rad)')]
 for ax,(key,title) in zip(axs.flat,panels):
  for d,col,style,backend in [(m,'#0072B2','--','MuJoCo'),(l,'#D55E00','-','Lab')]:
   y=np.cumsum(d['scored_tick_reward']) if key=='cumulative' else d[key];ax.plot(d['time'],y,style,color=col,label=backend);ax.plot([d['time'][-1]],[y[-1]],'x',color=col)
  if label!='zero':
   for backend,style in enumerate([':', '-.']):
    d=datasets['zero'][backend];y=zero_rewards[label][backend] if key in ('cumulative','scored_tick_reward') else d[key]
    if key=='cumulative':y=np.cumsum(y)
    ax.plot(d['time'],y,style,color='.5',linewidth=1,label=['MJ Zero (same α reward)','Lab Zero (same α reward)'][backend])
  if key=='actual_forward_speed':ax.axhline(2.3,color='.4',linestyle=':',label='User speed')
  ax.set(title=title,xlabel='Time since handover (s)');ax.legend(fontsize=8)
 fig.suptitle(f'{name} | H75 u77 + original Local300 | flat matched DEV | no disturbance');save(fig,label+'_responses')
 keys=[k for k in m if k.startswith('effective_cost_')];fig,axs=plt.subplots(2,5,figsize=(19,7),layout='constrained')
 for ax,key in zip(axs.flat,keys):
  for d,col,style in [(m,'#0072B2','--'),(l,'#D55E00','-')]:ax.plot(d['time'],-c['scale']*.005*d[key],style,color=col)
  ax.set(title=key.removeprefix('effective_cost_'),xlabel='Time (s)',ylabel='Signed reward contribution')
 fig.suptitle(name+' signed cost contributions; total/failure reward shown separately');save(fig,label+'_reward_components')
(out/'metrics.json').write_text(json.dumps(dict(reward_replay_max_abs_error=closure,cases=metrics),indent=2))
lines=['# H75 paired standard-plane DEV comparison','', '![XY paired](xy_paired.png)','', '![All alpha](xy_all_alpha.png)','', 'Original H75 u77 and orange-curve Local300 weights unchanged. Both backends use front friction 0.8 and the same full warm initial state; 5 kHz physics / 200 Hz lower / 50 Hz upper. Five alpha levels plus Zero (same Local300, no middle correction). One route and initial state, no disturbance; not independent generalization evidence.','', '**Historical correction:** the older MuJoCo source had front friction 10, while the selected orange Lab used 0.8. Older plots remain intact and cannot isolate engine effects. This is a new matched-0.8 comparison.','',f'Lab per-control reward replay closes against saved upper rewards: maximum absolute difference {closure:.3g}. Original NPZ retained; derived replay NPZ contains step rewards and signed component costs. No smoothing; endpoints stop at actual active ticks. Survival is not task qualification.','', '| Condition | XY RMSE m | End separation m | Cross-track RMSE MJ / Lab m | Speed RMSE MJ / Lab m/s |', '|---|---:|---:|---:|---:|']
for d in metrics:lines.append(f"|{d['condition']}|{d['xy_rmse_m']:.3f}|{d['xy_endpoint_separation_m']:.3f}|{d['mujoco_cross_track_rmse_m']:.3f} / {d['lab_cross_track_rmse_m']:.3f}|{d['mujoco_speed_rmse']:.3f} / {d['lab_speed_rmse']:.3f}|")
lines+=['','Detailed responses and signed reward components:']
lines+=['','Grouped by model: [MuJoCo five-alpha XY](xy_mujoco_all_alpha.png), [Lab five-alpha XY](xy_isaac_lab_all_alpha.png), [bend detail](xy_bend_detail.png).','', 'Goal/final-hold checks reuse stage_B_baseline thresholds: last 1 s, |cross-track|≤0.1 m, |heading|≤0.05 rad, |speed error|≤0.1 m/s; working roll≤0.302 rad. Goal crossing uses the fixed original path goal cross-section and progress. These checks cover this one declared case only.','', '| Condition | MJ goal / hold / working-roll exceeded | Lab goal / hold / working-roll exceeded |','|---|---|---|']
for d in metrics:lines.append(f"|{d['condition']}|{d['mujoco_goal_crossed']} / {d['mujoco_final_hold']} / {d['mujoco_working_roll_exceeded']}|{d['lab_goal_crossed']} / {d['lab_final_hold']} / {d['lab_working_roll_exceeded']}|")
for label in labels:lines.append(f'- [{label} responses]({label}_responses.png), [components]({label}_reward_components.png), [Lab replay data]({label}_lab_replayed.npz).')
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(metrics,indent=2))
