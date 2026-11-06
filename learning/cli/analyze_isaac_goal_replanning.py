"""Original-goal outcomes and saved XY for the integrated Lab replanning runs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sttw_control.isaac_path_reporting import replay_rewards
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--models',nargs='+',default=['original','best33']);a=p.parse_args()
colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7','#555555']
names=['alpha 0','alpha .25','alpha .5','alpha .75','alpha 1','alpha 1 to 0','Zero']
labels={'original':'Original H75 u77','best33':'Reward-peak update33','original_exit':'Original H75 u77: forward-exit handoff'}
models=[(key,labels.get(key,key)) for key in a.models]
all_rows=[];series={};closures={};model_groups={};model_names={}
for model,label in models:
 run=a.root/model;data=np.load(run/'trajectory.npz');contract=json.loads((run/'contract.json').read_text());settings=contract['settings'];cfg=json.loads((Path(contract['bundle'])/'config.json').read_text());route=np.load(Path(contract['bundle'])/'route.npz');events=json.loads((run/'replanning_decisions.json').read_text())
 names=['alpha '+format(float(v),'g') for v in settings['alphas'][:5]]+[settings['case_labels'][5], 'Zero'];model_names[model]=names
 goal_s=float(route['turn_end'])+settings['replanning']['goal_after_turn_m'] if 'goal_after_turn_m' in settings['replanning'] else float(route['goal'])
 goal=np.array([np.interp(goal_s,route['s'],route['xy'][:,k]) for k in (0,1)]);arrival_radius=settings['replanning']['arrival_radius_m']
 total,components,closure=replay_rewards(data,cfg);closures[model]=closure
 tr=data['lower_trace'].reshape(-1,settings['num_envs'],11)
 user_speed=float(settings.get('user_speed_m_s',2.3))
 groups=[(0,'--','Fixed route'),(7,'-','Replanning')] if settings['num_envs']==14 else [(0,'-','Replanning')]
 model_groups[model]=groups
 for i,case in enumerate(settings['case_labels']):
  ids=np.flatnonzero(tr[:,i,6]);t=ids*.005+tr[ids,i,7]*.0002;pose=data['pose'][ids,i];xy=pose[:,:2];distance=np.linalg.norm(xy-goal,axis=1)
  # Original-route projection is diagnostic only; never changes the rollout.
  base=route['xy'][:-1];segment=np.diff(route['xy'],axis=0);span=np.diff(route['s']);svals=[];cross=[]
  for start in range(0,len(xy),128):
   points=xy[start:start+128];u=np.clip(((points[:,None]-base)*segment).sum(-1)/(segment**2).sum(-1),0,1);proj=base+u[:,:,None]*segment;index=((points[:,None]-proj)**2).sum(-1).argmin(-1);rows=np.arange(len(points));ss=route['s'][index]+u[rows,index]*span[index];h=np.interp(ss,route['s'],route['heading']);delta=points-proj[rows,index];svals.extend(ss);cross.extend(-np.sin(h)*delta[:,0]+np.cos(h)*delta[:,1])
  svals=np.asarray(svals);cross=np.asarray(cross);plans=[e for e in events if e['lane']==i];arrival=np.flatnonzero(distance<=settings['replanning']['arrival_radius_m']);failed=bool(data['failure'][:,i].any())
  d=dict(time=t,xy=xy,speed=tr[ids,i,2],wheel_speed=tr[ids,i,10],original_progress=svals,original_cross=cross,execution_cross=tr[ids,i,3],roll=tr[ids,i,4],roll_rate=data['measurement'][ids,i,1],reward=total[ids,i],goal_distance=distance,alpha=data['alpha'][ids,i],nominal=data['nominal'][ids,i],reference=tr[ids,i,8:10],target=data['target'][ids,i],plans=plans)
  series[model,i]=d
  row=dict(model=model,case=case,failed=failed,survived_full_window=not failed and len(ids)==len(tr),goal_arrived=bool(len(arrival)),first_arrival_s=float(t[arrival[0]]) if len(arrival) else None,min_goal_distance_m=float(distance.min()),original_path_rmse_m=float(np.sqrt(np.mean(cross**2))),execution_path_rmse_m=float(np.sqrt(np.mean(d['execution_cross']**2))),speed_rmse_m_s=float(np.sqrt(np.mean((d['speed']-user_speed)**2))),peak_roll_rad=float(d['roll'].max()),replans=sum(e.get('event')!='goal_arrival_forward_exit' for e in plans),arrival_handoffs=sum(e.get('event')=='goal_arrival_forward_exit' for e in plans),planner_seconds=sum(e['planning_seconds'] for e in plans),reward_sum=float(d['reward'].sum()))
  task_end=int(arrival[0])+1 if len(arrival) else len(ids)
  row['before_goal_or_termination']=dict(window_end_s=float(t[task_end-1]),end_reason='first_goal_entry' if len(arrival) else ('termination' if failed else 'observation_end'),original_path_rmse_m=float(np.sqrt(np.mean(cross[:task_end]**2))),original_path_peak_m=float(np.max(np.abs(cross[:task_end]))),speed_rmse_m_s=float(np.sqrt(np.mean((d['speed'][:task_end]-user_speed)**2))),peak_roll_rad=float(np.max(d['roll'][:task_end])))
  row['domain_exit']=bool(data['domain_exit'][ids,i].any()) if 'domain_exit' in data.files else None
  row['policy_fault']=bool(data['policy_fault'][ids,i].any()) if 'policy_fault' in data.files else None
  if i>=7:
   first=plans[0]['task_tick'] if plans else len(tr);row['prefix_pose_max_abs_difference']=float(np.max(abs(data['pose'][:first,i]-data['pose'][:first,i-7]))) if first else 0.
  all_rows.append(row)
  np.savez_compressed(run/(case+'_analysis.npz'),**{k:v for k,v in d.items() if k!='plans'},**{f'reward_{k}':v[ids,i] for k,v in components.items()})
 (run/'reward_replay.json').write_text(json.dumps(dict(max_error=closure,note='Execution-path reward; geometry changes after planning, not a common original-path ranking'),indent=2))
 fig,axs=plt.subplots(7,4,figsize=(18,24),layout='constrained')
 for j,name in enumerate(names):
  for offset,style,title in groups:
   d=series[model,j+offset]
   for ax,y,yl in zip(axs[j],[d['reward'],np.cumsum(d['reward']),d['speed'],d['original_cross']],['Reward / 5ms','Cumulative reward','Forward speed (m/s)','Original cross-track (m)']):
    ax.plot(d['time'],y,style,color=colors[j],label=title);ax.set(xlabel='Time (s)',ylabel=yl);ax.grid(alpha=.2)
    for event in d['plans']:ax.axvline(event['time_s'],color=colors[j],alpha=.2)
   axs[j,2].plot(d['time'],d['wheel_speed'],style,color='.6',alpha=.5)
  axs[j,2].axhline(user_speed,color='black',linewidth=.7);axs[j,0].set_title(name);axs[j,0].legend(fontsize=7)
 fig.suptitle(label+' | all cases; execution-path rewards and unchanged-original-path errors\nGray speed: rear wheel estimate; vertical lines: replans; fixed-route traces are integration diagnostics')
 fig.savefig(a.root/(model+'_responses.png'),dpi=130);plt.close(fig)
 fig,axs=plt.subplots(7,2,figsize=(15,22),layout='constrained')
 for j,name in enumerate(names):
  for offset,style,title in groups:
   d=series[model,j+offset];npz=np.load(run/(settings['case_labels'][j+offset]+'_analysis.npz'))
   for key in components:
    axs[j,0].plot(d['time'],npz['reward_'+key],style,label=key if offset==groups[-1][0] else None,linewidth=.7)
    axs[j,1].plot(d['time'],np.cumsum(npz['reward_'+key]),style,linewidth=.7)
  axs[j,0].set_title(name+' | per-step components');axs[j,1].set_title('Cumulative components')
  for ax in axs[j]:ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
 axs[0,0].legend(fontsize=6,ncol=3);fig.suptitle(label+' | solid: replanning; dashed: fixed | execution reward components');fig.savefig(a.root/(model+'_reward_components.png'),dpi=120);plt.close(fig)

def reference(ax):
 keep=route['s']<24;ax.plot(*route['xy'][keep].T,':',color='.6',label='Original route');ax.scatter(*goal,marker='*',color='black',s=70,label='Original goal');ax.add_patch(Circle(goal,arrival_radius,fill=False,edgecolor='.7',linestyle=':'));ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
columns=max(len(g) for g in model_groups.values())
fig,axs=plt.subplots(len(models),columns,figsize=(6.5*columns,6*len(models)),squeeze=False,layout='constrained')
for row,(model,label) in enumerate(models):
 names=model_names[model]
 for col,(offset,style,title) in enumerate(model_groups[model]):
  ax=axs[row,col];reference(ax)
  for j in range(7):
   d=series[model,j+offset];ax.plot(*d['xy'].T,color=colors[j],label=names[j]);ax.plot(*d['xy'][-1],marker='x',color=colors[j])
  ax.set_title(label+' | '+title);ax.legend(fontsize=7)
 if len(model_groups[model])==1 and columns>1:axs[row,1].set_visible(False)
fig.savefig(a.root/'xy_models.png',dpi=160)
bend=route['xy'][abs(route['curvature'])>1e-5]
for ax in axs.flat:ax.set(xlim=(bend[:,0].min()-.4,bend[:,0].max()+.8),ylim=(bend[:,1].min()-.3,bend[:,1].max()+1.2))
fig.savefig(a.root/'xy_models_bend.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,len(models),figsize=(6.5*len(models),7),squeeze=False,layout='constrained');axs=axs[0]
for ax,(model,label) in zip(axs,models):
 reference(ax)
 for offset,style,name in model_groups[model]:
  i=offset+5;d=series[model,i];ax.plot(*d['xy'].T,style,label=name);ax.plot(*d['xy'][-1],marker='x')
  for event in d['plans']:
   q=np.load(a.root/model/event['file']);extent=float(q['goal']) if event.get('event')!='goal_arrival_forward_exit' else float(np.linalg.norm(d['xy'][-1]-event['start_pose'][:2])+1);ss=q['s']<=extent;ax.plot(*q['xy'][ss].T,':',alpha=.6,label='Forward exit reference' if event.get('event')=='goal_arrival_forward_exit' else None);ax.scatter(*event['start_pose'][:2],s=20,color='black')
 ax.set_title(label+' | '+model_names[model][5]);ax.legend(fontsize=8)
fig.savefig(a.root/'xy_switch.png',dpi=160);plt.close(fig)
(a.root/'metrics.json').write_text(json.dumps(dict(goal_xy=goal.tolist(),goal_progress_m=goal_s,arrival_radius_m=arrival_radius,cases=all_rows,reward_replay=closures,scope='one route and initial state, nominal DEV; no global superiority; declared original task goal disk;  full-window completion separate'),indent=2))
lines=['# Integrated replanning: configured frozen controllers','','![All five alphas grouped by model](xy_models.png)','','![Switch trajectories and actual plans](xy_switch.png)','','Original target and physical limits unchanged. Termination flags include follower-domain exits, so they are not all falls. Fixed-route panels diagnose the integration; replanning is part of the intended execution system. No optimizer updates. Goal entry means measured XY enters the declared goal disk, not endpoint rest or a safety guarantee. Crosses are actual observed endpoints. All termination flags retained. After goal entry, the replanned route hands off to a measured-heading straight forward extension; this tail is not tracking the old path and is not a validated next task.','','|Model|Case|Termination flag|Goal entry (s)|Replans|Original path RMSE (m)|Speed RMSE (m/s)|','|---|---|---|---|---|---|---|']
for r in all_rows:lines.append(f"|{r['model']}|{r['case']}|{r['failed']}|{r['first_arrival_s']}|{r['replans']}|{r['original_path_rmse_m']:.3f}|{r['speed_rmse_m_s']:.3f}|")
lines += ['', '## Original task before first goal entry', '', 'These windows end at first measured goal entry, true termination, or observation end. Full-window values above are retained separately because the post-arrival forward-exit reference can intentionally leave the original route. Different arrival times imply different observation windows; this table describes each completed task, not an equal-duration comparison.', '', '|Model|Case|End reason|End time s|Original path RMSE m|Original path peak m|Speed RMSE m/s|', '|---|---|---|---:|---:|---:|---:|']
for r in all_rows:
 q=r['before_goal_or_termination'];lines.append(f"|{r['model']}|{r['case']}|{q['end_reason']}|{q['window_end_s']:.3f}|{q['original_path_rmse_m']:.4f}|{q['original_path_peak_m']:.4f}|{q['speed_rmse_m_s']:.4f}|")
for model,label in models:lines.extend(['',f'[{label}: rewards, cumulative rewards, speed and original errors]({model}_responses.png)',f'[{label}: all per-step/cumulative reward components]({model}_reward_components.png)'])
lines += ['', '[Bend detail: all five alphas](xy_models_bend.png)', 'Full-window original-path errors include the forward-exit tail after arrival; do not treat that deliberately changed reference as old-path recovery performance.',f'Declared goal progress: {goal_s:.6f}m; goal XY: {goal.tolist()}; arrival radius: {arrival_radius}m.']
lines+=['','Execution-path reward is rebuilt from saved effective terms plus offset-rate/terminal terms; it is not used to claim original-task superiority after changing geometry. Per-case NPZ retains coordinates, both errors, actual/rear-wheel speeds and reward components. No terrain-obstacle planning certificate is implied.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(rows=len(all_rows),closures=closures)))
