"""Plot saved fixed-alpha GPU trajectories without rerunning physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--baseline-run',type=Path);p.add_argument('--rescore-zero',action='store_true',help='Use the Zero lane in this run for same-alpha paired rewards');p.add_argument('--title',default='H75 fixed-alpha GPU evaluation');a=p.parse_args();out=a.run if a.output is None else a.output;out.mkdir(parents=True,exist_ok=True)
from sttw_control.isaac_path_reporting import load_case,write_zero_scores,static_case_metrics
if a.rescore_zero:
 write_zero_scores(a.run);a.baseline_run=a.run
r,c=load_case(a.run,a.baseline_run);physics_dt=c['physics_dt'];user_speed=float(c["settings"].get("user_speed_m_s",2.3));palette_markers=['o','s','^','D','v','x'];route=np.load(Path(c['bundle'])/'route.npz');labels=['alpha 0','alpha .25','alpha .5','alpha .75','alpha 1','Zero'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500','#333333'];styles=['-']*5+['--'];summary=[]
count=r['reward'].shape[1];labels=labels[:count];colors=colors[:count]
fig,axs=plt.subplots(2,2,figsize=(12,9));xy,per,cumulative,speed=axs.ravel();last_progress=max(float(r['progress'][np.flatnonzero(r['lower_trace'][:,:,i,6].reshape(-1))[-1],i]) for i in range(count));ref=route['s']<=last_progress+1;xy.plot(*route['xy'][ref].T,':',color='gray',label='Original reference')
for lane,(label,color) in enumerate(zip(labels,colors)):
    trace=r['lower_trace'][:,:,lane].reshape(-1,11);mask=trace[:,6].astype(bool);indices=np.flatnonzero(mask);steps=np.flatnonzero(r['valid'][:,lane]);position=trace[indices,:2];t=indices*.005+trace[indices,7]*physics_dt;rewards=r['reward'][steps,lane]
    xy.plot(*position.T,color=color,label=label,linestyle=styles[lane]);xy.scatter(*position[-1],marker='x',color=color)

    if lane<5 or a.baseline_run is None:
        per.plot(steps+1,rewards,color=color,label=label,linestyle=styles[lane]);cumulative.plot(steps+1,np.cumsum(rewards),color=color)
    speed.plot(t,trace[indices,2]-user_speed,color=color)
    summary.append(dict(observation_end_s=float(t[-1]),lane=lane,label=label,observed_lower_steps=len(indices),failed=bool(r['failure'][:,lane].any()),last_progress_m=float(r['progress'][indices[-1],lane]),speed_rmse=float(np.sqrt(np.mean((trace[indices,2]-user_speed)**2))),cross_rmse=float(np.sqrt(np.mean(trace[indices,3]**2))),peak_roll_rad=float(trace[indices,4].max()),rate_limited_lower_steps=int(r['rate_clipped'][indices,lane].sum())))
xy.set(xlabel='World x (m)',ylabel='World y (m)',title='Actual XY; crosses mark observed endpoints');xy.axis('equal');xy.legend(fontsize=8);per.set(xlabel='Policy step',ylabel='Reward',title='Per-step reward (includes terminal penalty)');cumulative.set(xlabel='Policy step',ylabel='Cumulative reward');speed.set(xlabel='Time (s)',ylabel='True speed error (m/s)')
if a.baseline_run:
 for lane,stem in enumerate(['alpha0','alpha025','alpha05','alpha075','alpha1']):
  zero=np.load(a.baseline_run/(stem+'_zero_rescored.npz'));pr=np.add.reduceat(zero['reward'],np.arange(0,len(zero['reward']),4));steps=np.arange(len(pr))+1
  per.plot(steps,pr,'--',color=colors[lane],alpha=.6);cumulative.plot(steps,np.cumsum(pr),'--',color=colors[lane],alpha=.6)
fig.suptitle(a.title+' — exploratory DEV, one initial state\nZero = LOCAL300 + ECBC/ESO; weighted rewards are not cross-alpha performance ranks');fig.tight_layout();fig.savefig(out/'overview.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,6));ax.plot(*route['xy'][ref].T,':',color='gray',label='Original reference')
for lane,(label,color) in enumerate(zip(labels,colors)):
 trace=r['lower_trace'][:,:,lane].reshape(-1,11);v=trace[trace[:,6].astype(bool)];ax.plot(v[:,0],v[:,1],color=color,label=label,linestyle=styles[lane],marker=palette_markers[lane],markevery=400,markersize=3);ax.scatter(*v[-1,:2],marker='x',color=color)
ax.set(xlabel='World x (m)',ylabel='World y (m)',title=a.title+': actual/reference XY');ax.axis('equal');ax.legend();fig.tight_layout();fig.savefig(out/'xy.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,8),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
for ax in axs:
 ax.plot(*route['xy'][ref].T,':',color='.55',label='Fixed original path')
 for lane,(label,color) in enumerate(zip(labels,colors)):
  trace=r['lower_trace'][:,:,lane].reshape(-1,11);v=trace[trace[:,6].astype(bool)];ax.plot(v[:,0],v[:,1],color=color,label=label,linestyle=styles[lane],marker=palette_markers[lane],markevery=400,markersize=3);ax.plot(*v[-1,:2],marker='x',color=color)
 ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='box')
turn=np.flatnonzero(abs(route['curvature'])>1e-5);bend=route['xy'][turn] if len(turn) else route['xy'][ref];axs[1].set(xlim=(bend[:,0].min()-.5,bend[:,0].max()+.7),ylim=(bend[:,1].min()-.3,bend[:,1].max()+1),title='Bend detail' if len(turn) else 'Straight route detail');axs[0].set_title('Full trajectory; × actual endpoint');axs[1].legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9);fig.suptitle(a.title+(' | ECBC+ESO lower (zero residual)' if all(c['settings'].get('zero_lower',[False])) else ' | frozen residual lower'));fig.savefig(out/'xy_all_alpha.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for lane,(label,color,ax) in enumerate(zip(labels,colors,axs.flat)):
 trace=r['lower_trace'][:,:,lane].reshape(-1,11);ids=np.flatnonzero(trace[:,6]);t=ids*.005+trace[ids,7]*physics_dt
 ax.plot(t,trace[ids,2],color=color,label='True forward speed');ax.plot(t,trace[ids,10],'--',color='.4',label='Rear axle × 0.1 m');ax.axhline(user_speed,color='black',linestyle=':',label='User request');ax.plot(t,trace[ids,8],'-.',color=color,label='Executed speed reference');ax.set(title=label,xlabel='Time (s)',ylabel='Speed (m/s)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle(a.title+' — true / wheel proxy / requested / governed speeds');fig.savefig(out/'speed_measurements.png',dpi=150);plt.close(fig)
(out/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

(out/'figure_manifest.json').write_text(json.dumps(dict(source=str((a.run/'trajectory.npz').resolve()),outputs=['xy.png','overview.png','xy_all_alpha.png','speed_measurements.png'],role='exploratory DEV, one initial seed, no uncertainty estimate',filtering='active_lower mask only, no smoothing; each lane stops at observed failure',metric_window='individual observed duration; unequal durations do not support direct success rankings',reward='policy solid, same-alpha cached Zero dashed' if a.baseline_run else 'native alpha-weighted rewards; no Zero in five-lane protocol' if count==5 else 'native alpha-weighted rewards; Zero scored alpha0 only',checkpoint=c['identity']['files']['actor.npz'],baseline_reuse=c.get('baseline_reuse'),matplotlib=matplotlib.__version__),indent=2))

if a.baseline_run:
 from sttw_control.isaac_path_reporting import replay_rewards
 cfg=json.loads((Path(c['bundle'])/'config.json').read_text());replayed,parts,closure=replay_rewards(r,cfg);scale=cfg['path_reward']['scale']*.005
 for lane,stem in enumerate(['alpha0','alpha025','alpha05','alpha075','alpha1']):
  zero=np.load(a.baseline_run/(stem+'_zero_rescored.npz'));trace=r['lower_trace'][:,:,lane].reshape(-1,11);ids=np.flatnonzero(trace[:,6]);time=ids*.005+trace[ids,7]*physics_dt
  baseline_parts={key:-scale*zero[key] for key in parts if key!='terminal_adjustment'};baseline_parts['terminal_adjustment']=zero['reward']-sum(baseline_parts.values())
  fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');components,cum_components,errors,speeds,tick,cum=axes.flat
  for key in parts:
   if key=='terminal_adjustment':continue
   line=components.plot(time,parts[key][ids,lane],label=key)[0];components.plot(zero['time'],baseline_parts[key],'--',color=line.get_color(),alpha=.65)
   cum_components.plot(time,np.cumsum(parts[key][ids,lane]),color=line.get_color());cum_components.plot(zero['time'],np.cumsum(baseline_parts[key]),'--',color=line.get_color(),alpha=.65)
  baseline=r['lower_trace'][:,:,5].reshape(-1,11);bid=np.flatnonzero(baseline[:,6]);bt=bid*.005+baseline[bid,7]*physics_dt
  errors.plot(time,trace[ids,3],label='Policy signed cross-track');errors.plot(bt,baseline[bid,3],'--',label='Zero signed cross-track')
  speeds.plot(time,trace[ids,2]-user_speed,label='Policy true speed error');speeds.plot(bt,baseline[bid,2]-user_speed,'--',label='Zero true speed error');speeds.plot(time,trace[ids,2]-trace[ids,10],':',label='Policy true minus wheel proxy')
  tick.plot(time,replayed[ids,lane],color=colors[lane]);tick.plot(zero['time'],zero['reward'],'--',color='.4');cum.plot(time,np.cumsum(replayed[ids,lane]),color=colors[lane]);cum.plot(zero['time'],np.cumsum(zero['reward']),'--',color='.4')
  for ax,title,ylabel in zip(axes.flat,['Ordinary reward components','Cumulative ordinary components','Path error','Speed errors','Reward per lower step','Cumulative reward'],['Reward','Reward','m','m/s','Reward','Reward']):ax.set(title=title,xlabel='Time (s)',ylabel=ylabel);ax.grid(alpha=.2)
  terrain_path=next(iter(c['settings']['terrains'].values()))
  terrain_meta={} if terrain_path=='plane' else json.loads((Path(terrain_path)/'identity.json').read_text())
  if terrain_meta.get('height_m',0)>0:
   if 's_range_m' in terrain_meta:
    start,end=terrain_meta['s_range_m']
   elif 'start_s_m' in terrain_meta:
    start=terrain_meta['start_s_m'];end=terrain_meta['end_s_m']
   elif 'slope_start_s_m' in terrain_meta:
    start=terrain_meta['slope_start_s_m'];end=terrain_meta['slope_end_s_m']
   else:
    end=terrain_meta['terrain_support_end_s_m'];start=2*terrain_meta['center_s_m']-end
   progress=r['progress'][ids,lane];span=time[(progress>=start)&(progress<=end)]
   if len(span):
    for ax in axes.flat:ax.axvspan(span[0],span[-1],color='.5',alpha=.12)
  if r['failure'][:,5].any():
   for ax in axes.flat:ax.axvline(bt[-1],color='.4',linestyle=':',alpha=.7)
  if r['failure'][:,lane].any():
   for ax in axes.flat:ax.axvline(time[-1],color=colors[lane],linestyle=':',alpha=.7)
  components.legend(fontsize=7);errors.legend(fontsize=7);speeds.legend(fontsize=7);fig.suptitle(a.title+' | '+labels[lane]+' | policy solid / same-alpha Zero dashed; terminal cost in totals\nGray band: policy projection in terrain support or ascent; dotted lines: failures')
  fig.savefig(out/(stem+'_diagnostics.png'),dpi=150);plt.close(fig)
  np.savez_compressed(out/(stem+'_replayed.npz'),time=time,reward=replayed[ids,lane],**{key:value[ids,lane] for key,value in parts.items()})
 manifest=json.loads((out/'figure_manifest.json').read_text());manifest.update(reward_replay_error=closure,paired_baseline_scores=str(a.baseline_run.resolve()),additional_outputs=[stem+'_diagnostics.png' for stem in ['alpha0','alpha025','alpha05','alpha075','alpha1']]);(out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))

(out/'condition_metrics.json').write_text(json.dumps(static_case_metrics(r,c),indent=2)+'\n')

if count==5 and a.baseline_run is None:
 from sttw_control.isaac_path_reporting import replay_rewards
 cfg=json.loads((Path(c['bundle'])/'config.json').read_text());replayed,parts,closure=replay_rewards(r,cfg)
 np.savez_compressed(out/'reward_components.npz',reward=replayed,**parts)
 for lane,(label,color) in enumerate(zip(labels,colors)):
  ids=np.flatnonzero(r['lower_trace'][:,:,lane,6].reshape(-1));time=(ids+1)*.005
  fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
  for key,value in parts.items():
   axes[0].plot(time,value[ids,lane],label=key);axes[1].plot(time,np.cumsum(value[ids,lane]))
  axes[0].legend(fontsize=7,ncol=3);axes[0].set(ylabel='Reward per lower step');axes[1].set(xlabel='Time (s)',ylabel='Cumulative component reward');fig.suptitle(a.title+' | '+label+' | true failed endpoints retained')
  fig.savefig(out/('alpha'+str(lane)+'_components.png'),dpi=150);plt.close(fig)
 manifest=json.loads((out/'figure_manifest.json').read_text());manifest.update(reward_replay_error=closure);(out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
