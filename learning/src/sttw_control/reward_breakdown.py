"""Reconstruct recorded reward components; no simulation or policy invocation."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.media import state_series
COLORS=['#0072B2','#009E73','#E69F00','#CC79A7','#D55E00']
PARTS=['roll','roll_rate','speed','path','heading','path_excess','steer','action']
LABELS=['Roll error','Roll rate','Speed','Path error','Heading','Path excess','Steer error','Residual action']
PART_COLORS=['#AA4499','#882255','#4477AA','#228833','#66CCEE','#CCBB44','#EE6677','#BBBBBB']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def reconstruct(path):
 tr=dict(np.load(path/'trace.npz'));d=json.loads((path/'declaration.json').read_text());c=d['config'];dt=c['controller']['dt'];p=c['priority']
 if p is None or p.get('risk_gate',True) or c.get('action_mapping') is not None or (c.get('circle') is None and c.get('figure_eight') is None and c.get('bend') is None):
  raise ValueError('reward breakdown currently requires a reference path, direct residual and disabled risk gate')
 if len(tr['time'])<2 or not np.allclose(np.diff(tr['time']),dt,rtol=1e-5,atol=1e-8):raise ValueError('invalid recorded control timestamps')
 n=len(tr['time']);a=tr['priority_alpha']
 if not np.all(a==a[0]):raise ValueError('alpha comparison requires constant alpha per recorded episode')
 series=state_series(tr,c);ev=series['speed']-tr['motion_command'][:,1];er=tr['measurement'][:,0]-tr['reference_roll'];rr=tr['measurement'][:,1];es=tr['measurement'][:,2]-tr['motion_command'][:,0]
 if c.get('bend') is not None:
  from .path import BendConfig,bend_trace_features
  features=bend_trace_features(tr['pose'],BendConfig(**c['bend']));ey,eh=features[:,:2].T
 elif c.get('figure_eight') is not None:
  from .path import FigureEightConfig,eight_trace_features
  features=eight_trace_features(tr['pose'],FigureEightConfig(**c['figure_eight']));ey,eh=features[:,:2].T
 else:
  xy=tr['pose'][:,:2]-[c['circle']['center_x'],c['circle']['center_y']];ey=np.linalg.norm(xy,axis=1)-c['circle']['radius'];tangent=np.arctan2(xy[:,1],xy[:,0])+c['circle']['direction']*np.pi/2;eh=np.arctan2(np.sin(tr['pose'][:,2]-tangent),np.cos(tr['pose'][:,2]-tangent))
 from .priority import PriorityConfig,objective_weights
 v,w=objective_weights(a,PriorityConfig(**p),xp=np)
 rates={'roll':10*er**2,'roll_rate':rr**2,'speed':v*c['speed_error_weight']*ev**2/p['speed_cost_scale'],'path':w*c['path_error_weight']*ey**2/p['path_cost_scale'],'heading':w*c['heading_error_weight']*eh**2/p['path_cost_scale'],'path_excess':w*c['path_excess_weight']*np.maximum(abs(ey)-c['path_soft_limit'],0)**2/p['path_cost_scale'],'steer':es**2,'action':.01*np.sum(tr['action']**2,axis=1)}
 reward={k:-dt*x for k,x in rates.items()};reward['alive']=np.full(n,dt*c.get('alive_reward_rate',1.));failed=tr['terminated'].astype(bool)
 for values in reward.values():values[failed]=0.;values[0]=0.
 reward['failure']=np.where(failed,-c['failure_penalty'],0.);reward['bonus']=np.zeros(n)
 rc=c['recovery'];count=0;latched=False
 for i in range(1,n):
  event=tr['event'][i];finished=(event[2]!=0 or event[3]!=0 or (len(event)>5 and event[5]!=0)) and i-1>=event[1]
  valid=finished and not failed[i] and abs(er[i])<=rc['roll_tolerance'] and abs(rr[i])<=rc['roll_rate_tolerance'] and abs(ev[i])<=rc['speed_tolerance'] and abs(es[i])<=rc['steer_tolerance']
  count=count+1 if valid else 0
  if count>=math.ceil(rc['hold_seconds']/dt) and not latched:reward['bonus'][i]=5.;latched=True
 predicted=sum(reward.values());diff=np.abs(predicted[1:]-tr['reward'][1:]);err=float(diff.max())
 if not np.allclose(predicted[1:],tr['reward'][1:],rtol=3e-5,atol=3e-5):raise ValueError(f'reward mismatch {path}: max={err}, index={int(diff.argmax())+1}')
 return dict(speed_true=series['speed'],speed_estimate=tr['measurement'][:,5]*.1,speed_target=tr['motion_command'][:,1],alpha=float(a[0]),t=tr['time'],radial=ey,speed=ev,roll=er,rates=rates,reward=reward,recorded=tr['reward'],failed=bool(failed[-1]),error=err,event=json.loads((path/'event.json').read_text()),path=path,config=c,dt=dt)

def generate(root):
 ROOT=Path(root)
 OUT=ROOT/'analysis/reward_breakdown';OUT.mkdir(parents=True,exist_ok=True)
 declaration=json.loads((ROOT/'declaration.json').read_text())
 expected=declaration['priority_alphas']
 endpoint=json.loads((ROOT/'training/status.json').read_text())['last_checkpoint']
 panels=list(sorted((ROOT/'evaluation').glob('alpha_*/seed_*')))
 seeds=json.loads((ROOT/'frozen/panel.json').read_text())['evaluation_seeds']
 identities=[]
 for panel in panels:
  d=json.loads((panel/'declaration.json').read_text())
  identities.append((d['priority_alpha_override'],d['panel']['seed']))
  if {p.parent.name for p in panel.glob('*/residual')}!=set(d['scenarios']):raise ValueError('missing scenario traces')
 if len(identities)!=len(expected)*len(seeds) or set(identities)!={(a,s) for a in expected for s in seeds}:raise ValueError('incomplete alpha/seed panels')
 groups={};summaries=[]
 for panel in panels:
  for path in sorted(panel.glob('*/residual')):
   x=reconstruct(path);groups.setdefault((panel.name,path.parent.name),[]).append(x)
   row={'seed':int(panel.name[5:]),'scenario':path.parent.name,'alpha':x['alpha'],'duration':float(x['t'][-1]),'failed':x['failed'],'max_reward_reconstruction_error':x['error'],'component_returns':{k:float(v.sum()) for k,v in x['reward'].items()},'recorded_return':float(x['recorded'][1:].sum()),'source_trace_sha256':hashlib.sha256((path/'trace.npz').read_bytes()).hexdigest()}
   summaries.append(row)
   dest=OUT/panel.name/path.parent.name;dest.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(dest/f"alpha_{x['alpha']:g}.npz",time=x['t'],speed_true=x['speed_true'],speed_estimate=x['speed_estimate'],speed_target=x['speed_target'],path_error=x['radial'],speed_error=x['speed'],roll_error=x['roll'],recorded_reward=x['recorded'],**{f'reward_{k}':v for k,v in x['reward'].items()})
 
 if not groups:raise ValueError('no recorded alpha evaluations')
 indexes=[]
 for (seed,case),xs in sorted(groups.items()):
  xs.sort(key=lambda x:x['alpha'])
  if [x['alpha'] for x in xs]!=sorted(expected):raise ValueError('incomplete alpha group')
  if any(x['config']!=xs[0]['config'] or x['event']!=xs[0]['event'] for x in xs):raise ValueError('comparison configs/events differ')
  dest=OUT/seed/case
  fig,axs=plt.subplots(4,2,figsize=(16,15),layout='constrained')
  titles=['Signed path error [m]','True speed error [m/s]','Roll-reference error [deg]','Weighted path-group penalty [reward / s]','Weighted speed penalty [reward / s]','Weighted attitude penalty [reward / s]','Recorded cumulative reward','Mean penalty composition: SAME time window']
  end=min(x['t'][-1] for x in xs);common=end-xs[0]['dt'] if any(x['failed'] and abs(x['t'][-1]-end)<1e-5 for x in xs) else end
  totals=[]
  for x,color in zip(xs,[COLORS[i%len(COLORS)] for i in range(len(xs))]):
   t=x['t'][1:];label=f"alpha={x['alpha']:g}";rates=x['rates']
   curves=[x['radial'][1:],x['speed'][1:],np.rad2deg(x['roll'][1:]),sum(rates[k][1:] for k in ['path','heading','path_excess']),rates['speed'][1:],rates['roll'][1:]+rates['roll_rate'][1:],np.cumsum(x['recorded'][1:])]
   for ax,y in zip(axs.flat,curves):
    ax.plot(t,y,color=color,label=label,lw=1.3)
    if x['failed']:ax.plot(t[-1],y[-1],marker='x',color=color,ms=8,mew=2)
   select=(x['t']>0)&(x['t']<=common+1e-8)
   totals.append([float(rates[k][select].mean()) for k in PARTS])
   for r in summaries:
    if r['seed']==int(seed[5:]) and r['scenario']==case and r['alpha']==x['alpha']:
     r['common_window_end']=float(common);r['common_window_penalty_rates']={k:v for k,v in zip(PARTS,totals[-1])}
 
  for ax,title in zip(axs.flat,titles):ax.set_title(title,loc='left',fontweight='bold');ax.grid(alpha=.2)
  for ax in list(axs.flat)[:7]:
   ax.set_xlabel('Time [s]');ax.set_xlim(0,max(x['config']['horizon_seconds'] for x in xs))
   if case!='nominal':ax.axvspan(xs[0]['event']['start_seconds'],xs[0]['event']['end_seconds'],color='#999999',alpha=.18)
  for ax in [axs[1,1],axs[2,0],axs[2,1]]:ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0)
  axs[0,0].axhline(0,color='black',lw=.6);axs[0,1].axhline(0,color='black',lw=.6)
  base=np.zeros(len(xs))
  for i,(name,color) in enumerate(zip(LABELS,PART_COLORS)):
   values=np.array(totals)[:,i];axs[3,1].bar(np.arange(len(xs)),values,bottom=base,color=color,label=name);base+=values
  axs[3,1].set_xticks(np.arange(len(xs)),[f'{x["alpha"]:g}' for x in xs]);axs[3,1].set_xlabel(f'alpha | common window 0-{common:.3f} s');axs[3,1].set_ylabel('Mean penalty [reward / s]');axs[3,1].legend(fontsize=8,ncol=2)
  axs[0,0].legend(ncol=3,fontsize=9)
  failures=', '.join(f"alpha={x['alpha']:g}: {x['t'][-1]:.3f}s" for x in xs if x['failed']) or 'none'
  fig.suptitle(f"Frozen {Path(endpoint).name} | {case} | {seed}\nGrey: disturbance; x: terminal failure ({failures}). Lines end at recorded termination.\nPenalty panels use symlog scale; cumulative reward includes +{xs[0]['config'].get('alive_reward_rate',1.):g}/s, recovery bonus and -{xs[0]['config']['failure_penalty']:g} failure replacement.",fontsize=12)
  for ext in ['png','pdf']:fig.savefig(dest/f'overview.{ext}',dpi=155)
  plt.close(fig)
  fig,axes=plt.subplots(len(xs),2,figsize=(15,3.3*len(xs)),squeeze=False,layout='constrained',sharex=True)
  for row,x in zip(axes,xs):
   t=x['t']
   bp=x['path'].parent/'baseline'
   bt=dict(np.load(bp/'trace.npz'));bc=json.loads((bp/'declaration.json').read_text())['config']
   bs=state_series(bt,bc);be=bt['measurement'][:,5]*.1
   row[0].plot(bt['time'],bs['speed'],label='Baseline true forward',color='#009E73',ls='-',lw=1.3)
   row[0].plot(bt['time'],be,label='Baseline wheel estimate',color='#CC79A7',ls=':',lw=1.3)
   row[1].plot(bt['time'],be-bs['speed'],label='Baseline estimate - true',color='#009E73')
   if bt['terminated'][-1]:
    row[0].plot(bt['time'][-1],bs['speed'][-1],'x',color='#009E73')
    row[0].plot(bt['time'][-1],be[-1],'x',color='#CC79A7')
    row[1].plot(bt['time'][-1],(be-bs['speed'])[-1],'x',color='#009E73')
   np.savez_compressed(dest/f"speed_pair_alpha_{x['alpha']:g}.npz",baseline_time=bt['time'],baseline_true=bs['speed'],baseline_estimate=be,baseline_target=bs['reference'][:,1],residual_time=t,residual_true=x['speed_true'],residual_estimate=x['speed_estimate'],residual_target=x['speed_target'])
   for key,label,color,style in [('speed_target','Target','#333333','--'),('speed_true','Residual true forward','#0072B2','-'),('speed_estimate','Residual wheel estimate','#D55E00',':')]:
    row[0].plot(t,x[key],label=label,color=color,ls=style,lw=1.5)
    if x['failed']:row[0].plot(t[-1],x[key][-1],'x',color=color)
   error=x['speed_estimate']-x['speed_true']
   row[1].plot(t,error,color='#D55E00',label='Residual estimate - true')
   row[1].axhline(0,color='black',lw=.6)
   if x['failed']:row[1].plot(t[-1],error[-1],'x',color='#D55E00')
   row[0].set_title(f"alpha={x['alpha']:g} | baseline vs residual")
   row[1].set_title(f"alpha={x['alpha']:g} | estimation error")
   for ax in row:
    ax.set_ylabel('m/s');ax.set_xlabel('Time [s]');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set_xlim(0,x['config']['horizon_seconds'])
    if case!='nominal':ax.axvspan(x['event']['start_seconds'],x['event']['end_seconds'],color='grey',alpha=.18)
  fig.suptitle(f"{Path(endpoint).name} | {case} | {seed}\nTrue: body-forward velocity projection; estimate: measured rear shaft rate x 0.1 m. Grey: disturbance; x: failure.")
  for ext in ['png','pdf']:fig.savefig(dest/f'speed_comparison.{ext}',dpi=155)
  plt.close(fig)
  indexes.append(f'- {seed} / {case}: [overview]({seed}/{case}/overview.png) | [PDF]({seed}/{case}/overview.pdf) | [speed comparison]({seed}/{case}/speed_comparison.png) | [speed PDF]({seed}/{case}/speed_comparison.pdf)')
 (OUT/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
 (OUT/'INDEX.md').write_text('# Reward and error diagnostics\n\nAll declared alpha groups and available scenario/seed panels. No new simulation. Reward reconstructed from recorded states/actions and checked against each logged transition.\n\n'+ '\n'.join(indexes)+'\n\nNPZ files contain signed per-step components; penalty plots are positive rates before terminal replacement. The failure transition replaces all regular terms with the frozen failure penalty. Common-window bars exclude the earliest terminal transition. Recovery bonus uses roll/rate/speed/steer criteria, not path recovery criteria.\n')
 print('checked',len(summaries),'trajectories; figures',len(groups),'max reward residual',max(x['max_reward_reconstruction_error'] for x in summaries))
 return {'trajectories':len(summaries),'figures':len(groups),'max_reward_error':max(x['max_reward_reconstruction_error'] for x in summaries)}
