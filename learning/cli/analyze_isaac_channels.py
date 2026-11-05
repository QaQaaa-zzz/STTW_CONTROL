"""Analyze completed common-state channel cases from saved physics only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.isaac_path_reporting import replay_rewards,raised_contact_phases
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();status=json.loads((a.run/'status.json').read_text());cases=status['cases'];spec=json.loads((a.run/'protocol.json').read_text())
if not cases:p.error('no completed physics cases to analyze')
out=a.run/'analysis';out.mkdir(exist_ok=True);labels=spec['case_labels'];colors=['#333333','#777777','#D55E00','#0072B2','#009E73'];styles=['-',':','--','-.','-'];alpha_colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500'];all_metrics=[];all_lanes=[];report=['# Original H75 common-state channel mechanism DEV','','Saved original H75 u77 and selected orange Local300; no new optimizer updates. '+str(len(cases))+'/10 cases collected. Each case uses one alpha and one common exported state, not a shared state across alpha. PhysX caches are not copied. Natural continuation versus restored unmodified control measures the reload effect; intervention contrasts use the restored control. All raw prefixes remain saved; branch outcomes start at the fork.','','Contact stages use official mean wheel/filter points at0.2ms, force>1e-3N, local z>2.5mm and positive terrain height. Fork locations are projection triggers, not measured contact events. One initial state per condition; no uncertainty or generalization claim.']
for case in cases:
 run=Path(case['run']);raw=np.load(run/'trajectory.npz');contract=json.loads((run/'contract.json').read_text());cfg=json.loads((Path(contract['bundle'])/'config.json').read_text());route=np.load(Path(contract['bundle'])/'route.npz');target=out/case['phase']/('alpha'+str(case['alpha']));target.mkdir(parents=True,exist_ok=True);entry=dict(phase=case['phase'],alpha=case['alpha'],run=str(run),fork_reached=case['status']['fork_policy_step'] is not None);all_metrics.append(entry)
 trace=raw['lower_trace'].reshape(-1,5,11)
 if not entry['fork_reached']:
  ids=np.flatnonzero(trace[:,0,6]);fig,ax=plt.subplots();ax.plot(trace[ids,0,0],trace[ids,0,1]);ax.plot(*trace[ids[-1],0,:2],marker='x');ax.set(xlabel='World x (m)',ylabel='World y (m)',title='Fork not reached; natural continuation only');ax.set_aspect('equal');fig.savefig(target/'xy.png',dpi=150);plt.close(fig);report+=['',f'## {case["phase"]} alpha {case["alpha"]}: fork not reached',f'![XY]({target.relative_to(out)}/xy.png)'];continue
 fork=json.loads((run/'fork.json').read_text());saved=np.load(run/'fork_snapshot.npz');step=fork['policy_step'];start=step*4;t0=step*.02;hold=fork['protocol']['hold_s'];reward,parts,closure=replay_rewards(raw,cfg,(step,[1,2,3,4],saved['policy.offsets']));entry.update(fork_time_s=t0,fork_progress_m=float(saved['middle.projection.progress']),reward_replay_error=closure,branches=[]);lanes=[]
 for lane,label in enumerate(labels):
  ids=np.flatnonzero(trace[:,lane,6]&(np.arange(len(trace))>=start));time=ids*.005+trace[ids,lane,7]*.0002
  data=dict(ids=ids,time=time,progress=raw['progress'][ids,lane],xy=trace[ids,lane,:2],speed_error=trace[ids,lane,2]-2.3,cross=trace[ids,lane,3],roll=trace[ids,lane,4],target=raw['target'][ids,lane],offsets=raw['offsets'][ids,lane],reward=reward[ids,lane]);lanes.append(data);contact=raised_contact_phases(raw,contract,lane)
  row=dict(branch=label,failed_after_fork=bool(raw['failure'][step:,lane].any()),observed_end_s=float(time[-1]),postfork_speed_rmse_m_s=float(np.sqrt(np.mean(data['speed_error']**2))),postfork_path_rmse_m=float(np.sqrt(np.mean(data['cross']**2))),peak_roll_rad=float(data['roll'].max()),held_lower_ticks=int(raw['held_channels'][ids,lane].any(-1).sum()),contacts=contact,windows={})
  common=np.intersect1d(ids,lanes[1]['ids'] if lane>1 else np.flatnonzero(trace[:,1,6]&(np.arange(len(trace))>=start)))
  for name,left,right in [('hold',t0,t0+hold),('after_release_0p5',t0+hold,t0+hold+.5),('whole_common',t0,float('inf'))]:
   ticks=common[(common*.005+.005>left)&(common*.005+.005<=right+1e-8)]
   if len(ticks):
    delta=trace[ticks,lane,:2]-trace[ticks,1,:2];row['windows'][name]=dict(lower_ticks=len(ticks),end_s=float(ticks[-1]*.005+min(trace[ticks[-1],lane,7],trace[ticks[-1],1,7])*.0002),xy_difference_rmse_m=float(np.sqrt(np.mean((delta**2).sum(-1)))),xy_difference_max_m=float(np.linalg.norm(delta,axis=-1).max()),speed_error_rmse_m_s=float(np.sqrt(np.mean((trace[ticks,lane,2]-2.3)**2))),path_rmse_m=float(np.sqrt(np.mean(trace[ticks,lane,3]**2))))
  entry['branches'].append(row)
 entry['first_postfork_observation_difference_natural_vs_restored']=float(abs(raw['observation'][start,0]-raw['observation'][start,1]).max());all_lanes.append((case,lanes,route,t0))
 fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
 for ax in axes:
  ref=route['s']<=float(raw['progress'].max())+1;ax.plot(*route['xy'][ref].T,':',color='.6',label='Original fixed path');prefix=np.flatnonzero(trace[:start,0,6]);ax.plot(trace[prefix,0,0],trace[prefix,0,1],color='.7',label='Natural prefix')
  for lane,data in enumerate(lanes):ax.plot(*data['xy'].T,color=colors[lane],linestyle=styles[lane],label=labels[lane]);ax.plot(*data['xy'][-1],color=colors[lane],marker='x')
  ax.plot(*saved['middle.pose'][:2],marker='D',color='black',fillstyle='none');ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal')
 bend=route['xy'][abs(route['curvature'])>1e-5];axes[0].set_title('Full XY; x observed endpoints');axes[1].set(xlim=(bend[:,0].min()-.5,bend[:,0].max()+.8),ylim=(bend[:,1].min()-.3,bend[:,1].max()+1),title='Bend detail; diamond fork');axes[1].legend(fontsize=7)
 title=f"Original H75 | {case['phase']} | alpha {case['alpha']} | fork {t0:.2f}s, hold {hold:.2f}s";fig.suptitle(title);fig.savefig(target/'xy.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(3,3,figsize=(16,11),layout='constrained')
 for lane,data in enumerate(lanes):
  series=[data['speed_error'],data['cross'],data['roll'],data['target'][:,0],data['target'][:,1],data['offsets'][:,0],data['offsets'][:,1],data['reward'],np.cumsum(data['reward'])]
  for ax,values in zip(axes.flat,series):ax.plot(data['time']-t0,values,color=colors[lane],linestyle=styles[lane],label=labels[lane])
 for ax,label in zip(axes.flat,['True speed error (m/s)','Signed cross-track (m)','Peak roll (rad)','Mapped speed target (m/s)','Mapped steer target (rad)','Executed speed correction (m/s)','Executed steer correction (rad)','Reward per lower step','Cumulative reward']):
  ax.set(title=label,xlabel='Time after fork (s)');ax.axvspan(0,hold,color='#0072B2',alpha=.08);ax.axvline(hold,color='.4',linestyle=':');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle(title+' | shaded: hold; filter/history/ESO continue');fig.savefig(target/'responses.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(2,5,figsize=(20,8),layout='constrained')
 for lane,data in enumerate(lanes):
  for key,values in parts.items():
   if key=='terminal_adjustment':continue
   line=axes[0,lane].plot(data['time']-t0,values[data['ids'],lane],label=key)[0];axes[1,lane].plot(data['time']-t0,np.cumsum(values[data['ids'],lane]),color=line.get_color())
  axes[0,lane].set(title=labels[lane],xlabel='Time after fork (s)',ylabel='Ordinary reward / step');axes[1,lane].set(xlabel='Time after fork (s)',ylabel='Cumulative ordinary reward')
 axes[0,0].legend(fontsize=7);fig.suptitle(title+' | signed components; terminal penalties retained in response totals');fig.savefig(target/'components.png',dpi=150);plt.close(fig)
 (target/'metrics.json').write_text(json.dumps(entry,indent=2));report+=['',f'## {case["phase"]} alpha {case["alpha"]}',f'![XY]({target.relative_to(out)}/xy.png)',f'[Responses]({target.relative_to(out)}/responses.png), [reward components]({target.relative_to(out)}/components.png), [paired windows and contact phases]({target.relative_to(out)}/metrics.json).','','| Branch | Failed after fork | End s | Speed RMSE m/s | Path RMSE m | Hold ticks |','|---|---|---:|---:|---:|---:|']
 for row in entry['branches']:report.append(f"|{row['branch']}|{row['failed_after_fork']}|{row['observed_end_s']:.3f}|{row['postfork_speed_rmse_m_s']:.4f}|{row['postfork_path_rmse_m']:.4f}|{row['held_lower_ticks']}|")
for phase in spec['fork_phases_m']:
 selected=[item for item in all_lanes if item[0]['phase']==phase]
 if not selected:continue
 fig,axes=plt.subplots(2,3,figsize=(13,12),layout='constrained');axes.flat[-1].axis('off')
 for lane,ax in enumerate(list(axes.flat)[:5]):
  route=selected[0][2];limit=max(float(data[lane]['progress'].max()) for _,data,_,_ in selected)+1;ax.plot(*route['xy'][route['s']<=limit].T,':',color='.6',label='Original path')
  for case,data,_,t0 in selected:
   color=alpha_colors[spec['alphas'].index(case['alpha'])];ax.plot(*data[lane]['xy'].T,color=color,label='alpha '+str(case['alpha']));ax.plot(*data[lane]['xy'][-1],color=color,marker='x')
  ax.set(title=labels[lane],xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal')
 axes[0,0].legend(fontsize=8);fig.suptitle('Original H75 | '+phase+' | post-fork trajectories grouped by branch; alpha colors fixed');fig.savefig(out/(phase+'_all_alpha.png'),dpi=150);plt.close(fig)
 report+=['',f'![{phase} alpha overlay]({phase}_all_alpha.png)']
(out/'metrics.json').write_text(json.dumps(dict(completed_cases=len(cases),expected_cases=10,cases=all_metrics,scope='single-initial-state nominal DEV; descriptive paired branch windows; no automatic superiority or generalization claim'),indent=2));(out/'RESULTS.md').write_text('\n'.join(report)+'\n');print(json.dumps(dict(completed_cases=len(cases),forked_cases=sum(row['fork_reached'] for row in all_metrics),output=str(out))))
