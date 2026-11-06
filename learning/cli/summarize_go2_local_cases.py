"""Compare selected saved candidates on fixed nominal cases (DEV, no simulation)."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--start-relative',action='store_true',help='Translate each trajectory and original reference to its own start for changed-platform comparisons; no rotation or path reshaping');p.add_argument('--reference-label',default='MuJoCo Local300');p.add_argument('--config',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--zero',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot-top',type=int,default=0);a=p.parse_args()
s=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
colors=['#0072B2','#E69F00','#009E73','#CC79A7'];styles=['-','--','-.',':'];results=[];traces=[]
for folder in a.runs:
 identity=json.loads((folder/'identity.json').read_text());d=np.load(folder/'trajectory.npz');cases=[]
 for i,name in enumerate(s['cases']):
  n=int(d['valid'][:,i].sum());ev=d['speed'][:n,i]-d['reference'][:n,i,0];ed=d['measurement'][:n,i,2]-d['reference'][:n,i,1];failed=bool(d['failed'][:,i].any());survived=not failed and n==round(s['seconds']/s['control_dt'])
  tolerance=.01 if name.startswith('post_return_small') else .02
  hold=survived and bool((np.abs(ev[-100:])<=.05).all() and (np.abs(ed[-100:])<=tolerance).all())
  old=np.load(Path(s['mujoco_reference'])/(name+'.npz'));actual=d['pose'][:n,i,:2];reference=old['reference_xy'][:n]
  if a.start_relative:actual=actual-actual[0];reference=reference-reference[0]
  xy=actual-reference
  cases.append(dict(case=name,survived=survived,final_tracking_hold=hold,end_s=float(d['time'][n-1,i]+d['stop_tick'][n-1,i]*s['physics_dt']),speed_rmse=float(np.sqrt(np.mean(ev**2))),steer_rmse=float(np.sqrt(np.mean(ed**2))),xy_reference_rmse=float(np.sqrt(np.mean(np.sum(xy**2,axis=-1)))),peak_roll=float(d['peak_roll'][:n,i].max())))
 score=dict(survived=sum(c['survived'] for c in cases),tracking_hold=sum(c['final_tracking_hold'] for c in cases),observed_seconds=sum(c['end_s'] for c in cases),tracking=float(np.mean([c['speed_rmse']/.05+c['steer_rmse']/.02 for c in cases])))
 results.append(dict(**identity,run=str(folder.resolve()),cases=cases,score=score));traces.append(d)
rank=sorted(results,key=lambda r:(-r['score']['survived'],-r['score']['tracking_hold'],-r['score']['observed_seconds'],r['score']['tracking']))
plot_rows=list(zip(traces,results))
if a.plot_top:
 chosen={row['label'] for row in rank[:a.plot_top]}
 plot_rows=[(d,r) for d,r in plot_rows if r['label'] in chosen]
def coordinates(x):return x-x[0] if a.start_relative else x
zero=np.load(a.zero/'trajectory.npz');fig,axes=plt.subplots(2,4,figsize=(19,10));axes.flat[-1].axis('off')
for i,name in enumerate(s['cases']):
 ax=axes.flat[i];old=np.load(Path(s['mujoco_reference'])/(name+'.npz'));ax.plot(*coordinates(old['reference_xy']).T,':',color='.5',label='Original reference');ax.plot(*coordinates(old['actual_xy']).T,'--',color='black',lw=1.4,label=a.reference_label)
 nz=int(zero['valid'][:,i].sum());ax.plot(*coordinates(zero['pose'][:nz,i,:2]).T,color='.6',lw=1,label='Lab ECBC+ESO')
 for (d,result),color,style in zip(plot_rows,colors,styles):
  n=int(d['valid'][:,i].sum());xy=coordinates(d['pose'][:n,i,:2]);ax.plot(*xy.T,style,color=color,lw=1.4,label=(result['label'] if 'controller_parameters' in result else 'Update '+str(result['completed_updates'])));ax.scatter(*xy[-1],color=color,marker='x',s=18)
 ax.set(title=name,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=10)
fig.suptitle(('ECBC parameter screen, plotted top '+str(a.plot_top) if a.plot_top else ('ECBC parameter candidates' if all('controller_parameters' in r for r in results) else 'Lower-policy candidates'))+' | fixed nominal DEV\n'+('Start-relative translation only; changed platform, fresh warmup; ' if a.start_relative else 'Same saved initial state and issued references; ')+'crosses mark actual observation endpoints')
fig.tight_layout();fig.savefig(a.output/'all_candidates_xy.png',dpi=160);plt.close(fig)
result=dict(start_relative=a.start_relative,physics_dt=s['physics_dt'],control_dt=s['control_dt'],role='fixed-case DEV comparison, not held-out generalization',best=rank[0],ranked=rank,
 selection='survival count, final .5s speed/steer tracking count, observed duration, normalized tracking RMSE',
 adoption='not qualified if any case lacks survival or tracking hold; best only relative to tested candidates',
 limitation='Each error uses its actual observation window; early failure is ranked before error. XY uses original issued-reference trajectory, not a reward target.')
(a.output/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps([dict(label=r['label'],update=r['completed_updates'],**r['score']) for r in rank],indent=2))

# Command response is evaluated against this chassis, independently of old XY.
controller=json.loads(Path(s['controller_config']).read_text())
radius=controller['wheel_radius_m'];dt=s['control_dt'];response=[]
response_rows=[(zero,dict(label='ECBC+ESO'))]+list(zip(traces,results))
for i,name in enumerate(s['cases']):
 fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
 for (d,identity),color,style in zip(response_rows,['.45']+colors,[':']+styles):
  n=int(d['valid'][:,i].sum());t=d['time'][:n,i]+dt;r=d['reference'][:n,i];m=d['measurement'][:n,i]
  yaw=np.diff(np.unwrap(d['pose'][:n,i,2]))/dt
  target=r[:,0]*np.tan(r[:,1])*np.cos(controller['caster'])/controller['wheelbase']
  label=identity['label'];speed=d['speed'][:n,i];wheel=m[:,5]*radius;command=d['command'][:n,i,1]*radius
  axes[0].plot(t,speed,style,color=color,label=label)
  axes[1].plot(t,m[:,2],style,color=color)
  axes[2].plot(t[1:],yaw,style,color=color)
  axes[3].plot(t,command,style,color=color,label=label+' command')
  axes[3].plot(t,wheel,color=color,alpha=.35,lw=.7)
  tail=max(1,round(.5/dt))
  response.append(dict(model=label,case=name,source=str((a.zero if label=='ECBC+ESO' else Path(identity['run']))/'trajectory.npz'),
   final_window_s=.5,speed_mean=float(speed[-tail:].mean()),requested_speed=float(r[-tail:,0].mean()),
   wheel_command_speed=float(command[-tail:].mean()),wheel_measured_speed=float(wheel[-tail:].mean()),
   steer_mean=float(m[-tail:,2].mean()),requested_steer=float(r[-tail:,1].mean()),
   yawrate_mean=float(yaw[-tail:].mean()),kinematic_yawrate_target=float(target[-tail:].mean())))
 axes[0].plot(t,r[:,0],'k--',label='Request');axes[1].plot(t,r[:,1],'k--')
 axes[2].plot(t,target,'k--',label='Current-chassis kinematic target')
 for ax,label in zip(axes,['Body speed (m/s)','Steering (rad)','Yaw rate (rad/s)','Wheel speed × radius (m/s)']):
  ax.set_ylabel(label);ax.grid(alpha=.2)
 axes[0].legend(ncol=3);axes[2].legend();axes[3].set_xlabel('Time (s)')
 fig.suptitle(name+' | nominal DEV; all saved candidates\nWheel panel: strong lines = command, faint lines = measured; yaw uses backward difference')
 fig.tight_layout();fig.savefig(a.output/(name+'_response.png'),dpi=140);plt.close(fig)
(a.output/'response.json').write_text(json.dumps(dict(controller=controller,rows=response,
 limitation='Kinematic yaw target is diagnostic, not a physical feasibility guarantee. Old chassis XY is historical only. No smoothing; no simulation rerun.'),indent=2))
