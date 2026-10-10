#!/usr/bin/env python3
"""Existing trace reporting only; exact same-index upper/lower attribution."""
import argparse,sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import jax
import jax.numpy as jp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.local_integration_diagnostics import channel_stats,kappa,yaw_decomposition,DT
from sttw_control.preference_best import trace_metrics,atomic_json,digest
from sttw_control.lower_tracking_audit import intervals,held
from sttw_control.preference_command_reporting import reward_audit
from sttw_control.direct_command_policy import q_ratio
from sttw_control.controller import ControllerConfig
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--upper-run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.run);upper=Path(a.upper_run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
status=json.loads((r/'status.json').read_text());assert status['state'] in ['completed','stopped_mismatch'],status
manifest=json.loads((r/'manifest.json').read_text());result={};arrays={}
for name in ['integration_candidate.json','manifest.json','status.json']:
 shutil.copyfile(r/name,out/name)
for case in ['straight_hold','fast_turn']:
 if not (r/case).exists():continue
 old={f'old_upper{i}':dict(np.load(upper/'evaluationbest'/case/f'alpha{i}.npz')) for i in (0,1)}
 old['B0']=dict(np.load(upper/'evaluationbest'/case/'B0.npz'))
 new={f'new_upper{i}':dict(np.load(r/case/f'alpha{i}.npz')) for i in (0,1)}
 for method,d in {**old,**new}.items():
  alpha=int(method[-1]) if method!='B0' else 0
  result[case+'/'+method]={'original_protocol':trace_metrics(d,alpha,case)}
  if method.startswith('new'):
   audit=reward_audit(d,alpha,upper,16);assert audit['passed'],audit;result[case+'/'+method]['reward_audit']=audit
  t=np.asarray(d['time'],float);raw=np.asarray(d['limited_command'],float);gov=np.asarray(d['governed'],float);actual=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)
  uppererr=gov-raw;lowererr=actual-gov;taskerr=actual-raw;np.testing.assert_allclose(taskerr,uppererr+lowererr,atol=1e-12,rtol=0)
  windows={}
  for hi in ([10,16] if case=='fast_turn' else [10]):
   mask=t<hi-1e-6;final=(t>=hi-.5-1e-6)&mask;frames=int(mask.sum());failure=bool(np.any(d['physical_failure'][mask]));peak=np.asarray(d['peak_roll'])
   metrics={kind:{ch:channel_stats(e[:,j],t,mask,tol) for ch,j,tol in [('speed',0,.1),('steer',1,.04)]} for kind,e in [('upper',uppererr),('lower',lowererr),('task',taskerr)]}
   hold=bool(final.sum()>=100 and not failure and np.all(abs(taskerr[final,0])<=.1)&np.all(abs(taskerr[final,1])<=.04)&np.all(abs(d['e_psi_unwrapped'][final])<=.05)&np.all(peak[final]<=.302))
   if frames:
    last=np.flatnonzero(mask)[-1];metrics.update(complete=frames>=round(hi/DT),observed_ticks=frames,physical_failure=failure,joint_final_hold=hold,peak_roll=float(np.max(peak[mask])),endpoint_xy_error=(d['actual_xy'][last]-d['reference_xy'][last]).tolist(),endpoint_xy_distance=float(np.linalg.norm(d['actual_xy'][last]-d['reference_xy'][last])),endpoint_heading_error=float(d['e_psi_unwrapped'][last]))
   windows[str(hi)]=metrics
  result[case+'/'+method]['windows']=windows
  if method.startswith('new'):
   quiet=held(np.all(abs(d['lower_governed_rates'])<=np.array([.1,.02]),axis=1),DT,.5)
   result[case+'/'+method]['quiet_lower']={ch:channel_stats(lowererr[:,j],t,quiet,tol) for ch,j,tol in [('speed',0,.05),('steer',1,.02)]}
   ratio=np.asarray(jax.vmap(lambda v:q_ratio(v,ControllerConfig()))(jp.asarray(gov[:,0])));phi_request=gov[:,1]/ratio
   checks=dict(speed_edge=(gov[:,0]<1.5)|(gov[:,0]>3.),steer_edge=abs(gov[:,1])>.35,roll_reference_screen=abs(phi_request)>.26+1e-6,speed_slew=abs(d['lower_governed_rates'][:,0])>1.8+1e-5,steer_slew=abs(d['lower_governed_rates'][:,1])>1.25+1e-5)
   union=np.logical_or.reduce(list(checks.values()));result[case+'/'+method]['coverage']={key:dict(seconds=float(np.sum(mask)*DT),intervals=intervals(mask,t,DT)) for key,mask in {**checks,'union':union}.items()}
   result[case+'/'+method]['coverage']['range_summary']=dict(speed=[float(gov[:,0].min()),float(gov[:,0].max())],steer=[float(gov[:,1].min()),float(gov[:,1].max())],phi_request=[float(phi_request.min()),float(phi_request.max())])
   raw_rate=raw[:,0]*kappa(raw[:,1]);gov_rate=gov[:,0]*kappa(gov[:,1]);actual_rate=np.asarray(d['yaw_rate'],float)
   A=kappa(gov[:,1])*(gov[:,0]-actual[:,0]);B=actual[:,0]*(kappa(gov[:,1])-kappa(actual[:,1]));C=actual[:,0]*kappa(actual[:,1])-actual_rate
   hupper=raw_rate-gov_rate;hpieces=np.column_stack([hupper,A,B,C]);hcumulative=np.cumsum(hpieces*DT,axis=0)
   closure=float(np.max(abs(hcumulative.sum(axis=1)-d['e_psi_unwrapped'])))
   result[case+'/'+method]['heading_split']=dict(contribution_names=['upper raw-governed yaw request','A speed','B steer','C kinematic remainder'],end=hcumulative[-1].tolist(),closure_max_abs_error=closure,initial_error=0.,C_interpretation='kinematic/sampling/contact remainder; not tire slip rate')
   result[case+'/'+method]['force']=dict(peak_abs=np.maximum(abs(d['actuator_force_min']),abs(d['actuator_force_max'])).max(0).tolist(),limit_substeps=d['actuator_force_limit_substeps'].sum(0).tolist(),final_clip_fraction=float(np.mean(d['final_command_clipped'])))
   arrays.update({case+'_'+method+'__'+k:v for k,v in dict(time=t,upper_error=uppererr,lower_error=lowererr,task_error=taskerr,heading_contributions=hcumulative,phi_request=phi_request).items()})
   dest=out/'data'/case;dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(r/case/f'alpha{alpha}.npz',dest/f'alpha{alpha}.npz')
 # First row XY includes both endpoints, old matched upper/R196 and zero-upper R196.
 for alpha in (0,1):
  for horizon in ([16,10] if case=='fast_turn' else [10]):
   d=new[f'new_upper{alpha}'];b=old[f'old_upper{alpha}'];base=old['B0'];fig,axes=plt.subplots(5,2,figsize=(15,18));ax=axes[0,0]
   colors={'new_upper0':'tab:blue','new_upper1':'tab:orange','old_upper0':'tab:cyan','old_upper1':'tab:red','B0':'0.45'}
   mask=base['time']<horizon-1e-6;ax.plot(*base['reference_xy'][mask].T,'k--',label='original fixed raw reference')
   for name,trace in {**old,**new}.items():
    mask=trace['time']<horizon-1e-6;xy=trace['actual_xy'][mask];ax.plot(*xy.T,color=colors[name],ls='-' if name.startswith('new') else ':',label=name+' + '+('fixed300' if name.startswith('new') else 'R196'))
    if np.any(trace['physical_failure'][mask]):ax.plot(*xy[-1],marker='x',color=colors[name])
   ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.legend(fontsize=8);ax.grid(alpha=.3)
   t=d['time'];mask=t<horizon-1e-6
   for j,label in [(0,'speed (m/s)'),(1,'steer (rad)')]:
    ax=axes[1,j];ax.plot(t[mask],d['limited_command'][mask,j],'k--',label='raw');ax.plot(t[mask],d['governed'][mask,j],label='new governed');ax.plot(t[mask],np.column_stack([d['actual_forward_speed'],d['actual_delta']])[mask,j],label='new actual')
    bm=b['time']<horizon-1e-6;ax.plot(b['time'][bm],np.column_stack([b['actual_forward_speed'],b['actual_delta']])[bm,j],':',label='old upper+R196 actual');ax.set_ylabel(label)
    err=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)-d['limited_command'];lo=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)-d['governed'];up=d['governed']-d['limited_command']
    ax=axes[2,j]
    for values,l in [(up,'upper governed-raw'),(lo,'lower actual-governed'),(err,'task actual-raw')]:ax.plot(t[mask],values[mask,j],label=l)
    ax.axhline(0,color='k',lw=.5);ax.set_ylabel('Signed '+label)
   ax=axes[0,1];data=arrays[case+f'_new_upper{alpha}__heading_contributions']
   for j,name in enumerate(['upper','A speed','B steer','C remainder']):ax.plot(t[mask],data[mask,j],label=name)
   ax.plot(t[mask],d['e_psi_unwrapped'][mask],'k--',label='raw ref - actual yaw');ax.set_ylabel('Cumulative yaw error contribution (rad)')
   for trace,label,ls in [(d,'new fixed300','-'),(b,'old R196',':'),(base,'zero-upper R196','--')]:
    m=trace['time']<horizon-1e-6;tm=trace['time'][m]
    axes[3,0].plot(tm,trace['e_psi_unwrapped'][m],ls,label=label);axes[3,1].plot(tm,trace['phi'][m],ls,label=label)
    parts=reward_audit(trace,alpha,upper,16,include_reconstruction=True);reward=sum(parts.values())
    axes[4,0].plot(np.arange(1,m.sum()+1),reward[m],ls,label=label);axes[4,1].plot(np.arange(1,m.sum()+1),np.cumsum(reward[m]),ls,label=label)
   axes[3,0].set_ylabel('Raw-reference heading error (rad)');axes[3,1].set_ylabel('roll (rad)');axes[3,1].axhline(.302,color='r',ls='--');axes[3,1].axhline(-.302,color='r',ls='--')
   axes[4,0].set_ylabel('Per-control-step reward, same alpha');axes[4,1].set_ylabel('Cumulative reward');axes[4,0].set_xlabel('Control steps');axes[4,1].set_xlabel('Control steps')
   for rr in [0,1,2,3]:
    for ax in axes[rr]:
     if rr!=0 or ax is axes[0,1]:
      ax.set_xlabel('Time (s)');ax.axvline(10,color='k',ls=':',alpha=.3) if horizon==16 else None
      for row in manifest['protocol']['cases'][case][1:]:ax.axvline(row[0],color='.5',alpha=.3)
     if ax is not axes[0,0]:ax.legend(fontsize=7);ax.grid(alpha=.25)
   for ax in axes[4]:ax.legend(fontsize=7);ax.grid(alpha=.25)
   fig.suptitle(f'{case} | alpha{alpha} upper best + fixed lower300 | {horizon}s | seed77001 | diagnostic only\nOriginal reference and scoring windows unchanged; all old traces reuse same complete preparation')
   fig.tight_layout(rect=(0,0,1,.965));fig.savefig(out/f'{case}_alpha{alpha}_{horizon}s.png',dpi=130);plt.close(fig)
atomic_json(out/'comparison_metrics.json',result);np.savez_compressed(out/'error_decomposition.npz',**arrays)
print(json.dumps({k:{h:{x:v for x,v in m.items() if x in ['complete','joint_final_hold','physical_failure','peak_roll','endpoint_xy_distance','endpoint_heading_error']} for h,m in row['windows'].items()} for k,row in result.items()},indent=2))
