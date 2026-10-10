#!/usr/bin/env python3
"""Plot original saved intervals + offline controller reconstruction; no simulation."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--parent-run',required=True);p.add_argument('--offline',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.parent_run)/'evaluation100/fast_turn';off=Path(a.offline);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
windows=[('B0',5.445,6.205),('B0',6.315,8.745),('alpha0',9.020,10.240)]
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,limit in zip(axes,[10.,12.]):
 for method,color in [('B0','black'),('alpha0','tab:blue'),('alpha1','tab:orange')]:
  d=dict(np.load(r/f'{method}.npz'));mask=d['time']<limit-1e-6;xy=d['actual_xy'][mask];ax.plot(*xy.T,label=method+(' V5.1@100' if method!='B0' else ' frozen R196'),color=color)
  for when,marker in [(1,'o'),(5,'s'),(limit,'x')]:
   i=np.argmin(abs(d['time']-min(when,limit-.005)));ax.scatter(*d['actual_xy'][i],marker=marker,color=color,s=22)
  if method=='B0':ax.plot(*d['reference_xy'][mask].T,'--',color='gray',label='Original command integral')
 ax.set(title=f'fast_turn / seed77001 / 0-{limit:g}s',xlabel='World X [m]',ylabel='World Y [m]');ax.axis('equal');ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.suptitle('Original saved trajectories; 10s main window and separate 12s diagnostic extension\nCommand: turn at1s, straighten at5s; no external pulse; true endpoints retained',fontsize=10);fig.tight_layout();fig.savefig(out/'xy_fast_turn.png',dpi=160);plt.close(fig)
summary=[]
for method,start,end in windows:
 d=dict(np.load(r/f'{method}.npz'));q=dict(np.load(off/f'{method}.npz'));n=len(q['ecbc_output']);t=d['time'][:n];mask=(t>=start-1e-5)&(t<end-1e-5);view=(t>=start-.15)&(t<end+.15);tt=t[view]
 row=dict(method=method,interval=[start,end],new_physics_steps=0)
 for name,values in {'ecbc_terms':q['ecbc_terms'],'eso_disturbance':q['disturbance'],'eso_shift':q['equilibrium_shift'],'reference_roll':q['reference_roll'],'delta_error':d['actual_delta'][:n]-d['governed'][:n,1],'ecbc':d['u_nom'][:n,0],'nn':d['applied_residual'][:n,0],'final_rate':d['final_command'][:n,0],'actual_rate':d['actual_delta_rate'][:n],'rate_error':d['actual_delta_rate'][:n]-d['final_command'][:n,0],'return_elapsed':q['return_elapsed'],'return_pending':q['return_pending'],'return_missed':q['return_deadline_missed'],'path':d['lower_path_features'][:n]}.items():
  x=values[mask].astype(float);row[name]=dict(mean=x.mean(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),rmse=np.sqrt(np.mean(x*x,axis=0)).tolist())
 toward=-np.sign(q['pre_measurement'][:,1]-d['governed'][:n,1]);row['fractions']={k:float(np.mean((values*toward)[mask]>0)) for k,values in {'ecbc_correcting':d['u_nom'][:n,0],'nn_correcting':d['applied_residual'][:n,0],'final_correcting':d['final_command'][:n,0],'actual_correcting':d['actual_delta_rate'][:n]}.items()};summary.append(row)
 fig,aa=plt.subplots(4,2,figsize=(13,12),sharex=True);axs=aa.ravel()
 def draw(i,x,label,style='-'):axs[i].plot(tt,np.asarray(x)[:n][view],style,label=label,linewidth=1.1)
 draw(0,d['governed'][:,1],'governed delta');draw(0,d['limited_command'][:,1],'original delta','--');draw(0,d['actual_delta'],'actual delta (post)');axs[0].set_ylabel('Steer [rad]')
 draw(1,d['final_command'][:,0],'final command (pre)');draw(1,d['actual_delta_rate'],'actual rate (post)','--');axs[1].set_ylabel('Steer rate [rad/s]')
 for i,label in enumerate(['wheel angle','roll error','ESO shift','roll rate']):draw(2,q['ecbc_terms'][:,i],label)
 draw(2,q['ecbc_output'],'ECBC sum','--');axs[2].set_ylabel('ECBC contributions [rad/s]')
 draw(3,d['u_nom'][:,0],'ECBC');draw(3,d['applied_residual'][:,0],'R196 scaled residual');draw(3,d['final_command'][:,0],'final','--');axs[3].set_ylabel('Steer rate [rad/s]')
 draw(4,d['phi'],'actual roll (post)');draw(4,q['reference_roll'],'ECBC roll ref (pre)');draw(4,q['equilibrium_shift'],'ESO shift (pre)','--');axs[4].set_ylabel('Roll / shift [rad]')
 draw(5,d['lower_path_features'][:,1],'path heading [rad]');draw(5,d['lower_path_features'][:,0],'path lateral [m]');axs[5].set_ylabel('Path fields [rad, m]')
 draw(6,q['return_elapsed'],'return elapsed [s]');draw(6,q['return_pending'],'pending [bool]');draw(6,q['return_deadline_missed'],'deadline missed [bool]','--');axs[6].set_ylabel('Pre-action return state')
 draw(7,q['disturbance'],'ESO disturbance [rad/s²]');axs[7].set_ylabel('ESO disturbance [rad/s²]')
 right=axs[7].twinx();right.plot(tt,q['projection_progress'][view],color='tab:orange',label='projection progress [m]');right.set_ylabel('Selected path progress [m]');right.legend(fontsize=7,loc='upper right')
 for ax in axs:ax.axvspan(start,end,color='gold',alpha=.1);ax.axvline(start,color='gray',linewidth=.6);ax.axvline(end,color='gray',linewidth=.6);ax.grid(alpha=.25);ax.legend(fontsize=7,loc='best')
 axs[6].set_xlabel('Interval start [s]; post state at t + 0.005s');axs[7].set_xlabel('Interval start [s]')
 fig.suptitle(f'{method} fast_turn [{start:.3f},{end:.3f}) / seed77001 / frozen R196'+('\nV5.1 upper alpha0@100' if method=='alpha0' else '\nZero upper')+'; original trace, offline ECBC/ESO/return reconstruction; no score shift',fontsize=11)
 fig.tight_layout();fig.savefig(out/f'chain_{method}_{start:.3f}_{end:.3f}.png',dpi=150);plt.close(fig)
(out/'interval_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'offline_checks.json').write_text((off/'checks.json').read_text())
