import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src=Path('/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/runs/pulse_pilot_v1_attempt2');out=Path('/tmp/sttw_nextstep_20261022');rows=json.loads((src/'results.json').read_text())['cases'];path=np.load(src/'path.npz')
def load(alpha,t,f):return dict(np.load(src/f'alpha{alpha}_t{t}_F{f:+g}.npz'))
def constraints(d):
 tail=d['time']+.005>19.+1e-5
 return np.concatenate([d['peak_roll']-.302,abs(d['actual_forward_speed'][tail]-2.6)-.1,abs(d['path_cross_track'][tail])-.1,abs(d['path_heading_error'][tail])-.05])
probes=[];checks=[];fig,axs=plt.subplots(2,3,figsize=(15,9));colors=plt.cm.coolwarm(np.linspace(0,1,7))
for alpha in [0,1]:
 for col,t in enumerate([1,2,4]):
  zero=load(alpha,t,0);g0=constraints(zero);on=int(t/.005)-1
  for sign in [-1,1]:
   unit=load(alpha,t,sign);q=constraints(unit)-g0;mask=q>0;limits=-g0[mask]/q[mask];bound=float(limits.min()) if mask.any() else None
   predicted=[]
   for f in [3,6]:
    actual=load(alpha,t,sign*f);observed=next(r for r in rows if r['alpha']==alpha and r['onset_s']==t and r['force_N']==sign*f)
    gp=g0+f*q;ga=constraints(actual)
    predicted.append(dict(force_N=sign*f,predicted_violation=bool(gp.max()>0),observed_violation=not observed['success'],max_constraint_prediction_error=float(abs(ga-gp).max()),peak_roll_predicted=float(np.max(zero['peak_roll']+f*(unit['peak_roll']-zero['peak_roll']))),peak_roll_observed=observed['peak_roll']))
   probes.append(dict(alpha=alpha,onset=t,sign=sign,finite_difference_margin_N=bound,within_6N=bound is not None and bound<=6,validation=predicted))
  ax=axs[alpha,col]
  for color,f in zip(colors,[-6,-3,-1,0,1,3,6]):
   d=load(alpha,t,f);sel=d['time']<6.;ax.plot(d['actual_xy'][sel,0],d['actual_xy'][sel,1],color=color,lw=1.2,label=f'{f:+g}N')
   impulse=d['applied_force_impulse'].sum(axis=0);checks.append(dict(alpha=alpha,onset=t,force=f,ticks=len(d['time']),impulse_norm=float(np.linalg.norm(impulse)),expected_norm=abs(f)*.1,peak_roll=float(d['peak_roll'].max())))
  sel=path['s']<15;ax.plot(path['xy'][sel,0],path['xy'][sel,1],'k--',lw=1,label='original path');ax.set_xlim(7,16);ax.set_ylim(-1,10);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_title(f'alpha={alpha}, pulse={t}s');ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.legend(fontsize=6,ncol=2)
fig.tight_layout();fig.savefig(out/'xy_turn_zoom.png',dpi=160);plt.close(fig)
failure=next(r for r in rows if not r['success']);d=load(0,1,-6);bad=np.flatnonzero(d['peak_roll']>.302)
failure.update(first_working_violation_s=float(d['time'][bad[0]]+.005),violation_samples=len(bad),violation_sample_duration_s=len(bad)*.005,peak_time_s=float(d['time'][np.argmax(d['peak_roll'])]+.005),peak_degrees=float(np.rad2deg(d['peak_roll'].max())))
fig,ax=plt.subplots(figsize=(9,4))
for f in [0,-1,-3,-6,6]:
 d=load(0,1,f);ax.plot(d['time']+.005,np.rad2deg(d['peak_roll']),label=f'{f:+g} N')
ax.axhline(np.rad2deg(.302),color='k',ls='--',label='working limit');ax.axvspan(1,1.1,alpha=.15,color='gray');ax.set_xlim(.5,5);ax.set_ylabel('absolute roll peak [deg]');ax.set_xlabel('episode time [s]');ax.legend();fig.tight_layout();fig.savefig(out/'failure_roll.png',dpi=160)
validation=[x for row in probes for x in row['validation']];cm={'TP':0,'FP':0,'FN':0,'TN':0}
for r in validation:cm[('T' if r['predicted_violation']==r['observed_violation'] else 'F')+('P' if r['predicted_violation'] else 'N')]+=1
report=dict(failure=failure,force_integrity=checks,closed_loop_finite_difference_probes=probes,amplitude_holdout_confusion=cm,scope='same six DEV states; +/-1N secants predict +/-3,6N. Not infinitesimal derivatives, not independent scene validation or calibrated trust radii.')
(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({'failure':failure,'confusion':cm,'proxies':[(r['alpha'],r['onset'],r['sign'],r['finite_difference_margin_N']) for r in probes]},indent=2))
