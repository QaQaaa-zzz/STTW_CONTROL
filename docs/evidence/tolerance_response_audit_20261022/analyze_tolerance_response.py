"""Offline trajectory-sensitivity audit; no simulator or model updates."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
pilot=a.root/'runs/pulse_pilot_v1_attempt2';boundary=a.root/'runs/boundary_focus_v1'
rows=json.loads((pilot/'results.json').read_text())['cases'];extra=json.loads((boundary/'summary.json').read_text())['cases']
checks=[];bounds=[];responses=[]
def signals(d):
 hold=d['time']>=19.
 return [(d['phi'],.302,'roll'),((d['actual_forward_speed']-d['v_user'])[hold],.1,'speed_hold'),(d['path_cross_track'][hold],.1,'cross_track_hold'),(d['path_heading_error'][hold],.05,'heading_hold')]
for alpha in [0,1]:
 for onset in [1,2,4]:
  prefix=f'alpha{alpha}_t{onset}_F';ds={f:np.load(pilot/(prefix+f'{f:+g}.npz')) for f in [-1,0,1]}
  nominal=signals(ds[0]);plus=signals(ds[1]);minus=signals(ds[-1]);sens=[(v[0]-w[0])/2 for v,w in zip(plus,minus)]
  for sign in [-1,1]:
   limits=[]
   for (x,limit,name),s in zip(nominal,sens):
    q=sign*s
    for side in [-1,1]:
     den=side*q;mask=den>0
     if mask.any():limits.append((float(np.min((limit-side*x[mask])/den[mask])),name))
   bound,active=min(limits);bounds.append(dict(alpha=alpha,onset=onset,sign=sign,bound_N=bound,active_constraint=active))
  selected=[(pilot,r) for r in rows if r['alpha']==alpha and r['onset_s']==onset and abs(r['force_N']) in [3,6]]
  if alpha==0 and onset==1:selected += [(boundary,r) for r in extra if abs(r['force_N']) in [4,4.5,5,5.5]]
  for root,r in selected:
   predicted=all(np.max(np.abs(x+r['force_N']*s))<=limit for (x,limit,_),s in zip(nominal,sens))
   checks.append(dict(name=r['name'],force_N=r['force_N'],predicted_pass=bool(predicted),actual_pass=r['success'],source=root.name))
  for t in [onset+.1,onset+.5,onset+.78]:
   i=int(np.argmin(abs(ds[0]['time']-t)));responses.append(dict(alpha=alpha,onset=onset,time=float(ds[0]['time'][i]),roll_sensitivity_rad_per_N=float((ds[1]['phi'][i]-ds[-1]['phi'][i])/2)))
  if alpha==0 and onset==1:
   fig,ax=plt.subplots(1,2,figsize=(12,4));t=ds[0]['time'];mask=(t>=.9)&(t<=2.5)
   for f in [-1,0,1]:ax[0].plot(t[mask],ds[f]['phi'][mask],label=f'{f:+g} N')
   ax[0].set(xlabel='time [s]',ylabel='roll [rad]',title='Saved closed-loop responses');ax[0].legend()
   ax[1].plot(t[mask],sens[0][mask]);ax[1].axhline(0,color='black',lw=.7);ax[1].set(xlabel='time [s]',ylabel='d roll / d force [rad/N]',title='Central finite difference from +/-1 N')
   for axis in ax:axis.axvspan(1,1.1,alpha=.15,color='orange');axis.grid(alpha=.2)
   fig.tight_layout();fig.savefig(a.out/'response_sign.png',dpi=160);plt.close(fig)
summary=dict(bounds=bounds,validation=checks,responses=responses,correct=sum(r['predicted_pass']==r['actual_pass'] for r in checks),total=len(checks),unsafe_predictions=sum(r['predicted_pass'] and not r['actual_pass'] for r in checks),conservative_predictions=sum(not r['predicted_pass'] and r['actual_pass'] for r in checks),scope='Post-hoc DEV diagnostic, same six snapshots; not independent validation or certified tolerance. Linear roll prediction does not certify substep peaks, physical failures, domain, goal, or faults.')
(a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
