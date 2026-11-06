"""Planar rigid-circle trail threshold from the nominal vehicle geometry."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
cfg=json.loads(a.controller.read_text());c=cfg['trail'];lam=cfg['caster'];radius=.1;length=.8
critical=lam-np.arcsin(np.sin(lam)-c*np.cos(lam)/radius)
beta=np.deg2rad(np.linspace(-15,25,401));trail=(c*np.cos(lam)-radius*np.sin(lam)+radius*np.sin(lam-beta))/np.cos(lam-beta)
# Independent line/plane intersection in x-z checks the algebra.
axis=np.array([-np.sin(lam),np.cos(lam)]);origin=np.array([c,0]);normal=np.stack([-np.sin(beta),np.cos(beta)],axis=1);forward=np.stack([np.cos(beta),np.sin(beta)],axis=1);point=np.array([0,radius])-radius*normal
q=origin+np.sum(normal*(point-origin),axis=1)[:,None]/(normal@axis)[:,None]*axis;geometric=np.sum((q-point)*forward,axis=1)
rows=[]
for h in [.02,.04,.06]:
 b=np.arctan(np.pi*h/length);t=(c*np.cos(lam)-radius*np.sin(lam)+radius*np.sin(lam-b))/np.cos(lam-b);rows.append(dict(height_m=h,nominal_peak_grade=float(np.tan(b)),nominal_peak_slope_deg=float(np.rad2deg(b)),trail_at_nominal_peak_m=float(t)))
result=dict(radius_m=radius,nominal_trail_m=c,caster_rad=lam,bump_length_m=length,critical_uphill_slope_deg=float(np.rad2deg(critical)),critical_smooth_bump_height_m=float(length*np.tan(critical)/np.pi),intersection_formula_max_difference_m=float(abs(geometric-trail).max()),cases=rows,assumptions='upright fixed-pitch frame, rigid circular wheel, one tangent plane, zero steer/roll; smooth centerline sin² bump; not the discretized oblique-contact prediction')
(a.output/'nominal_threshold.json').write_text(json.dumps(result,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,5),layout='constrained');ax.plot(np.rad2deg(beta),trail*1000,label='Nominal planar geometry');ax.axhline(0,color='k',lw=.8);ax.axvline(np.rad2deg(critical),color='grey',ls='--',label=f'Zero trail: {np.rad2deg(critical):.2f}°')
for r in rows:ax.scatter(r['nominal_peak_slope_deg'],r['trail_at_nominal_peak_m']*1000,label=f"{r['height_m']*100:g}cm smooth bump")
ax.set(xlabel='Uphill contact-plane angle (deg)',ylabel='Trail along contact plane (mm)',title='Nominal geometry prediction: R=0.1m, c=24mm, caster=25°\nFixed upright frame; L_bump=0.8m; verify against actual contact data');ax.grid(alpha=.2);ax.legend();fig.savefig(a.output/'nominal_threshold.png',dpi=150);plt.close(fig);print(json.dumps(result,indent=2))
