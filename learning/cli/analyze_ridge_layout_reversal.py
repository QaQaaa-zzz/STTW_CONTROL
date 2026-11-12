"""Compare saved ridge placements: all cases, paired clocks and contact switching."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_roll_contacts import roll_state

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
colors={0:'#00796b',4:'#d55e00'};data={};rows=[];checks=[]
for layout in (0,1):
 for seed in (401,402,403,404):
  folder=a.root/f'ridge100_v2.6_layout{layout}_seed{seed}';c=json.loads((folder/'contract.json').read_text());z=np.load(folder/'trajectory.npz');tr=z['lower_trace'].reshape(-1,12,11)
  metrics=json.loads((folder/'condition_metrics.json').read_text());nfront=c['bodies'].index('frontwheel');nrear=c['bodies'].index('rearwheel')
  for lane in range(6):
   n=np.flatnonzero(tr[:,lane,6])[-1]+1;t=np.arange(n)*c['lower_dt']+tr[:n,lane,7]*c['physics_dt'];body=z['body_state'][:n,lane,c['bodies'].index('chasis')]
   phi,omega=roll_state(body);m=z['measurement'][:n,lane];front=np.linalg.norm(z['contact_force'][:n,lane,nfront],axis=-1);rear=np.linalg.norm(z['contact_force'][:n,lane,nrear],axis=-1)
   off=np.flatnonzero((t>1)&(t<3.2)&(front<.1));segments=np.split(off,np.flatnonzero(np.diff(off)>1)+1);long=[s for s in segments if len(s)*.005>=.1];first=long[0]
   j=first[0];reg=(t>=1)&(t<=2.6);ip=np.flatnonzero(reg)[np.argmax(abs(omega[reg]))]
   reasons=[key for key in ('roll_limit_failure','contact_failure','domain_exit','numerical_failure') if z[key][:n,lane].any()]
   row=dict(layout=layout,seed=seed,lane=lane,label=c['settings']['case_labels'][lane],alpha=c['settings']['alphas'][lane],full_window=not metrics[lane]['failed'],final_hold=metrics[lane]['final_hold'],end_s=float(t[-1]),failure='+'.join(reasons),first_front_unload_s=float(t[j]),first_front_unload_duration_s=float(len(first)*.005),onset_speed_m_s=float(tr[j,lane,2]),onset_roll_deg=float(np.degrees(phi[j])),onset_euler_roll_rate_deg_s=float(np.degrees(omega[j])),onset_steer_deg=float(np.degrees(m[j,2])),peak_euler_roll_rate_1_to_2p6_deg_s=float(np.degrees(omega[ip])),peak_rate_time_s=float(t[ip]),last100ms_front_mean_N=float(front[-20:].mean()),last100ms_front_unloaded_fraction=float((front[-20:]<.1).mean()),actual_rate_above_2p9_fraction_1_to_2p6=float((abs(m[reg,3])>=2.9).mean()),command_at_3_fraction_1_to_2p6=float((abs(z['final_command'][:n,lane,0][reg])>=2.999).mean()))
   rows.append(row);data[layout,seed,lane]=dict(t=t,xy=tr[:n,lane,:2],speed=tr[:n,lane,2],phi=np.degrees(phi),omega=np.degrees(omega),front=front,rear=rear,steer_rate=m[:,3],command=z['final_command'][:n,lane,0],delta=m[:,2])
  # Compare forces on the same 100 ms immediately preceding the failing endpoint.
  fail_lane=4 if layout==0 else 0;end=data[layout,seed,fail_lane]['t'][-1]
  for lane in (0,4):
   d=data[layout,seed,lane];mask=(d['t']>end-.1+1e-8)&(d['t']<=end+1e-8)
   checks.append(dict(layout=layout,seed=seed,alpha=lane/4,window_end_s=float(end),mean_front_N=float(d['front'][mask].mean()),front_unloaded_fraction=float((d['front'][mask]<.1).mean()),mean_rear_N=float(d['rear'][mask].mean())))
with (a.output/'case_metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with (a.output/'matched_failure_windows.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
summary=[]
for layout in (0,1):
 for lane in range(6):
  group=[r for r in rows if r['layout']==layout and r['lane']==lane]
  summary.append(dict(layout=layout,label=group[0]['label'],n=4,full_window=sum(r['full_window'] for r in group),final_hold=sum(r['final_hold'] for r in group),failure_counts={reason:sum(reason in r['failure'] for r in group) for reason in ('roll_limit_failure','contact_failure','domain_exit')}))
(a.output/'summary.json').write_text(json.dumps(dict(source=str(a.root),summary=summary,definition='Full window and final hold reported separately; four matched physics groups, one rollout each. Contact norm threshold 0.1 N; actual rate diagnostic >=2.9 rad/s; command clipping diagnostic >=2.999 rad/s; true Euler roll derivative computed from world body quaternion and angular velocity.',paired_windows=checks),indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
route=np.load('runs/h75_selected_local300_flat/bundle/route.npz');fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for layout,ax in enumerate(axs):
 terrain=np.load(a.root/f'terrains/ridge100_layout{layout}/terrain.npz');v=terrain['vertices'];pts=v[v[:,2]>.001];ax.scatter(pts[:,0],pts[:,1],c='#cbb98e',s=12,alpha=.6,rasterized=True,label='10 cm ridge footprint')
 keep=(route['s']>=2)&(route['s']<=7);ax.plot(*route['xy'][keep].T,'k--',lw=1.4,label='Original route')
 for lane in (0,4):
  for seed in (401,402,403,404):
   d=data[layout,seed,lane];mask=d['t']<=3.2;xy=d['xy'][mask];ax.plot(*xy.T,color=colors[lane],lw=2 if seed==401 else 1,alpha=1 if seed==401 else .45,ls='-' if lane==0 else '--',label=f'alpha={lane/4:g}, four groups' if seed==401 else None)
   row=next(r for r in rows if (r['layout'],r['seed'],r['lane'])==(layout,seed,lane))
   if not row['full_window']:ax.scatter(*xy[-1],marker='x',color=colors[lane],s=35,zorder=5)
 ax.set(title=f'Layout {layout}: ridge centre s={3.9+.6*layout:.1f} m',xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Same route and ridge dimensions, different placement | trajectories shown to 3.2 s or actual failure')
fig.savefig(a.output/'xy_layout_reversal.png');plt.close(fig)
fig,axs=plt.subplots(5,2,figsize=(13,13),sharex=True,sharey='row',layout='constrained')
fields=[('speed','Actual speed (m/s)'),('phi','Signed roll (deg)'),('omega','Euler roll rate (deg/s)'),('front','Front contact norm (N)'),('command','Steer command (rad/s)')]
for layout in (0,1):
 for row,(key,ylabel) in enumerate(fields):
  ax=axs[row,layout]
  for lane in (0,4):
   for seed in (401,402,403,404):
    d=data[layout,seed,lane];mask=(d['t']>=1)&(d['t']<=2.7)
    ax.plot(d['t'][mask],d[key][mask],color=colors[lane],lw=1.8 if seed==401 else .8,alpha=1 if seed==401 else .35,ls='-' if lane==0 else '--',label=f'alpha={lane/4:g}' if seed==401 else None)
    if d['t'][-1]<2.7:ax.scatter(d['t'][-1],d[key][-1],marker='x',s=25,color=colors[lane],zorder=6)
  if key=='command':
   for bound in (-3,3):ax.axhline(bound,color='gray',ls=':',lw=1)
  if key in ('phi','omega'):ax.axhline(0,color='gray',lw=.6)
  ax.grid(alpha=.2);ax.set_xlim(1,2.7)
  if layout==0:ax.set_ylabel(ylabel)
  if row==0:ax.set_title(f'Layout {layout} | alpha0 / alpha1 full-window: '+('4/4 vs 0/4' if layout==0 else '0/4 vs 4/4'));ax.legend(loc='upper right')
  if row==4:ax.set_xlabel('Original recording time (s)')
fig.suptitle('Reversal follows contact switching and steering recovery\nAll four matched physics groups; bold = group401; x = actual endpoint',fontsize=14)
fig.savefig(a.output/'contact_phase_reversal.png');plt.close(fig)
print(json.dumps(summary,indent=2))
