"""Fit nominal continuation envelope on declared CALIB lanes, then evaluate HOLDOUT."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
d=json.loads((a.run/'declaration.json').read_text()); z=np.load(a.run/'evaluation/trajectories.npz'); summary=json.loads((a.run/'evaluation/summary.json').read_text())
cfg=json.loads((a.run/'config.json').read_text())
mount=np.array(json.loads(Path(cfg['asset_spec']).read_text())['mount_rotation_wxyz'])
r=z['root_xyzw'];q=r[...,3:7]
# Match TorchPendulumObservation.privileged: vehicle root frame has a90deg mount.
xyz=q[...,3:]*mount[1:]+mount[0]*q[...,:3]+np.cross(q[...,:3],mount[1:])
qw=q[...,3]*mount[0]-(q[...,:3]*mount[1:]).sum(-1)
x,y,zz=np.moveaxis(xyz,-1,0);w=qw
roll=np.arctan2(2*(w*x+y*zz),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-zz*x),-1,1));yaw=np.arctan2(2*(w*zz+x*y),1-2*(y*y+zz*zz))
f=np.stack([roll,pitch,yaw,r[...,1],r[...,7]-1.5],-1)
t=z['time'];window=(t>=d['window_s'][0]-1e-7)&(t<=d['window_s'][1]+1e-7)
cal=d['calibration_lanes'];hold=d['holdout_lanes'];floor=np.array([np.deg2rad(1)]*3+[.02,.1])
bound=np.maximum(floor,1.2*np.abs(f[window][:,cal]).max(axis=(0,1)))
inside=(np.abs(f)<=bound).all(-1)&z['valid']&~z['road_violation_history']
rows=[]
for i in range(r.shape[1]):
    full=bool(z['valid'][window,i].all())
    rows.append(dict(lane=i,role='CALIB' if i in cal else 'HOLDOUT',observed_full_window=full,window_inside_envelope=bool(inside[window,i].all()),window_forward_progress_m=float(r[np.flatnonzero(window)[-1],i,0]-r[np.flatnonzero(window)[0],i,0]),max_normalized_feature=(np.abs(f[window,i])/bound).max(0).tolist(),end_code=summary['cases'][i]['end_code'],road_violation=summary['cases'][i]['road_violation']))
contact=z['physics_contact_mode']; contact_seen=(contact!=0).any(axis=0)
accepted=all(rows[i]['observed_full_window'] and not rows[i]['road_violation'] for i in cal) and bool(np.all(bound[:2]<np.deg2rad([35,75]))) and bound[3]<=.6 and bool(contact_seen.all())
result=dict(adopted=accepted,feature_names=['roll_rad','pitch_rad','yaw_rad','world_y_m','vx_error_m_s'],absolute_bounds=bound.tolist(),cases=rows,calib_inside=sum(rows[i]['window_inside_envelope'] for i in cal),holdout_inside=sum(rows[i]['window_inside_envelope'] for i in hold),contact_seen_lanes=int(contact_seen.sum()),elapsed_s=summary['elapsed_s'],physics_lane_steps=summary['physics_lane_steps'],scope=d['scope'],limitations='Nominal empirical reference envelope only; no recovery-domain or safety claim. Contact presence checked; raw peak buffer occupancy not exported. Holdout used for coverage report, not threshold fitting.')
(a.run/'envelope.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axs=plt.subplots(2,2,figsize=(12,8));ax=axs[0,0];ax.plot([-5,20],[0,0],'k--',label='fixed reference')
for i,row in enumerate(rows):
    valid=z['valid'][:,i];c='tab:blue' if i in cal else 'tab:orange'; label=row['role'] if i in (0,12) else None
    ax.plot(r[valid,i,0],r[valid,i,1],color=c,alpha=.6,label=label); ax.scatter(r[valid,i,0][-1],r[valid,i,1][-1],color=c,s=10)
    axs[0,1].plot(t[valid],r[valid,i,7],color=c,alpha=.5)
    axs[1,0].plot(t[valid],z['reward'][valid,i],color=c,alpha=.5)
    axs[1,1].plot(t[valid],np.cumsum(z['reward'][valid,i]),color=c,alpha=.5)
ax.axis('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title='All24 actual XY and original reference');ax.legend()
axs[0,1].set(ylabel='world vx (m/s)',title='Nominal speed1.5 m/s');axs[0,1].axhline(1.5,color='k',ls='--')
axs[1,0].set(ylabel='step reward',title='Existing goal reward: timeout penalty retained');axs[1,1].set(ylabel='cumulative reward',title='Goal outside calibration window; not task completion')
for ax in axs.flat:ax.grid(alpha=.2)
for ax in (axs[0,1],axs[1,0],axs[1,1]):ax.set_xlabel('time (s)');ax.axvspan(*d['window_s'],alpha=.08,color='green')
fig.suptitle('Frozen H75 + ECBC/ESO; zero lower residual; alpha0.5; no jump/no disturbance');fig.tight_layout();fig.savefig(a.run/'xy_all.png',dpi=160);plt.close(fig)
lines=['# Nominal continuation calibration','', '![All actual/reference XY and diagnostics](xy_all.png)','',f"24 ground starts;12CALIB/12HOLDOUT, {result['elapsed_s']:.2f}s wall time, {result['physics_lane_steps']} physical lane steps,0training.",'',f"Envelope adopted: **{accepted}**. Full-window envelope coverage: CALIB {result['calib_inside']}/12; HOLDOUT {result['holdout_inside']}/12. Ground-contact recorded in {result['contact_seen_lanes']}/24 lanes.",'','|Feature|Absolute bound|','|---|---|']
for k,v in zip(result['feature_names'],bound):lines.append(f'|{k}|{v:.6g}|')
lines+=['','Window3.5–5.5s; limits=max(predeclared floor,1.2×CALIB-window maximum). Holdout does not change limits. Straight world-y/heading metrics cannot be used as curved-road errors. These limits describe ordinary dynamic motion, not static support or a certified recoverable set.','',result['limitations'],'','All original episode terminal codes and failed conditions remain in envelope.json. Horizon goal failure is distinct from a fully observed continuation window. No historical task success labels are rewritten.','', '|Lane|Role|Full window|Inside|Progress(m)|End code|','|---|---|---|---|---|---|']
for row in rows:lines.append(f"|{row['lane']}|{row['role']}|{row['observed_full_window']}|{row['window_inside_envelope']}|{row['window_forward_progress_m']:.3f}|{row['end_code']}|")
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
