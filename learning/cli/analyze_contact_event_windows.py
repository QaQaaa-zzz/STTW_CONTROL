"""Offline event-aligned contact diagnostics from saved solved-step wrenches.

Last-contact boundaries use hindsight and are NOT an online clearance detector.
"""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
a=p.parse_args()
archive=np.load(a.run/'physics_wrench.npz')
d={key:archive[key] for key in archive.files}
c=json.loads((a.run/'contract.json').read_text())
idx=json.loads((a.run/'physics_contacts/index.json').read_text())
t=d['time_s'];dt=idx['physics_dt_s'];steer=idx['joint_order'].index('steering_joint')
limit=max(abs(v) for v in idx['motor_limits'][0])

def periods(mask):
    changes=np.diff(np.r_[False,mask,False].astype(int))
    start=np.flatnonzero(changes==1);end=np.flatnonzero(changes==-1)
    return dict(total_s=float(mask.sum()*dt),longest_s=float((end-start).max()*dt) if len(start) else 0.,count=int(len(start)))

rows=[]
for lane,alpha in enumerate(c['settings']['alphas']):
    valid=d['alive'][:,lane];front=d['ridge_contact'][:,lane,0]&valid;rear=d['ridge_contact'][:,lane,1]&valid
    fi=np.flatnonzero(front);ri=np.flatnonzero(rear)
    first=float(t[fi[0]]) if len(fi) else None
    last=float(t[ri[-1]]) if len(ri) else None
    intervals={'full_observed':(0,float(t[-1])+dt)}
    if first is not None:
        intervals['pre_front_150ms']=(max(0,first-.15),first)
    if first is not None and last is not None:
        intervals['two_wheel_ridge_envelope']=(first,last+dt)
        intervals['post_rear_500ms']=(last+dt,min(float(t[-1])+dt,last+.5+dt))
    load=d['normal_force'][:,lane,0,2]
    baseline_mask=(t>=max(0,first-.15))&(t<first)&valid if first is not None else (t<=.15)&valid
    reference=float(load[baseline_mask].mean()) if baseline_mask.any() else None
    row=dict(lane=lane,alpha=alpha,first_front_contact_s=first,last_rear_contact_s=last,operational_precontact_load_N=reference,windows={})
    for name,(lo,hi) in intervals.items():
        m=(t>=lo)&(t<hi)&valid
        if not m.any():continue
        moment=np.column_stack((d['moment'][:,lane,0],d['friction_components'][:,lane,:2]))[m]
        total=moment.sum(1)
        load_window=load[m]
        saturated=m&(abs(d['motor'][:,lane,steer])>=.999*limit)
        low=m&(load<.2*reference) if reference is not None else np.zeros_like(m)
        # Near-zero normal load is reported separately from the20% criterion.
        unloaded=m&(np.linalg.norm(d['normal_force'][:,lane,0],axis=1)<.1)
        row['windows'][name]=dict(start_s=lo,end_s=hi,observed_s=float(m.sum()*dt),signed_impulse_Nms=(moment.sum(0)*dt).tolist(),absolute_impulse_Nms=(abs(moment).sum(0)*dt).tolist(),net_contact_signed_impulse_Nms=float(total.sum()*dt),net_contact_absolute_impulse_Nms=float(abs(total).sum()*dt),motor_signed_impulse_Nms=float(d['motor'][m,lane,steer].sum()*dt),steering_saturation=periods(saturated),front_low_load=periods(low),front_near_zero_load=periods(unloaded),front_peak_vertical_load_N=float(load_window.max()),steering_rate_start_rad_s=float(d['steer_velocity'][m,lane][0]),steering_rate_end_rad_s=float(d['steer_velocity'][m,lane][-1]),steering_rate_peak_abs_rad_s=float(abs(d['steer_velocity'][m,lane]).max()))
    rows.append(row)
report=dict(role='offline event-aligned DEV; last rear ridge contact is retrospective, not an online confirmed clearance event',components=['normal','longitudinal_friction','lateral_friction'],load_reference='mean150ms immediately before first front ridge contact; flat uses first150ms; not theoretical static load',ridge_definition='loaded normal patch above world z=3mm, force norm>0.01N',unloading_definition='front normal resultant norm<0.1N',lanes=rows)
(a.run/'contact_event_windows.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(lanes=len(rows),with_two_wheel_events=sum('two_wheel_ridge_envelope' in r['windows'] for r in rows))))

# One panel per lane keeps paired event markers and true observation endpoints.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for begin in range(0,len(rows),5):
    page=rows[begin:begin+5]
    fig,axes=plt.subplots(len(page),2,figsize=(11,2.4*len(page)),squeeze=False,layout='constrained')
    for panel,row in enumerate(page):
        lane=row['lane']
        m=d['alive'][:,lane];time=t[m];left,right=axes[panel]
        left.plot(time,d['moment'][m,lane,0],label='Normal',lw=.7)
        left.plot(time,d['friction_components'][m,lane,1],label='Lateral friction',lw=.7)
        right.plot(time,d['normal_force'][m,lane,0,2],lw=.7,label='Front vertical normal load')
        reference=row['operational_precontact_load_N']
        if reference is not None:right.axhline(.2*reference,ls='--',color='gray',label='20% precontact load')
        for ax in (left,right):
            if row['first_front_contact_s'] is not None and row['last_rear_contact_s'] is not None:ax.axvspan(row['first_front_contact_s'],row['last_rear_contact_s'],color='orange',alpha=.12)
            ax.grid(alpha=.2);ax.set_xlabel('Time (s)')
        left.set_ylabel('Steering-axis torque (Nm)');right.set_ylabel('Load (N)')
        left.set_title(f"lane {lane}, alpha={row['alpha']:g}")
    axes[0,0].legend();axes[0,1].legend()
    fig.suptitle('Solved-step contacts | orange: first front to last rear ridge contact\nObserved windows only; last-contact boundaries are retrospective')
    fig.savefig(a.run/('contact_event_windows.png' if len(rows)<=5 else f'contact_event_windows_{begin//5:02d}.png'),dpi=140);plt.close(fig)
