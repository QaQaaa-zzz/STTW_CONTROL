"""Measure yaw evolution and applied steering between platform loading and physical relock."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from report_platform_controls import flight
from run_playground_task_reversal import write_rows


def report(baseline, changed, dest):
    dest.mkdir(exist_ok=True,parents=True)
    rows=[]
    fig,axes=plt.subplots(4,4,figsize=(14,9),layout='constrained',sharex='col')
    for j,(variant,out) in enumerate((('baseline',baseline),('signed_tangent',changed))):
        for i,skill in enumerate(('source','rb_run1')):
            cap=out/(skill+'_capture')
            model=mujoco.MjModel.from_binary_path(str(cap/'terrain.mjb'))
            schema=json.loads((cap/'model_state_schema.json').read_text())
            cases=json.loads((cap/'status.json').read_text())['results']
            for case in cases:
                lane=case['route'];root=case['variant']['bank_row']
                path=cap/case['case_dir']/'trajectory.npz'
                with np.load(path) as z:
                    tr,q,loads,flags=[z[k] for k in ('trace','qpos','contact_substep','wheel_contact_flags')]
                    u=z['applied_targets'][:,0]
                t=tr[:,0]
                take,end,loaded=flight(model,q,loads,flags[:,2]&(loads[:,-1,:2]>1e-5).all(1),t,
                                      10.3,10.3,.16,case['rb_entry_s'] if case['rb_entry_s'] is not None else np.inf)
                hand=case['handoff_s']
                ok=loaded is not None and hand is not None and loaded<=hand+1e-5
                row=dict(route=variant,skill=skill,root=root,approach=('left','right')[lane%2],
                    valid_flight=take is not None,loaded_s=loaded,handoff_s=hand,verified=int(ok),
                    termination=case['termination'],yaw_delta_deg=None,roll_at_load_deg=None,roll_at_lock_deg=None,
                    applied_steer_mean_deg=None,measured_steer_mean_deg=None,
                    load_to_lock_s=None,world_vy_at_load=None,world_vy_at_lock=None,reference=str(path))
                if loaded is not None:
                    lo=round(loaded/.005);last=round(hand/.005) if ok else len(t)-1
                    window=slice(lo,last+1)
                    tt=t[window]-t[lo]
                    yaw=np.rad2deg(np.unwrap(tr[window,6]))
                    col=j*2+i
                    color=('#3a749b','#c5682a')[lane%2]
                    measured=np.rad2deg(q[window,schema['steer_qpos']])
                    for ax,v in zip(axes[:,col],(yaw,np.rad2deg(tr[window,4]),measured,np.rad2deg(u[window])),strict=True):
                        ax.plot(tt,v,color=color,lw=.8,alpha=.65)
                        ax.plot(tt[-1],v[-1],'.' if ok else 'x',color=color,ms=3)
                    if ok:
                        row.update(yaw_delta_deg=yaw[-1]-yaw[0],roll_at_load_deg=np.rad2deg(tr[lo,4]),
                            roll_at_lock_deg=np.rad2deg(tr[last,4]),applied_steer_mean_deg=np.rad2deg(u[lo:last].mean()) if last>lo else None,
                            measured_steer_mean_deg=measured[:-1].mean() if last>lo else None,
                            load_to_lock_s=t[last]-t[lo],world_vy_at_load=tr[lo,8],world_vy_at_lock=tr[last,8])
                rows.append(row)
            axes[0,j*2+i].set_title(variant+' / '+skill)
    for ax,label in zip(axes[:,0],('world yaw (deg)','roll (deg)','actual steer (deg)','applied steer target (deg)'),strict=True):
        ax.set_ylabel(label)
    for ax in axes[-1]:ax.set_xlabel('time since both wheels loaded (s)')
    fig.suptitle('Solid platform loading to lock (dot) or termination (cross) | blue: left approach, orange: right')
    fig.savefig(dest/'landing_controls.png',dpi=135)
    plt.close(fig)
    write_rows(dest/'landing_controls.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','changed','dest'):p.add_argument(name,type=Path)
    a=p.parse_args()
    report(a.baseline,a.changed,a.dest)
