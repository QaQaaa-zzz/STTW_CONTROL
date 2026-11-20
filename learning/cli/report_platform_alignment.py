"""Score the locked-handback funnel for every root on a frozen physical layout."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from report_platform_controls import flight


def score(path, output, route_names):
    status = json.loads((path/'status.json').read_text())
    if status['stage'] != 'complete':
        raise ValueError('Scoring needs the terminal physical batch')
    cfg = json.loads((path/'resolved_config.json').read_text())
    terrain = cfg['ditch_terrain']
    front = terrain['front_x_m']
    model = mujoco.MjModel.from_binary_path(str(path/'terrain.mjb'))
    skills = list(dict.fromkeys(r['skill'] for r in status['results']))
    fig,axes = plt.subplots(len(skills),len(route_names),figsize=(3.7*len(route_names),2.5*len(skills)),
                            sharex=True,sharey=True,squeeze=False,layout='constrained')
    records = []
    for case in status['results']:
        route,skill = case['route'],case['skill']
        with np.load(path/case['case_dir']/'trajectory.npz') as data:
            trace,q,loads,contact = (data[k] for k in ('trace','qpos','contact_substep','wheel_contact_flags'))
            reference = data['reference_xy']
        time = trace[:,0]
        entry, handback = case['rb_entry_s'],case['handoff_s']
        supported = contact[:,2] & (loads[:,-1,:2] > 1e-5).all(1)
        takeoff, end, landing = flight(model,q,loads,supported,time,front,front,terrain['far_bank_height_m'],
                                       np.inf if entry is None else entry)
        qualified = landing is not None and handback is not None and landing <= handback+1e-5
        k = None if entry is None else np.argmin(abs(time-entry))
        h = None if handback is None else np.argmin(abs(time-handback))
        variant = case['variant']['bank_row']
        records.append(dict(skill=skill,root=variant,route=route,profile=route_names[route],
                            direction='left' if route%2==0 else 'right',case=case['case_dir'],entry=entry,
                            entry_y=q[k,1] if k is not None else None,
                            entry_yaw_deg=np.rad2deg(trace[k,6]) if k is not None else None,
                            entry_roll_deg=np.rad2deg(trace[k,4]) if k is not None else None,
                            entry_vx=trace[k,7] if k is not None else None,flight_s=takeoff,flight_end_s=end,
                            loaded_s=landing,homing=int((trace[:,-1]==4).any()),handoff_s=handback,
                            verified=int(qualified),handoff_y=q[h,1] if h is not None else None,
                            handoff_yaw_deg=np.rad2deg(trace[h,6]) if h is not None else None,
                            goal=int(qualified and case['goal']),termination=case['termination']))
        ax = axes[skills.index(skill),route]
        color = '#27634f' if qualified else '#ac5548'
        ax.plot(q[:,0],q[:,1],color=color,alpha=.6,lw=.8)
        ax.plot(q[-1,0],q[-1,1],'.',color=color,ms=3)
        if variant==status['results'][0]['variant']['bank_row']:
            ax.plot(*reference.T,'k--',lw=.8)
            ax.axvline(front,color='gray',lw=.7)
            for lo,hi,ylo,yhi in terrain.get('forbidden_rectangles',[]):
                ax.add_patch(plt.Rectangle((lo,ylo),hi-lo,yhi-ylo,color='black'))
            ax.set(xlim=(1,cfg['ground_handoff']['goal']['x_max_m']+1),ylim=(-1.3,1.3),
                   aspect='equal',title=f'{skill} / {route_names[route]}',xlabel='x (m)',ylabel='y (m)')
    output.mkdir(parents=True,exist_ok=True)
    fig.suptitle('XY to termination | green: verified handback | red: failed | dashed: reference')
    fig.savefig(output/'xy.png',dpi=135)
    plt.close(fig)
    with (output/'cases.csv').open('w') as stream:
        writer=csv.DictWriter(stream,records[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('route_names',nargs='+')
    args=parser.parse_args()
    score(args.run,args.output,args.route_names)
