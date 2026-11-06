"""Plot all paired candidates against fixed original task routes from saved traces."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--tasks',nargs='+');p.add_argument('--source-only',action='store_true');a=p.parse_args()
protocol=json.loads((a.run/'protocol.json').read_text());spec=protocol['spec'];physics_dt=protocol['settings']['plant_options'].get('physics_dt',.0002);tasks=spec['tasks'];candidates=spec['candidates']
if a.source_only:
    data=np.load(a.run/'source_trajectory.npz');trace=data['trace'];trace=trace.reshape(-1,trace.shape[-2],trace.shape[-1])[:,0]
    valid=trace[:,6]>0;tr=trace[valid];t=np.flatnonzero(valid)*.005+tr[:,7]*physics_dt
    route=np.load(spec['source_route']);fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    axes[0].plot(route['xy'][:,0],route['xy'][:,1],':',color='gray',label='Original source route')
    axes[0].plot(tr[:,0],tr[:,1],label='Actual lane 0');axes[0].plot(*tr[-1,:2],'x',color='black')
    lo=tr[:,:2].min(0)-.2;hi=tr[:,:2].max(0)+.2
    axes[0].set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World x (m)',ylabel='World y (m)');axes[0].set_aspect('equal',adjustable='box')
    axes[1].plot(t,tr[:,2],label='True forward speed');axes[1].plot(t,tr[:,10],label='Wheel-axis speed estimate')
    axes[1].axhline(spec['source_speed_m_s'],ls=':',color='black',label='Requested speed');axes[1].set(xlabel='Time (s)',ylabel='Speed (m/s)')
    axes[2].plot(t,tr[:,3],label='Signed cross-track');axes[2].set(xlabel='Time (s)',ylabel='Cross-track (m)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    status=json.loads((a.run/'status.json').read_text());fig.suptitle(f"{spec['origin_id']} | {status['status']} | one upstream root, lane 0")
    fig.savefig(a.run/'xy_source.png',dpi=150)
    (a.run/'source_summary.json').write_text(json.dumps(dict(origin_id=spec['origin_id'],status=status['status'],duration_s=float(t[-1]),request_speed_m_s=spec['source_speed_m_s'],exit_or_terminal_speed_m_s=float(tr[-1,2]),max_cross_track_m=float(abs(tr[:,3]).max()),peak_roll_deg=float(np.rad2deg(tr[:,4].max())),role='Source reachability only; not downstream task completion'),indent=2)+'\n')
    raise SystemExit
records=[json.loads(line) for line in (a.run/'records.jsonl').read_text().splitlines() if line.strip()]
traces={r['task_id']:r['trace'] for r in records}
if a.tasks:tasks=[t for t in tasks if t['task_id'] in a.tasks]
fig,axes=plt.subplots(len(tasks),3,figsize=(15,5*len(tasks)),squeeze=False,layout='constrained')
reward_fig,reward_axes=plt.subplots(len(tasks),2,figsize=(12,4*len(tasks)),squeeze=False,layout='constrained')
colors={0.:'tab:blue',.5:'tab:orange',1.:'tab:green'};speeds=sorted({c['speed_m_s'] for c in candidates})
for row,task in enumerate(tasks):
    data=np.load(traces[task['task_id']]);trace=data['trace'];original=data['original_xy'];error=data['original_path_error']
    ax=axes[row];ax[0].plot(original[:,0],original[:,1],':',color='gray',label='Original reference')
    ax[0].plot(*data['goal'],'*',color='black',markersize=10,label='Goal')
    plotted=[]
    for lane,c in enumerate(candidates):
        valid=trace[:,lane,6]>0;tr=trace[valid,lane];err=error[valid,lane]
        if not len(tr):continue
        t=np.flatnonzero(valid)*.005+tr[:,7]*physics_dt
        color=colors[c['alpha']];style='--' if c['speed_m_s']==speeds[0] else '-';label=f"v={c['speed_m_s']:g}, alpha={c['alpha']:g}"
        ax[0].plot(tr[:,0],tr[:,1],style,color=color,label=label);ax[0].plot(tr[-1,0],tr[-1,1],'x',color=color);plotted.append(tr[:,:2])
        ax[1].plot(t,tr[:,2],style,color=color,label=label);ax[2].plot(t,err,style,color=color,label=label)
        reward_axes[row,0].plot(t,tr[:,5],style,color=color,label=label)
        reward_axes[row,1].plot(t,np.cumsum(tr[:,5]),style,color=color,label=label)
    # Fixed reference has long support after the goal; view only the executed task.
    xy=np.concatenate([*plotted,data['goal'][None]]);lo=xy.min(0)-1;hi=xy.max(0)+1
    ax[0].set(xlabel='World x (m)',ylabel='World y (m)',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),title=task['task_id']);ax[0].set_aspect('equal',adjustable='box')
    ax[1].set(xlabel='Time since connection (s)',ylabel='True speed (m/s)');ax[2].set(xlabel='Time since connection (s)',ylabel='Original path distance (m)');ax[2].axhline(spec['corridor_half_width_m'],color='black',ls=':',label='Declared corridor')
    reward_axes[row,0].set(xlabel='Time (s)',ylabel='Recorded lower-step reward',title=task['task_id']);reward_axes[row,1].set(xlabel='Time (s)',ylabel='Sum of recorded lower-step reward')
for ax in [*axes.flat,*reward_axes.flat]:ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.suptitle('Same accessible exit state | frozen H75 | paired nominal DEV\nAlpha colors; speed line styles; crosses are last observed points; replanning enabled for all')
reward_fig.suptitle('Recorded lower-step reward only; upper-step rate/terminal adjustments excluded\nDifferent alpha-weighted rewards are not comparable physical rankings')
fig.savefig(a.run/'xy_candidates.png',dpi=150);reward_fig.savefig(a.run/'recorded_lower_rewards.png',dpi=150)
