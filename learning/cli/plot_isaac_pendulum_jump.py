"""Plot all selected jump policies from saved physical trajectories."""
from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
p=args.run;d=np.load(p/'trajectory.npz');t=d['trace'];v=d['valid'];r=json.loads((p/'results.json').read_text());fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained');ax=axes.flatten()
protocol=json.loads((p/'protocol.json').read_text());handoff=json.loads((p/'handoff.json').read_text())['time'] if (p/'handoff.json').exists() else float('inf')
landing_path=p/'landing_handoff.json'
landing=json.loads(landing_path.read_text())['time'] if landing_path.exists() else None
for i,c in enumerate(r['cases']):
 a=t[v[:,i],i];label=c['skill'];color=plt.get_cmap('tab10')(i)
 ax[0].plot(a[:,1],a[:,2],label=label,color=color);ax[0].plot(a[-1,1],a[-1,2],'x',color=color)
 ax[1].plot(a[:,1],a[:,3],label=label,color=color);ax[1].plot(a[-1,1],a[-1,3],'x',color=color)
 ax[2].plot(a[:,0],np.rad2deg(a[:,5]),label=label,color=color)
 ax[3].plot(a[:,0],a[:,4],label=label,color=color)
 mapping=protocol['policies'][i]['mapping'];command=np.clip(mapping['base_rear_speed']+d['action'][v[:,i],i,1]*mapping['rear_speed_delta'],mapping['ctrl_min'][1],mapping['ctrl_max'][1])
 jump=(a[:,0]>=handoff)&((a[:,0]<landing) if landing is not None else True)
 ax[4].plot(a[jump,0],command[jump],label=label,color=color)
 ax[5].plot(a[:,0],a[:,11],label=label,color=color)
if protocol['spec'].get('ground_route'):
 route=np.load(p/'ground_route.npz');endpoints=t[-1,:,:][:,1:3]
 end=max(int(np.argmin(np.linalg.norm(route['xy']-point,axis=1))) for point in endpoints)
 ref=route['xy'][:min(end+51,len(route['xy']))];ax[0].plot(ref[:,0],ref[:,1],color='gray',ls=':',label='Original ground route')
else:ax[0].axhline(0,color='gray',ls=':',label='Straight approach reference')
ax[0].set(xlabel='World x (m)',ylabel='World y (m)');ax[0].set_aspect('equal',adjustable='datalim');ax[0].legend(fontsize=8)
ax[1].plot([1,protocol['spec'].get('platform_front_x',3.6),protocol['spec'].get('platform_front_x',3.6),protocol['spec'].get('platform_back_x',7.6)],[0,0,.16,.16],color='gray',ls=':',label='Platform surface');ax[1].set(xlabel='World x (m)',ylabel='Root height (m)');ax[1].legend(fontsize=8)
ax[2].set(xlabel='Time (s)',ylabel='Roll (deg)');ax[2].legend(fontsize=8)
ax[3].axhline(protocol['spec']['approach_speed'],ls=':',color='gray',label='Approach speed request');ax[3].set(xlabel='Time (s)',ylabel='Measured forward speed (m/s)')
if landing is not None:
 ax[3].hlines(protocol['spec']['landing_return']['speed_m_s'],landing,t[-1,0,0],ls='--',color='black',label='Resumed ground request')
 for axis in (ax[2],ax[3],ax[4],ax[5]):axis.axvline(landing,ls='--',color='green',alpha=.6,label='Landing-to-ground handoff')
ax[4].set(xlabel='Time (s)',ylabel='Raw jump-policy rear target, before transition override (rad/s)')
ax[5].axhline(0,ls=':',color='gray');ax[5].set(xlabel='Time (s)',ylabel='Frontmost collision distance to edge (m)')
if np.isfinite(handoff):
 for a in (ax[2],ax[3],ax[4],ax[5]):a.axvline(handoff,ls=':',color='black',alpha=.5,label='Ground-to-jump handoff')
for a in ax:a.grid(alpha=.25);a.legend(fontsize=7)
fig.suptitle('Common pendulum bicycle | selected frozen checkpoints | nominal paired DEV\nActual trajectories end at physical termination; root height is not wheel clearance')
fig.savefig(p/'xy_jump.png',dpi=150)
