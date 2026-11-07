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
if protocol['spec'].get('initial_state_bank'):
 plt.close(fig)
 groups=protocol['spec']['skills']
 if protocol['spec'].get('jump_heading_feedback'):
  groups=[f'{skill}: yaw gain {gain:g}' for skill,gain in zip(groups,protocol['spec']['jump_heading_feedback']['yaw_gain'])]
 models=list(dict.fromkeys(groups));indices=protocol['spec']['entry_indices']
 fig,axes=plt.subplots(2,len(models),figsize=(6*len(models),9),layout='constrained',sharex='row',sharey='row',squeeze=False)
 cmap=plt.get_cmap('viridis');normal=matplotlib.colors.Normalize(min(indices),max(indices))
 for col,model in enumerate(models):
  for lane,skill in enumerate(groups):
   if skill!=model:continue
   a=t[v[:,lane],lane];color=cmap(normal(indices[lane]))
   for row,y in [(0,2),(1,3)]:
    axes[row,col].plot(a[:,1],a[:,y],color=color,alpha=.8)
    axes[row,col].scatter(a[-1,1],a[-1,y],color=color,marker='x',s=20)
  axes[0,col].axhline(0,color='gray',ls=':',label='Platform centerline')
  axes[0,col].add_patch(matplotlib.patches.Rectangle((3.6,-1),4,2,fill=False,ls='--',color='gray'))
  axes[0,col].set(title=model,xlabel='World x (m)',ylabel='World y (m)');axes[0,col].set_aspect('equal',adjustable='box')
  axes[1,col].plot([1,3.6,3.6,7.6],[0,0,.16,.16],color='gray',ls=':')
  axes[1,col].set(xlabel='World x (m)',ylabel='Root height (m)')
 for axis in axes.flat:axis.grid(alpha=.2)
 fig.colorbar(matplotlib.cm.ScalarMappable(norm=normal,cmap=cmap),ax=axes,label='Paired saved entry index')
 fig.suptitle('Same 21 actual entry states and policy histories; TRAIN-origin DEV\nAll trajectories stop at observed endpoint; crosses are endpoints, not success labels')
 fig.savefig(p/'xy_jump.png',dpi=150)
 raise SystemExit(0)
landing_path=p/'landing_handoff.json'
landing=json.loads(landing_path.read_text())['time'] if landing_path.exists() else None
branch=json.loads((p/'forward_branch_decision.json').read_text()) if (p/'forward_branch_decision.json').exists() else None
for i,c in enumerate(r['cases']):
 a=t[v[:,i],i];label='H75+ECBC bypass (jump unused)' if branch and branch['executed_route']=='bypass' else c['skill'];color=plt.get_cmap('tab10')(i)
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
if protocol['spec'].get('forward_branch'):
 branch_map=json.loads(Path(protocol['spec']['forward_branch']['map_contract']).read_text())
 mapped=np.load(branch_map['allowed_route_map'])['xy'];last=t[-1,0,1:3]
 end=int(np.argmin(np.linalg.norm(mapped-last,axis=1)))
 ax[0].plot(mapped[:end+51,0],mapped[:end+51,1],'--',color='purple',label='Declared bypass map')
 ax[0].plot(*branch_map['common_goal_xy'],'*',color='black',label='Shared destination')
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
trial_role='Frozen-policy paired DEV'
if protocol['spec'].get('external_force_pulse'):
 force=json.loads((p/'external_force.json').read_text());trial_role='External-force paired DEV'
 if force['triggered']:
  begin=force['trigger_time_s'];end=begin+force['protocol']['duration_s']
  label=f"World force {force['protocol']['force_world_N']} N"
  for axis in (ax[2],ax[3],ax[4],ax[5]):axis.axvspan(begin,end,color='red',alpha=.2,label=label)
for a in ax:a.grid(alpha=.25);a.legend(fontsize=7)
fig.suptitle(f'Common pendulum bicycle | {trial_role}\nActual trajectories stop at the observed endpoint; root height is not wheel clearance')
fig.savefig(p/'xy_jump.png',dpi=150)
