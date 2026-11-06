"""Report matched original skill baselines; distinguish spawn clearance from takeoff."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
m=json.loads(a.manifest.read_text());rows=[];traces={}
for case in m['cases']:
 spec=json.loads(Path(case['config']).read_text());out=Path(spec['output'])
 result=json.loads((out/'results.json').read_text());physical=result['cases'][0]
 landing=json.loads((out/'landing.json').read_text())['cases'][0]
 with np.load(out/'trajectory.npz') as d:t=d['trace'][d['valid'][:,0],0]
 with np.load(out/'body_contact.npz') as d:
  names=d['body_names'].tolist();force=d['contact'][:,0,[names.index('frontwheel'),names.index('rearwheel')]];time=d['time'];valid=d['valid'][:,0]
 with np.load(out/'landing_diagnostics.npz') as d:bottom=d['wheel_bottom'][:,0]
 contact=(force[:,:,2]>1.).any(-1)&valid
 had_contact=np.maximum.accumulate(contact)
 cleared=(np.linalg.norm(force,axis=-1)<=1.).all(-1)&(bottom>.02).all(-1)&valid&had_contact
 change=np.diff(np.r_[False,cleared,False].astype(int));starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)-1
 flight=[(float(time[b]),float(time[e])) for b,e in zip(starts,ends) if time[e]-time[b]>=.02]
 row=dict(case_id=case['case_id'],skill=case['skill'],initial_speed_m_s=spec['approach_speed'],duration_s=physical['duration_s'],
  full_window=not physical['physical_termination'] and physical['duration_s']>=spec['seconds']-.006,
  strict_support=landing['landing_hold_1s'],longest_support_s=landing['longest_hold_s'],
  peak_root_z_m=physical['peak_root_z'],terminal_flags=physical['terminal_flags'],
  raw_first_airborne_interval_s=landing['first_airborne_interval_s'],post_contact_airborne_intervals_s=flight,
  first_wheel_contact_s=float(time[np.flatnonzero(contact)[0]]) if contact.any() else None,
  simulation_wall_s=result['elapsed_s'],physics_steps=result['physics_lane_steps'],source_run=str(out))
 rows.append(row);traces[(row['skill'],row['initial_speed_m_s'])]=t
speeds=sorted({r['initial_speed_m_s'] for r in rows});fig,axes=plt.subplots(4,len(speeds),figsize=(12,12),layout='constrained')
styles={'source':('#0072B2','-'),'rb_run1_candidate':('#D55E00','--')}
vmin=min(t[:,4].min() for t in traces.values());vmax=max(t[:,4].max() for t in traces.values());margin=.05*(vmax-vmin)
for col,speed in enumerate(speeds):
 for skill,(color,style) in styles.items():
  t=traces[(skill,speed)];label='Source' if skill=='source' else 'RB Run1'
  for row,x,y in ((0,t[:,1],t[:,2]),(1,t[:,1],t[:,3]),(2,t[:,0],t[:,4]),(3,t[:,0],np.rad2deg(t[:,5]))):
   axes[row,col].plot(x,y,color=color,ls=style,label=label);axes[row,col].plot(x[-1],y[-1],'x',color=color)
 axes[0,col].add_patch(Rectangle((3.6,-1),4,2,fill=False,edgecolor='gray',ls=':',label='Platform footprint'))
 axes[0,col].axhline(0,color='gray',ls=':',lw=.7);axes[0,col].set(title=f'Initial vx = {speed:g} m/s',xlabel='World x (m)',ylabel='World y (m)',xlim=(1.3,7.8),ylim=(-1.2,1.2));axes[0,col].set_aspect('equal')
 axes[1,col].plot([1.3,3.6,3.6,7.6],[0,0,.16,.16],':',color='gray');axes[1,col].set(xlabel='World x (m)',ylabel='Root height (m)',xlim=(1.3,7.8),ylim=(0,.65))
 axes[2,col].set(xlabel='Time (s)',ylabel='Measured forward speed (m/s)',xlim=(0,4.2),ylim=(vmin-margin,vmax+margin))
 axes[3,col].axhline(-45,color='gray',ls=':');axes[3,col].axhline(45,color='gray',ls=':');axes[3,col].set(xlabel='Time (s)',ylabel='Roll (deg)',xlim=(0,4.2),ylim=(-48,48))
 for axis in axes[:,col]:axis.grid(alpha=.2);axis.legend(fontsize=7)
fig.suptitle('Original Source / RB nominal-start DEV | same Isaac model, 400 Hz physics\nRaw traces end at actual termination; crosses mark endpoints; root height is not wheel clearance')
fig.savefig(a.output/'paired_xy.png',dpi=150);plt.close(fig)
report=dict(role='Six fixed nominal-start DEV cases, not independent recovery-distribution validation',rows=rows,
 full_window=sum(r['full_window'] for r in rows),strict_support=sum(r['strict_support'] for r in rows),
 physics_steps=sum(r['physics_steps'] for r in rows),simulation_wall_s=sum(r['simulation_wall_s'] for r in rows),
 diagnostic='Additional takeoff diagnostic excludes initial clearance before any observed wheel contact. Requires both wheels clear >.02m with <=1N contact for >=.02s after support; does not change strict landing or terminal criteria. Raw airborne fields retained unchanged.')
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('full_window','strict_support','physics_steps','simulation_wall_s')}))
