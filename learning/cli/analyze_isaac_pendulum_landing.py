"""Adjudicate platform support from saved geometry and contact telemetry."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--geometry',type=Path)
p.add_argument('--model-config',type=Path)
a=p.parse_args()
protocol=json.loads((a.run/'protocol.json').read_text());spec=protocol['spec']
d=np.load(a.run/'body_contact.npz');states=d['body_state'];forces=d['contact'];valid=d['valid'];time=d['time']
names=list(d['body_names']);geometry=json.loads((a.geometry or Path(spec['geometry'])).read_text())
mount=json.loads((a.model_config or Path(spec['model_config'])).read_text())['mount_rotation_wxyz']
mount=Rotation.from_quat(np.array(mount)[[1,2,3,0]])
root=states[:,:,names.index('chasis')]
angles=(Rotation.from_quat(root[:,:,3:7].reshape(-1,4))*mount).as_euler('xyz').reshape(*root.shape[:2],3)
support=np.ones(valid.shape,bool);wheel_bottom=[]
for name in ('frontwheel','rearwheel'):
    index=names.index(name);g=next(g for g in geometry if g['body']==name)
    body=states[:,:,index];rotation=Rotation.from_quat(body[:,:,3:7].reshape(-1,4))
    center=body[:,:,:3]+rotation.apply(np.tile(g['pos'],(body.shape[0]*body.shape[1],1))).reshape(*valid.shape,3)
    q=np.array(g['quat'])[[1,2,3,0]]
    matrix=(rotation*Rotation.from_quat(q)).as_matrix().reshape(*valid.shape,3,3)
    extent=np.linalg.norm(matrix*np.array(g['size']),axis=-1)
    if protocol['plant'].get('wheel_collision')=='cylinder':
        count=protocol['plant'].get('wheel_profile_slices',1);width=protocol['plant']['wheel_width_m'];radius=protocol['plant']['wheel_radius_m']
        offsets=(np.arange(count)-(count-1)/2)*width/count
        radii=radius*np.sqrt(1-(2*offsets/width)**2)
        radial=np.linalg.norm(matrix[:,:,:,:2],axis=-1)
        axial=np.abs(matrix[:,:,:,2])
        extent=np.max(radial[...,None]*radii+axial[...,None]*(np.abs(offsets)+width/(2*count)),axis=-1)
    bottom=center[:,:,2]-extent[:,:,2];wheel_bottom.append(bottom)
    support&=(center[:,:,0]>=spec.get('platform_front_x',3.6))&(center[:,:,0]<=spec.get('platform_back_x',7.6))&(np.abs(center[:,:,1])+extent[:,:,1]<=1.)
    support&=(np.abs(bottom-.16)<=.015)&(forces[:,:,index,2]>1.)
nonwheel=[i for i,name in enumerate(names) if name not in ('frontwheel','rearwheel')]
no_collision=(np.linalg.norm(forces[:,:,nonwheel],axis=-1)<=1.).all(-1)
upright=(np.abs(angles[:,:,:2])<=np.deg2rad(15)).all(-1)
qualified=support&no_collision&upright&valid
handoff_file=a.run/'handoff.json'
handoff=json.loads(handoff_file.read_text())['time'] if handoff_file.exists() else float('inf')
wheel_ids=[names.index(name) for name in ('frontwheel','rearwheel')]
airborne=(np.linalg.norm(forces[:,:,wheel_ids],axis=-1)<=1.).all(-1)&(np.stack(wheel_bottom,-1)>.02).all(-1)&valid&(time[:,None]>=handoff)
rows=[]
for i,skill in enumerate(spec['skills']):
    changes=np.diff(np.r_[False,qualified[:,i],False].astype(int));starts=np.where(changes==1)[0];ends=np.where(changes==-1)[0]-1
    durations=time[ends]-time[starts];j=int(np.argmax(durations)) if len(durations) else None
    flight_changes=np.diff(np.r_[False,airborne[:,i],False].astype(int));flight_start=np.where(flight_changes==1)[0];flight_end=np.where(flight_changes==-1)[0]-1
    flight=[(b,e) for b,e in zip(flight_start,flight_end) if time[e]-time[b]>=.02]
    first_support=float(time[starts[np.flatnonzero(durations>=1.)[0]]]+1.) if (durations>=1.).any() else None
    rows.append(dict(skill=skill,first_strict_support_completion_s=first_support,airborne_after_handoff=bool(flight),first_airborne_interval_s=[float(time[flight[0][0]]),float(time[flight[0][1]])] if flight else None,landing_hold_1s=bool(len(durations) and durations[j]>=1.),longest_hold_s=float(durations[j]) if j is not None else 0.,interval_s=[float(time[starts[j]]),float(time[ends[j]])] if j is not None else None))
report=dict(role='nominal DEV landing diagnostic, not complete continuous-task success',criteria=dict(platform_x_m=[spec.get('platform_front_x',3.6),spec.get('platform_back_x',7.6)],platform_y_m=[-1,1],top_z_m=.16,wheel_bottom_tolerance_m=.015,min_each_wheel_vertical_force_N=1.,max_nonwheel_contact_force_N=1.,max_abs_roll_pitch_deg=15,continuous_hold_s=1.,airborne_diagnostic=dict(min_wheel_bottom_m=.02,max_each_wheel_contact_N=1.,minimum_duration_s=.02,scope='both wheels clear ground after handoff; not an independent jump success criterion')),cases=rows)
(a.run/'landing.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez(a.run/'landing_diagnostics.npz',time=time,qualified=qualified,wheel_support=support,upright=upright,no_nonwheel_contact=no_collision,wheel_bottom=np.stack(wheel_bottom,-1),roll_pitch=angles[:,:,:2],valid=valid)
print(json.dumps(report,indent=2))
