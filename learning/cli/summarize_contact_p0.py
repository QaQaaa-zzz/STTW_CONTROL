"""Matched terrain-minus-flat P0 diagnostics; no fitted causal claim."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--terrains',nargs='+',choices=['oblique_4cm','oblique_6cm'],default=['oblique_4cm','oblique_6cm']);a=p.parse_args()
lanes=json.loads((a.root/'lanes.json').read_text())
flat=np.load(a.root/'flat/physics_wrench.npz');flat={k:flat[k] for k in flat.files}
rows=[]
for terrain in a.terrains:
    archive=np.load(a.root/terrain/'physics_wrench.npz');d={k:archive[k] for k in archive.files}
    np.testing.assert_array_equal(d['time_s'],flat['time_s'])
    events=json.loads((a.root/terrain/'contact_event_windows.json').read_text())
    for lane,identity in enumerate(lanes):
        row=dict(terrain=terrain,**identity,windows={})
        for name,window in events['lanes'][lane]['windows'].items():
            lo,hi=window['start_s'],window['end_s']
            paired=(d['time_s']>=lo)&(d['time_s']<hi)&d['alive'][:,lane]&flat['alive'][:,lane]
            if not paired.any():
                row['windows'][name]=dict(paired_steps=0);continue
            measurements={}
            for label,source in [('terrain',d),('flat',flat)]:
                dt=float(source['time_s'][1]-source['time_s'][0]);mom=np.column_stack((source['moment'][:,lane,0],source['friction_components'][:,lane,:2]))[paired]
                force=source['normal_force'][paired,lane,0]
                measurements[label]=dict(signed_impulse_Nms=(mom.sum(0)*dt).tolist(),absolute_impulse_Nms=(abs(mom).sum(0)*dt).tolist(),net_absolute_impulse_Nms=float(abs(mom.sum(1)).sum()*dt),front_near_zero_load_s=float((np.linalg.norm(force,axis=1)<.1).sum()*dt),steer_rate_rms_rad_s=float(np.sqrt(np.mean(source['steer_velocity'][paired,lane]**2))))
                for field in ['negative_trail_lateral_moment','restore_reference_lateral_delta']:
                    torque=source[field][paired,lane]
                    measurements[label][field+'_signed_impulse_Nms']=float(torque.sum()*dt)
                    measurements[label][field+'_absolute_impulse_Nms']=float(np.abs(torque).sum()*dt)
                trail=source['friction_patch_min_trail'][paired,lane]
                measurements[label]['negative_trail_duration_s']=float((trail<0).sum()*dt)
            delta={key:(np.asarray(measurements['terrain'][key])-np.asarray(measurements['flat'][key])).tolist() for key in measurements['flat']}
            row['windows'][name]=dict(start_s=lo,end_s=hi,paired_steps=int(paired.sum()),**measurements,terrain_minus_flat=delta)
        rows.append(row)
report=dict(role='same speed/alpha/frozen parameter seed; same clock windows and shared alive mask; descriptive terrain-minus-flat differences, not same-state causal intervention',components=['normal','longitudinal_friction','lateral_friction'],conditions=rows)
name='paired_contact_summary.json' if set(a.terrains)=={'oblique_4cm','oblique_6cm'} else 'paired_contact_'+ '_'.join(a.terrains)+'.json'
(a.root/name).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(paired_conditions=len(rows))))
