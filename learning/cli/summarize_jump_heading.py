"""Summarize saved paired-entry jump trajectories without hiding early failures."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
s=json.loads((r/'protocol.json').read_text())['spec'];data=np.load(r/'trajectory.npz');body=np.load(r/'body_contact.npz')
landing=json.loads((r/'landing.json').read_text())['cases'];results=json.loads((r/'results.json').read_text())['cases']
rows=[];groups={};chassis=list(body['body_names']).index('chasis')
for lane,model in enumerate(s['skills']):
 label=model
 if s.get('jump_heading_feedback'):label+=f"_gain{s['jump_heading_feedback']['yaw_gain'][lane]:g}"
 mask=data['valid'][:,lane];t=data['trace'][mask,lane];root=body['body_state'][mask,lane,chassis]
 x,y,z,w=root[:,3:7].T;yaw=np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
 prefix=t[:,0]<=3.;on_route=t[:,1]<=s.get('platform_back_x',7.6)
 row=dict(group=label,entry=s['entry_indices'][lane],lane=lane,observed_s=float(t[-1,0]),full_3s=bool(t[-1,0]>=3),full_8s=not results[lane]['physical_termination'],strict_landing=landing[lane]['landing_hold_1s'],contact20_landing=landing[lane]['contact_filtered_20ms_hold_1s'],observed_prefix3_y_rmse_m=float(np.sqrt(np.mean(t[prefix,2]**2))),observed_prefix3_yaw_rmse_deg=float(np.sqrt(np.mean(yaw[prefix]**2))),through_platform_max_abs_y_m=float(abs(t[on_route,2]).max()),through_platform_max_abs_yaw_deg=float(abs(yaw[on_route]).max()),final_y_m=float(t[-1,2]))
 rows.append(row);groups.setdefault(label,[]).append(row)
summary={}
for group,values in groups.items():
 summary[group]=dict(n=len(values),full_3s=sum(v['full_3s'] for v in values),full_8s=sum(v['full_8s'] for v in values),strict_landing=sum(v['strict_landing'] for v in values),contact20_landing=sum(v['contact20_landing'] for v in values))
 for key in ['observed_prefix3_y_rmse_m','observed_prefix3_yaw_rmse_deg','through_platform_max_abs_y_m','through_platform_max_abs_yaw_deg']:
  summary[group]['median_'+key]=float(np.median([v[key] for v in values]))
report=dict(scope='Same saved TRAIN-origin entry states; DEV. Prefix statistics use only actual observed samples; early termination counts must accompany errors. Not independent TEST or full-chain completion.',summary=summary,cases=rows)
(r/'heading_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
