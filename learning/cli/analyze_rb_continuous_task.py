"""Summarize full-route execution after the existing landing contact analysis."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
spec=json.loads((a.run/'protocol.json').read_text())['spec'];criteria=spec['completion']
d=np.load(a.run/'trajectory.npz');trace=d['trace'][:,0];valid=d['valid'][:,0];t=trace[valid]
route=np.load(a.run/'ground_route.npz')['xy'];delta=route[1:]-route[:-1]
u=((t[:,None,1:3]-route[None,:-1])*delta).sum(-1)/(delta*delta).sum(-1)
error=np.linalg.norm(t[:,None,1:3]-route[None,:-1]-u.clip(0,1)[...,None]*delta,axis=-1).min(-1)
# Ordered waypoint encounters exclude jumping directly to a later stage.
passed=[];cursor=0
for point in criteria['waypoints']:
 hits=np.flatnonzero(np.linalg.norm(t[cursor:,1:3]-point,axis=-1)<=criteria['waypoint_radius_m'])
 if not len(hits):break
 cursor+=int(hits[0]);passed.append(dict(point=point,time_s=float(t[cursor,0])))
landing=json.loads((a.run/'landing.json').read_text())['cases'][0]
handoff_path=a.run/'landing_handoff.json';handoff=json.loads(handoff_path.read_text())['time'] if handoff_path.exists() else None
post=t[t[:,0]>=handoff] if handoff is not None else t[:0]
post_time=float(post[-1,0]-post[0,0]) if len(post) else 0.
post_distance=float(post[-1,1]-post[0,1]) if len(post) else 0.
results=json.loads((a.run/'results.json').read_text());case=results['cases'][0]
checks=dict(full_window=bool(case['duration_s']>=spec['seconds']-.006 and not case['physical_termination']),ordered_approach=len(passed)==len(criteria['waypoints']),original_path_corridor=bool((error<=criteria['corridor_half_width_m']).all()),stable_landing=landing['landing_hold_1s'],ground_return=handoff is not None,continued_time=post_time>=criteria['minimum_post_handoff_s'],continued_forward=post_distance>=criteria['minimum_post_handoff_forward_m'])
report=dict(role='Single nominal DEV, no robustness claim',complete=all(checks.values()),checks=checks,criteria=criteria,passed_waypoints=passed,max_original_path_error_m=float(error.max()),path_rmse_m=float(np.sqrt(np.mean(error**2))),post_handoff_s=post_time,post_handoff_forward_m=post_distance,landing=landing)
(a.run/'chain_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
