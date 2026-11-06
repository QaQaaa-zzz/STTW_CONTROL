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
task_time=None
strict_time=landing.get('first_strict_support_completion_s')
if 'final_goal_xy' in criteria:
 final_hits=np.flatnonzero((np.linalg.norm(post[:,1:3]-criteria['final_goal_xy'],axis=1)<=criteria['waypoint_radius_m']) & (post[:,0]-post[0,0]>=criteria['minimum_post_handoff_s']) & (post[:,0]>=(strict_time if strict_time is not None else float('inf'))) & (post[:,1]-post[0,1]>=criteria['minimum_post_handoff_forward_m'])) if len(post) else np.array([],dtype=int)
 checks['final_goal']=bool(len(final_hits))
 if len(final_hits):task_time=float(post[final_hits[0],0])
jump_path=a.run/'handoff.json'
jump_time=json.loads(jump_path.read_text())['time'] if jump_path.exists() else None
stages=dict(reached_release_line=jump_time is not None,airborne_after_handoff=landing.get('airborne_after_handoff'),strict_support=landing['landing_hold_1s'],ground_return=handoff is not None,continued=checks['continued_time'] and checks['continued_forward'])
decision_path=a.run/'connection_decision.json'
decision_time=json.loads(decision_path.read_text())['time'] if decision_path.exists() else None
violations=np.flatnonzero(error>criteria['corridor_half_width_m'])
prefix=t[:,0]<=decision_time+1e-6 if decision_time is not None else np.ones(len(t),dtype=bool)
prefix_report=dict(decision_time_s=decision_time,decision_reached=decision_time is not None,max_path_error_before_decision_m=float(error[prefix].max()),corridor_intact_at_decision=bool((error[prefix]<=criteria['corridor_half_width_m']).all()) if decision_time is not None else None,first_corridor_violation_s=float(t[violations[0],0]) if len(violations) else None,scope='Observed full-task corridor history; no relabeling of complete or success_time')
report=dict(decision_prefix=prefix_report,stages=stages,stage_times=dict(release=jump_time,strict_support=strict_time,ground_return=handoff,final_conditions=task_time),role='Single nominal DEV, no robustness claim',complete=all(checks.values()),success_time_s=task_time if all(checks.values()) else None,first_final_conditions_time_s=task_time,checks=checks,criteria=criteria,passed_waypoints=passed,max_original_path_error_m=float(error.max()),path_rmse_m=float(np.sqrt(np.mean(error**2))),post_handoff_s=post_time,post_handoff_forward_m=post_distance,landing=landing)
midpoint_path=a.run/'midpoint_decision.json'
if midpoint_path.exists():
 midpoint_time=json.loads(midpoint_path.read_text())['time'];before=t[:,0]<=midpoint_time+1e-6
 report['midpoint_prefix']=dict(decision_time_s=midpoint_time,decision_reached=True,max_path_error_before_decision_m=float(error[before].max()),corridor_intact_at_decision=bool((error[before]<=criteria['corridor_half_width_m']).all()),scope='Observed corridor history through actual midpoint, including the first connection segment')
if spec.get('forward_branch'):
 branch_path=a.run/'forward_branch_decision.json'
 branch=json.loads(branch_path.read_text()) if branch_path.exists() else None
 executed=branch['executed_route'] if branch else 'original_jump'
 branch_map=json.loads(Path(spec['forward_branch']['map_contract']).read_text())
 route_error=error;half_width=criteria['corridor_half_width_m']
 if executed=='bypass':
  mapped=np.load(branch_map['allowed_route_map'])['xy'];edges=np.diff(mapped,axis=0)
  along=((t[:,None,1:3]-mapped[None,:-1])*edges).sum(-1)/(edges*edges).sum(-1)
  route_error=np.linalg.norm(t[:,None,1:3]-mapped[None,:-1]-along.clip(0,1)[...,None]*edges,axis=-1).min(-1)
  half_width=branch_map['half_width_m']
 hits=np.flatnonzero(np.linalg.norm(t[:,1:3]-branch_map['common_goal_xy'],axis=1)<=criteria.get('goal_radius_m',2.))
 prefix_passed=len(passed)>=spec['forward_branch']['common_waypoint_count']
 destination_complete=bool(checks['full_window'] and prefix_passed and len(hits) and (route_error<=half_width).all())
 jump_complete=bool(checks['stable_landing'] and checks['ground_return'] and checks['continued_time'] and checks['continued_forward'])
 report['forward_task']=dict(executed_route=executed,branch_event=branch,
  common_goal_xy=branch_map['common_goal_xy'],full_window=checks['full_window'],common_prefix_passed=prefix_passed,
  first_goal_arrival_s=float(t[hits[0],0]) if len(hits) else None,
  max_mapped_path_error_m=float(route_error.max()),mapped_corridor_half_width_m=half_width,
  mapped_corridor_complete=bool((route_error<=half_width).all()),
  optional_platform_task_complete=destination_complete,
  mandatory_platform_task_complete=destination_complete and jump_complete,
  platform_skill_complete=jump_complete,
  bypass_exit_complete=destination_complete and executed=='bypass',
  scope='E5 declared mapped-route center corridor and full window; bypass cannot complete mandatory platform skill. Not a body-envelope clearance guarantee. Original jump-task labels retained separately.')
(a.run/'chain_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
