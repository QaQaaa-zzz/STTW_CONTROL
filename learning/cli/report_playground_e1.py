"""All E1 DEV roots, fixed-map task outcomes and executed prediction errors."""
import argparse,csv,json
from pathlib import Path
from collections import Counter
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sttw_control.playground_arx import states,robot_radius_bound,geometry_error_bound
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();manifest=json.loads((a.root/'manifest.json').read_text());query=np.load(a.root/'query_states.npz');episodes=[];fig,axes=plt.subplots(4,3,figsize=(18,13),layout='constrained');colors={'geometry':'#0072B2','dynamic':'#D55E00'}
for entry,ax in zip(manifest['entries'],axes.flat):
 folder=Path(entry['config']).parent;ev=Path(entry['evaluation']);status=json.loads((ev/'status.json').read_text());plans=json.loads((folder/'plans.json').read_text());world=json.loads((folder/'world_map.json').read_text());schema=json.loads((ev/'model_state_schema.json').read_text());radius,jr=robot_radius_bound(mujoco.MjModel.from_binary_path(str(ev/'terrain.mjb')))
 for row in status['results']:
  root=row['route']//2;method=['geometry','dynamic'][row['route']%2];plan=next(x for x in plans if x['root']==root and x['method']==method);data=np.load(ev/row['case_dir']/'trajectory.npz');trace=data['trace'];pose,xi=states(data['qpos'],data['qvel'],schema);prediction=np.load(folder/f'{method}_root{root}_prediction.npz');actual_query=int(np.argmin(abs(trace[:,0]-.02)))
  record=dict(source_evaluation=entry.get('reused_evaluation',str(ev)),layout=entry['layout'],root=root,method=method,plan_feasible=plan['feasible'],physical_goal=row['goal'],task_success=bool(plan['feasible'] and row['goal']),primary_outcome=row['termination'] if plan['feasible'] else 'no_plan',physical_termination=row['termination'],duration_s=row['duration_s'],penalized_time_s=row['duration_s'] if plan['feasible'] and row['goal'] else manifest['seconds'],planning_ms=plan['planning_s']*1000,publish_s=entry['publish_s'],query_pose_max_difference=float(abs(pose[actual_query]-query['pose'][root]).max()),query_xi_max_difference=float(abs(xi[actual_query]-query['xi'][root]).max()),forbidden_contact_steps=row['forbidden_contact_steps'],max_tracking_error_m=float(trace[:,11].max()),max_abs_roll_rad=float(abs(trace[:,4]).max()),prediction_observed_s=None,max_executed_xy_error_m=None,max_executed_body_bound_m=None)
  if plan['feasible']:
   actual=pose[actual_query::4][:len(prediction['predicted_pose'])];pred=prediction['predicted_pose'][:len(actual)];record.update(prediction_observed_s=(len(actual)-1)*.02,max_executed_xy_error_m=float(np.linalg.norm(pred[:,:2]-actual[:,:2],axis=1).max()),max_executed_body_bound_m=float(geometry_error_bound(pred,actual,radius,jr).max()))
  episodes.append(record);color=colors[method];ax.plot(*pose[:,:2].T,color=color,alpha=.8,lw=1,label=method if root==0 else None);ax.plot(*pose[-1,:2],marker='o' if row['goal'] else 'x',color=color,ms=4)
  ref=data['published_reference_xy'];mask=(ref[:,0]>=1.2)&(ref[:,0]<=8.8);ax.plot(*ref[mask].T,'--',color=color,alpha=.45,lw=.7)
 for xmin,xmax,ymin,ymax in world['terrain']['forbidden_rectangles']:ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,color='#777777',alpha=.6))
 goal=world['goal'];ax.add_patch(Rectangle((goal['x_min_m'],-goal['abs_y_max_m']),goal['x_max_m']-goal['x_min_m'],2*goal['abs_y_max_m'],color='#009E73',alpha=.12));ax.set(title=f'Layout{entry["layout"]}: 3 paired roots',xlabel='World x (m)',ylabel='World y (m)',xlim=(1.2,8.8),ylim=(-1.6,1.6),aspect='equal');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('E1 DEV: actual solid; published reference dashed; x real failure; gray forbidden obstacle; green common goal');fig.savefig(a.root/'xy.png',dpi=160);plt.close(fig)
with (a.root/'episodes.csv').open('w') as stream:writer=csv.DictWriter(stream,fieldnames=list(episodes[0]),lineterminator="\n");writer.writeheader();writer.writerows(episodes)
(a.root/'episodes.json').write_text(json.dumps(episodes,indent=2));summary=[]
for method in ['geometry','dynamic']:
 z=[x for x in episodes if x['method']==method];latency=np.array([x['planning_ms'] for x in z]);summary.append(dict(method=method,total=len(z),plan_feasible=sum(x['plan_feasible'] for x in z),physical_goal=sum(x['physical_goal'] for x in z),task_success=sum(x['task_success'] for x in z),outcomes=dict(Counter(x['primary_outcome'] for x in z)),physical_terminations=dict(Counter(x['physical_termination'] for x in z)),penalized_mean_s=float(np.mean([x['penalized_time_s'] for x in z])),planning_median_ms=float(np.median(latency)),planning_p95_ms=float(np.quantile(latency,.95)),planning_max_ms=float(max(latency)),planning_over_200ms=int((latency>200).sum())))
(a.root/'summary.json').write_text(json.dumps(summary,indent=2));lines=['# E1 ground planning DEV pilot','','![All paired actual/published reference XY](xy.png)','','Static saved-state planning with shared delayed publication; no live rolling query or independent TEST. No-plan cases execute their existing straight reference but remain planner failures. See manifest for exact frozen physics, skills, maps, roots, budget and common target.','','|Method|Plans /36|Physical goal /36|Task success /36|Penalized mean s|Planner median/P95/max ms|','|---|---:|---:|---:|---:|---:|']
for s in summary:lines.append(f'|{s["method"]}|{s["plan_feasible"]}|{s["physical_goal"]}|{s["task_success"]}|{s["penalized_mean_s"]:.3f}|{s["planning_median_ms"]:.2f}/{s["planning_p95_ms"]:.2f}/{s["planning_max_ms"]:.2f}|')
lines+=['','|Layout|Root|Method|Plan|Outcome|Physical outcome|Time s|Query pose/velocity gap|Executed XY error m|','|---|---|---|---|---|---|---|---|---|']
for z in episodes:lines.append(f'|{z["layout"]}|{z["root"]}|{z["method"]}|{z["plan_feasible"]}|{z["primary_outcome"]}|{z["physical_termination"]}|{z["duration_s"]:.3f}|{z["query_pose_max_difference"]:.3g}/{z["query_xi_max_difference"]:.3g}|{z["max_executed_xy_error_m"]}|')
lines+=['','Prediction comparisons use only the committed, actually executed route and stop at the real terminal state. Rejected hypothetical curves are not scored as executed predictions. The shared .36m engineering margin is not a probability or safety guarantee. Planner timing includes both topologies and GPU synchronization but not live state acquisition, initial graph compilation or the physical publication call. No claim of complete online 200ms acceptance.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
