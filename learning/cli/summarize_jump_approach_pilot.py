"""Summarize every declared approach case, including pending and failed cases."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
m=json.loads(a.manifest.read_text());rows=[];prefixes={}
lengths=sorted({c['distance_m'] for c in m['cases']});fig,axes=plt.subplots(len(lengths),2,figsize=(12,4*len(lengths)),squeeze=False,layout='constrained')
for c in m['cases']:
 spec=json.loads(Path(c['config']).read_text());run=Path(spec['output']);row=dict(**c,status='pending',complete=None)
 summary=run/'chain_summary.json'
 if summary.exists():
  result=json.loads(summary.read_text());row.update(status='evaluated',complete=result['complete'],success_time_s=result['success_time_s'],checks=result['checks'],max_original_path_error_m=result['max_original_path_error_m'],stages=result.get('stages'),stage_times=result.get('stage_times'))
  decision=run/'connection_decision.json';handoff=run/'handoff.json'
  row['decision']=json.loads(decision.read_text()) if decision.exists() else None
  row['handoff']=json.loads(handoff.read_text()) if handoff.exists() else None
  d=np.load(run/'trajectory.npz');t=d['trace'][d['valid'][:,0],0];i=lengths.index(c['distance_m']);color={0.:'tab:blue',.5:'tab:orange',1.:'tab:green'}[c['alpha']];style='--' if c['speed_m_s']==1.8 else '-';label=f"v={c['speed_m_s']}, alpha={c['alpha']}"
  row['observed_duration_s']=float(t[-1,0])
  row['failure_phase']=('none' if result['complete'] else 'upstream' if row['decision'] is None else 'approach' if row['handoff'] is None else 'jump_or_landing' if not result['checks']['stable_landing'] else 'ground_return' if not result['checks']['ground_return'] else 'continuation_or_corridor')
  for key in ('decision','handoff'):
   if row[key] is None:continue
   event=row[key];root=np.asarray(event['root'])[0];qx,qy,qz,qw=root[3:7];yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));sample=t[np.argmin(np.abs(t[:,0]-event['time']))]
   row[key+'_state']=dict(time_s=event['time'],x_m=float(root[0]),y_m=float(root[1]),forward_speed_m_s=float(sample[4]),lateral_velocity_heading_frame_m_s=float(-np.sin(yaw)*root[7]+np.cos(yaw)*root[8]),heading_rad=float(yaw),world_yaw_rate_rad_s=float(root[12]),roll_rad=float(sample[5]),roll_rate_rad_s=float(sample[6]),steer_rad=float(sample[7]),steer_rate_rad_s=float(sample[8]))
  if row['decision'] is not None:prefixes[c['case_id']]=t[t[:,0]<=row['decision']['time']+.000001]
  axes[i,0].plot(t[:,1],t[:,2],style,color=color,label=label);axes[i,0].plot(t[-1,1],t[-1,2],'x',color=color)
  axes[i,1].plot(t[:,0],t[:,4],style,color=color,label=label)
  if not axes[i,0].get_title():
   route=np.load(run/'ground_route.npz')['xy'];axes[i,0].plot(route[:,0],route[:,1],':',color='gray',label='Original map route')
   axes[i,0].set_title(f"Approach {c['distance_m']} m")
 elif (run/'results.json').exists():row['status']='simulation_complete_analysis_pending'
 elif (run/'runtime.log').exists():row['status']='started_no_final_result'
 rows.append(row)
baseline=next((c['case_id'] for c in m['cases'] if c['speed_m_s']==2. and c['alpha']==.5 and c['case_id'] in prefixes),None)
for row in rows:
 if baseline is None or row['case_id'] not in prefixes:continue
 reference=prefixes[baseline];observed=prefixes[row['case_id']]
 times,ri,oi=np.intersect1d(reference[:,0],observed[:,0],return_indices=True)
 if not len(times):continue
 delta=np.abs(reference[ri]-observed[oi])
 row['prefix_comparison']=dict(reference_case=baseline,matched_samples=len(times),through_time_s=float(times[-1]),max_position_difference_m=float(np.linalg.norm(reference[ri,1:4]-observed[oi,1:4],axis=1).max()),max_speed_difference_m_s=float(delta[:,4].max()),max_roll_difference_rad=float(delta[:,5].max()),max_roll_rate_difference_rad_s=float(delta[:,6].max()),scope='Observed uninterrupted prefix comparison, not proof of hidden PhysX cache equality')
for ax in axes[:,0]:ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim')
for ax in axes[:,1]:ax.set(xlabel='Time (s)',ylabel='Measured forward speed (m/s)')
for ax in axes.flat:
 ax.grid(alpha=.2)
 if ax.lines:ax.legend(fontsize=8)
fig.suptitle('Frozen update2 / H75 + ECBC | same nominal origin | connection-only candidates\nUnexecuted cases remain pending; crosses mark actual endpoints')
fig.savefig(a.manifest.parent/'xy_candidates.png',dpi=150)
(a.manifest.parent/'summary.json').write_text(json.dumps(dict(protocol_id=m['protocol_id'],role=m['role'],declared_cases=len(rows),evaluated_cases=sum(r['status']=='evaluated' for r in rows),rows=rows),indent=2)+'\n')
print(json.dumps({r['case_id']:dict(status=r['status'],complete=r['complete']) for r in rows},indent=2))
