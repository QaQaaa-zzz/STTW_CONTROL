"""Summarize every declared approach case, including pending and failed cases."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--include-comparison-references',action='store_true');a=p.parse_args()
m=json.loads(a.manifest.read_text());rows=[];prefixes={};midpoint_prefixes={}
if a.include_comparison_references:
 references={}
 for c in m['cases']:
  spec=json.loads(Path(c['config']).read_text());ref=spec['comparison_reference']
  references[ref['config']]=dict(c,case_id=c['case_id']+'_reference_'+ref['skill'],config=ref['config'],skill=ref['skill'],reused_reference=True)
 m['cases'].extend(references.values())
skills=sorted({c['skill'] for c in m['cases'] if 'skill' in c})
methods=sorted({c['method'] for c in m['cases'] if 'method' in c})
sequences=any('second_candidate' in c for c in m['cases'])
for c in m['cases']:
 c['origin_id']=json.loads(Path(c['config']).read_text())['origin_id']
 if sequences:c.update(speed_m_s=c['second_candidate']['speed_m_s'],alpha=c['second_candidate']['alpha'])
 c['panel_key']=((c['distance_m'],c['skill']) if skills else (c['distance_m'],c['first_candidate']['speed_m_s'],c['first_candidate']['alpha']) if sequences else (c['distance_m'],))
panels={origin:sorted({tuple(c['panel_key']) for c in m['cases'] if c['origin_id']==origin}) for origin in {c['origin_id'] for c in m['cases']}};figures={}
for c in m['cases']:
 spec=json.loads(Path(c['config']).read_text());origin=spec['origin_id'];run=Path(spec['output']);row=dict(c,origin_id=origin,status='pending',complete=None)
 summary=run/'chain_summary.json'
 if summary.exists():
  if origin not in figures:figures[origin]=plt.subplots(len(panels[origin]),2,figsize=(12,4*len(panels[origin])),squeeze=False,layout='constrained')
  fig,axes=figures[origin]
  result=json.loads(summary.read_text());row.update(status='evaluated',complete=result['complete'],success_time_s=result['success_time_s'],checks=result['checks'],max_original_path_error_m=result['max_original_path_error_m'],stages=result.get('stages'),stage_times=result.get('stage_times'))
  decision=run/'connection_decision.json';handoff=run/'handoff.json'
  row['decision']=json.loads(decision.read_text()) if decision.exists() else None
  row['handoff']=json.loads(handoff.read_text()) if handoff.exists() else None
  d=np.load(run/'trajectory.npz');t=d['trace'][d['valid'][:,0],0];i=panels[origin].index(tuple(c['panel_key']));color={0.:'tab:blue',.5:'tab:orange',1.:'tab:green'}[c['alpha']];style='--' if c['speed_m_s']==1.8 else '-';label=f"v={c['speed_m_s']}, alpha={c['alpha']}"+(" after midpoint" if sequences else "")
  if skills:label=c['skill']+' | '+label
  if c.get('method'):
   color=plt.get_cmap('tab10')(methods.index(c['method']));label=c['method'];style='-'
   midpoint=run/'midpoint_decision.json'
   row['midpoint']=json.loads(midpoint.read_text()) if midpoint.exists() else None
   if row['midpoint'] is not None:axes[i,1].axvline(row['midpoint']['time'],color=color,alpha=.2)
  row['observed_duration_s']=float(t[-1,0])
  if sequences:
   midpoint=run/'midpoint_decision.json'
   row['midpoint']=json.loads(midpoint.read_text()) if midpoint.exists() else None
   if row['midpoint'] is not None:
    axes[i,1].axvline(row['midpoint']['time'],color=color,alpha=.2)
    midpoint_prefixes[c['case_id']]=t[t[:,0]<=row['midpoint']['time']+.000001]
   if c['diagonal']:
    paired_spec=json.loads(Path(spec['paired_C6_config']).read_text());paired=Path(paired_spec['output'])
    if (paired/'chain_summary.json').exists():
     pd=np.load(paired/'trajectory.npz');pt=pd['trace'][pd['valid'][:,0],0]
     times,pi,ti=np.intersect1d(pt[:,0],t[:,0],return_indices=True)
     pr=json.loads((paired/'chain_summary.json').read_text())
     row['diagonal_C6_comparison']=dict(reference=str(paired),matched_samples=len(times),reference_samples=len(pt),sequence_samples=len(t),reference_duration_s=float(pt[-1,0]),sequence_duration_s=float(t[-1,0]),same_complete_label=bool(pr['complete']==result['complete']),reference_checks=pr['checks'],sequence_checks=result['checks'],reference_T_s=pr['success_time_s'],sequence_T_s=result['success_time_s'],max_position_difference_m=float(np.linalg.norm(pt[pi,1:4]-t[ti,1:4],axis=1).max()) if len(times) else None,max_speed_difference_m_s=float(np.abs(pt[pi,4]-t[ti,4]).max()) if len(times) else None,scope='Matched observed full trajectories; unequal lengths and labels retained, no hidden simulator-state equality claim')

  row['decision_prefix']=result.get('decision_prefix')
  row['failure_phase']=('none' if result['complete'] else 'upstream_corridor_before_decision' if result.get('decision_prefix',{}).get('corridor_intact_at_decision') is False else 'upstream' if row['decision'] is None else 'approach' if row['handoff'] is None else 'jump_or_landing' if not result['checks']['stable_landing'] else 'ground_return' if not result['checks']['ground_return'] else 'continuation_or_corridor')
  for key in ('decision','handoff'):
   if row[key] is None:continue
   event=row[key];root=np.asarray(event['root'])[0];qx,qy,qz,qw=root[3:7];yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));sample=t[np.argmin(np.abs(t[:,0]-event['time']))]
   row[key+'_state']=dict(time_s=event['time'],x_m=float(root[0]),y_m=float(root[1]),forward_speed_m_s=float(sample[4]),lateral_velocity_heading_frame_m_s=float(-np.sin(yaw)*root[7]+np.cos(yaw)*root[8]),heading_rad=float(yaw),world_yaw_rate_rad_s=float(root[12]),roll_rad=float(sample[5]),roll_rate_rad_s=float(sample[6]),steer_rad=float(sample[7]),steer_rate_rad_s=float(sample[8]))
  prefix_event=row['handoff'] if skills else row['decision']
  if prefix_event is not None:prefixes[c['case_id']]=t[t[:,0]<=prefix_event['time']+.000001]
  axes[i,0].plot(t[:,1],t[:,2],style,color=color,label=label);axes[i,0].plot(t[-1,1],t[-1,2],'x',color=color)
  axes[i,1].plot(t[:,0],t[:,4],style,color=color,label=label)
  if not axes[i,0].get_title():
   route=np.load(run/'ground_route.npz')['xy'];axes[i,0].plot(route[:,0],route[:,1],':',color='gray',label='Original map route')
   axes[i,0].set_title(f"Approach {c['distance_m']} m"+(f" | {c['skill']}" if skills else "")+(f" | first v={c['first_candidate']['speed_m_s']}, alpha={c['first_candidate']['alpha']}" if sequences else ""))
 elif (run/'results.json').exists():row['status']='simulation_complete_analysis_pending'
 elif (run/'runtime.log').exists():row['status']='started_no_final_result'
 rows.append(row)
baselines={origin:next((r['case_id'] for r in rows if r['origin_id']==origin and r['speed_m_s']==2. and r['alpha']==.5 and r['case_id'] in prefixes),next((r['case_id'] for r in rows if r['origin_id']==origin and r['case_id'] in prefixes),None)) for origin in figures}
for row in rows:
 baseline=(next((r['case_id'] for r in rows if r['origin_id']==row['origin_id'] and r['distance_m']==row['distance_m'] and r['speed_m_s']==row['speed_m_s'] and r['alpha']==row['alpha'] and r['case_id'] in prefixes),None) if skills else baselines.get(row['origin_id']))
 if baseline is None or row['case_id'] not in prefixes:continue
 reference=prefixes[baseline];observed=prefixes[row['case_id']]
 times,ri,oi=np.intersect1d(reference[:,0],observed[:,0],return_indices=True)
 if not len(times):continue
 delta=np.abs(reference[ri]-observed[oi])
 row['prefix_comparison']=dict(reference_case=baseline,matched_samples=len(times),through_time_s=float(times[-1]),max_position_difference_m=float(np.linalg.norm(reference[ri,1:4]-observed[oi,1:4],axis=1).max()),max_speed_difference_m_s=float(delta[:,4].max()),max_roll_difference_rad=float(delta[:,5].max()),max_roll_rate_difference_rad_s=float(delta[:,6].max()),scope='Observed uninterrupted prefix comparison, not proof of hidden PhysX cache equality')
if sequences:
 for row in rows:
  if row['case_id'] not in midpoint_prefixes:continue
  base=next(r for r in rows if r['origin_id']==row['origin_id'] and r['distance_m']==row['distance_m'] and r['first_candidate']==row['first_candidate'] and r['diagonal'])
  if base['case_id'] not in midpoint_prefixes:continue
  reference=midpoint_prefixes[base['case_id']];observed=midpoint_prefixes[row['case_id']]
  times,ri,oi=np.intersect1d(reference[:,0],observed[:,0],return_indices=True)
  row['midpoint_prefix_comparison']=dict(reference_case=base['case_id'],matched_samples=len(times),reference_samples=len(reference),sequence_samples=len(observed),reference_midpoint_s=base['midpoint']['time'],sequence_midpoint_s=row['midpoint']['time'],max_observed_state_difference=float(np.abs(reference[ri,1:]-observed[oi,1:]).max()) if len(times) else None,scope='Same first reference, complete stored trajectory through decision before applying second reference; no hidden simulator-state claim')
for origin,(fig,axes) in figures.items():
 for ax in axes[:,0]:ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim')
 for ax in axes[:,1]:ax.set(xlabel='Time (s)',ylabel='Measured forward speed (m/s)')
 for ax in axes.flat:
  ax.grid(alpha=.2)
  if ax.lines:ax.legend(fontsize=8)
 fig.suptitle(f'Frozen policy / H75 + ECBC | origin {origin} | connection-only candidates\nUnexecuted cases remain pending; crosses mark actual endpoints')
 name='xy_candidates.png' if len(figures)==1 else f'xy_candidates_{origin}.png'
 fig.savefig(a.manifest.parent/name,dpi=150)
 plt.close(fig)
(a.manifest.parent/'summary.json').write_text(json.dumps(dict(protocol_id=m.get('protocol_id',json.loads(Path(m['cases'][0]['config']).read_text())['protocol_id']),role=m['role'],declared_cases=len(rows),evaluated_cases=sum(r['status']=='evaluated' for r in rows),rows=rows),indent=2)+'\n')
print(json.dumps({r['case_id']:dict(status=r['status'],complete=r['complete']) for r in rows},indent=2))
