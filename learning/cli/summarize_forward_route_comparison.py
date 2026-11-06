"""Summarize actual mapped-route outcomes, async timing and retained references."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifests',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();rows=[]
for path in a.manifests:
 for case in json.loads(path.read_text())['cases']:
  spec=json.loads(Path(case['config']).read_text());run=Path(spec['output']);summary=run/'chain_summary.json'
  row=dict(case_id=case['case_id'],config=case['config'],origin_id=spec['origin_id'],status='pending',
   method=case.get('method'),disturbance_protocol=spec.get('external_force_pulse'))
  if summary.exists():
   result=json.loads(summary.read_text());task=result['forward_task'];event=task['branch_event']
   row['route_prefix']=task.get('route_prefix')
   if event is None:
    row.update(status='evaluated_no_route_decision',executed_route=task['executed_route'],optional_platform_complete=task['optional_platform_task_complete'],mandatory_platform_complete=task['mandatory_platform_task_complete'],full_window=task['full_window'],arrival_s=task['first_goal_arrival_s'],max_mapped_path_error_m=task['max_mapped_path_error_m'],source_summary=str(summary))
    rows.append(row);continue
   query=event.get('query_state')
   row.update(status='evaluated',executed_route=task['executed_route'],requested_route=event['requested_route'],
    optional_platform_complete=task['optional_platform_task_complete'],mandatory_platform_complete=task['mandatory_platform_task_complete'],
    full_window=task['full_window'],arrival_s=task['first_goal_arrival_s'],max_mapped_path_error_m=task['max_mapped_path_error_m'],
    selection=event.get('selection'),selection_wall_s=event['selection_wall_s'],query_time_s=query['time'] if query else None,
    application_time_s=event['time_s'],application_delay_sim_s=event.get('application_delay_sim_s'),
    asynchronous=event.get('asynchronous',False),available_routes_at_application=event['availability']['available_routes'],
    requested_route_unavailable=event['requested_route'] not in event['availability']['available_routes'])
   if query:
    before=np.asarray(query['root'])[0];after=np.asarray(event['root'])[0]
    row['world_xy_motion_during_query_m']=float(np.linalg.norm(after[:2]-before[:2]))
    with np.load(run/'trajectory.npz') as d:
     t=d['trace'][d['valid'][:,0],0];window=t[(t[:,0]>query['time'])&(t[:,0]<=event['time_s']+1e-6)]
     row['stored_samples_after_request_through_application']=len(window)
   row['source_summary']=str(summary)
  rows.append(row)
paired={}
for row in rows:
 if row['method'] in ('rule','capability'):
  key=(row['origin_id'],json.dumps(row['disturbance_protocol'],sort_keys=True))
  if row['method'] in paired.setdefault(key,{}):raise ValueError('Repeated method for one origin/disturbance pair')
  paired[key][row['method']]=row
comparisons=[]
for (origin,pulse),methods in paired.items():
 ready=all(m in methods and methods[m]['status'].startswith('evaluated') for m in ('rule','capability'))
 pair=dict(origin_id=origin,disturbance_protocol=json.loads(pulse),status='evaluated' if ready else 'pending')
 if ready:
  rule,capability=methods['rule'],methods['capability']
  pair['case_ids']={m:r['case_id'] for m,r in methods.items()}
  for task in ('optional_platform','mandatory_platform'):
   a,b=rule[task+'_complete'],capability[task+'_complete']
   pair[task]=dict(rule_complete=a,capability_complete=b,
    outcome='both_complete' if a and b else 'capability_only' if b else 'rule_only' if a else 'both_failed',
    capability_minus_rule_arrival_s=capability['arrival_s']-rule['arrival_s'] if a and b else None)
 comparisons.append(pair)
result=dict(role='Actual forward-route DEV and interface timing; no pooled success claim across task roles',declared_cases=len(rows),evaluated_cases=sum(r['status'].startswith('evaluated') for r in rows),rows=rows,
 paired_comparisons=comparisons,paired_distinct_origins=len({r['origin_id'] for r in comparisons if r['status']=='evaluated'}),
 limits='Wall query latency and simulated control advance are distinct. A zero simulated delay does not demonstrate pending-query motion. Mandatory platform success requires the platform skill; bypass is at most safe exit. Pending rows are not failures; actual XY is retained in each source run.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(declared_cases=result['declared_cases'],evaluated_cases=result['evaluated_cases'])))
