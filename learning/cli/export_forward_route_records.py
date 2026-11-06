"""Export observed E5 route decisions and distinct optional/mandatory task labels."""
import argparse
import hashlib
import json
from pathlib import Path
from sttw_control.jump_connection import decision_features,fields

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifests',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=[];inventory=[]
for manifest in a.manifests:
 for case in json.loads(manifest.read_text())['cases']:
  spec=json.loads(Path(case['config']).read_text());run=Path(spec['output'])
  summary=run/'chain_summary.json'
  if not summary.exists():
   inventory.append(dict(case_id=case['case_id'],status='pending'));continue
  result=json.loads(summary.read_text());task=result['forward_task'];event=task['branch_event']
  if event is None:
   inventory.append(dict(case_id=case['case_id'],origin_id=spec['origin_id'],status='evaluated_no_route_decision',optional_platform_complete=task['optional_platform_task_complete'],mandatory_platform_complete=task['mandatory_platform_task_complete'],source_summary=str(summary)))
   continue
  protocol=json.loads((run/'protocol.json').read_text());ground=json.loads((run/'ground_controller.json').read_text())
  state_event=event.get('decision_state');source='branch_decision'
  if state_event is None and (run/'connection_decision.json').exists():
   connection=json.loads((run/'connection_decision.json').read_text())
   # Legacy route choices immediately followed the connection update, with no physics step.
   if connection['time']==event['time_s'] and connection['root']==event['root']:
    state_event=dict(connection);alpha=connection['candidate']['alpha']
    state_event['previous_alpha_duration_s']=connection['previous_alpha_duration_s'] if alpha==connection['previous_alpha'][0] else 0.
    state_event['previous_alpha']=[alpha];source='same_tick_same_root_connection_event_after_reference_assignment'
  state,named=decision_features(state_event,run,protocol,spec) if state_event else (None,None)
  plant=json.loads(Path(spec['plant_config']).read_text());contract=Path(plant['asset']).with_name('contract.json')
  route=event['executed_route'];candidate=spec['connection_candidate'];arrival=task['first_goal_arrival_s']
  common=dict(origin_id=spec['origin_id'],case_id=case['case_id'],data_role='pilot_DEV',target_semantics='Q_route',
   feature_schema='rb_jump_decision_features_v2',state=state,named_state=named,supervised_input_complete=state is not None and all(v is not None for v in state),state_source=source,
   policy_sha256=dict(ground=ground['upper_sha256'],jump=protocol['policies'][0]['artifact_sha256']),
   model_contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),map_id=hashlib.sha256(Path(spec['forward_branch']['map_contract']).read_bytes()).hexdigest(),
   execution_reference_sha256=hashlib.sha256(Path(spec['ground_route']).read_bytes()).hexdigest(),
   physics_dt=protocol['physics_dt'],zero_lower=spec['zero_lower'],skill_id='matched_runtime',route_id=route,
   candidate_id=f"v{candidate['speed_m_s']}_a{candidate['alpha']}",candidate=candidate,
   decision_time_s=event['time_s'],available_routes=event['availability']['available_routes'],
   disturbance_protocol=spec.get('external_force_pulse'),source_config=case['config'],source_summary=str(summary),
   analysis_only=dict(full_window=task['full_window'],max_mapped_path_error_m=task['max_mapped_path_error_m'],platform_skill_complete=task['platform_skill_complete']))
  for requirement in ('optional','mandatory'):
   complete=task[requirement+'_platform_task_complete']
   rows.append(dict(common,task_id=route+':'+requirement,platform_requirement=requirement,complete=int(complete),success_cost=arrival-event['time_s'] if complete else None))
  inventory.append(dict(case_id=case['case_id'],status='exported',state_source=source,supervised_input_complete=common['supervised_input_complete']))
(a.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(a.output/'feature_schema.json').write_text(json.dumps(dict(features=[dict(name=n,unit=u) for n,u in fields],target='Full-window mapped-route completion; conditional first goal arrival time minus decision time',platform_requirement='Separate task_id; mandatory bypass is failure even if safe exit succeeds',limits='Same-root disturbance descendants are not independent roots. Leave entire origin out. Legacy state recovery requires identical recorded root and timestamp.'),indent=2)+'\n')
report=dict(records=len(rows),physical_cases=len(inventory),independent_origins=len({json.loads(Path(c['config']).read_text())['origin_id'] for m in a.manifests for c in json.loads(m.read_text())['cases']}),usable_records=sum(r['supervised_input_complete'] for r in rows),cases=inventory)
(a.output/'inventory.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
