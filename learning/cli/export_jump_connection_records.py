"""Export decision-time inputs and whole-task labels without future-state leakage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--summary',type=Path,help='Export only cases evaluated in this frozen collection summary');p.add_argument('--label-protocol',choices=['strict','contact20'],default='strict');p.add_argument('--protocol-id',help='Declared common protocol identity for this export');a=p.parse_args()
m=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
allowed={r['case_id'] for r in json.loads(a.summary.read_text())['rows'] if r['status']=='evaluated'} if a.summary else None
from sttw_control.jump_connection import fields, decision_features
schema=dict(schema_version='rb_jump_decision_features_v2',features=[dict(name=n,unit=u) for n,u in fields],candidate_features=[dict(name='speed_m_s',unit='m/s'),dict(name='alpha',unit='1')],categorical_features=[dict(name='decision_stage',values=['exit','midpoint'])],deployment_sources='IMU, localization/velocity estimation, joint encoders, command history and known map. Simulation state substitutes measurements in this DEV; deployment estimation not validated.',missing_joint_velocity='Older protocol lacks live joint ordering: retain null, never assume donor or URDF order. These rows cannot train the full feature set.',previous_alpha_duration='Recorded duration since last alpha change; legacy exit records use episode time under unchanged upstream alpha.',excluded_inputs=['handoff_state','future_contacts','future_disturbance','max_path_error','terminal_outcome'],normalization='Not fitted; fit only on training roots if supervised learning is justified')

rows=[];midpoint_rows=[];inventory=[];identities={}
for case in m['cases']:
 spec=json.loads(Path(case['config']).read_text());run=Path(spec['output']);result=run/'chain_summary.json'
 if not result.exists() or (allowed is not None and case['case_id'] not in allowed):
  inventory.append(dict(case_id=case['case_id'],status='not_evaluated'));continue
 result=json.loads(result.read_text());protocol=json.loads((run/'protocol.json').read_text());decision_path=run/'connection_decision.json'
 decision=json.loads(decision_path.read_text()) if decision_path.exists() else None
 ground=json.loads((run/'ground_controller.json').read_text())
 plant_cfg=json.loads(Path(spec['plant_config']).read_text());contract=Path(plant_cfg['asset']).with_name('contract.json')
 key=str(contract)
 if key not in identities:identities[key]=hashlib.sha256(contract.read_bytes()).hexdigest()
 map_identity=json.loads((Path(spec['scene_terrain'])/'identity.json').read_text())
 state,named=decision_features(decision,run,protocol,spec) if decision is not None else (None,None)
 strict_result=result
 if a.label_protocol=='contact20':result=dict(result,**result['contact_filtered_protocol'])
 protocol_id=a.protocol_id or (spec['protocol_id']+('__contact20ms' if a.label_protocol=='contact20' else ''))
 gain=spec.get('rear_speed_feedback_gain',0.)
 if isinstance(gain,list):gain=gain[protocol['batch_execution']['lane']]
 controller_contract=dict(rear_speed_feedback_gain=gain,landing_return=spec.get('landing_return'),jump_steering_gains=spec.get('jump_steering_gains'),ground_settings=spec.get('ground_settings'))
 total=result['success_time_s']
 rows.append(dict(protocol_id=protocol_id,label_protocol=a.label_protocol,strict_Y=int(strict_result['complete']),controller_contract=controller_contract,feature_schema=schema['schema_version'],origin_id=spec['origin_id'],case_id=case['case_id'],data_role='pilot_DEV',model_contract_sha256=identities[key],map_sha256=map_identity['sha256'],policy_sha256=dict(ground=ground['upper_sha256'],jump=protocol['policies'][0]['artifact_sha256']),zero_lower=spec['zero_lower'],physics_dt=protocol['physics_dt'],candidate=spec['connection_candidate'],state=state,named_state=named,decision_reached=decision is not None,supervised_input_complete=state is not None and all(v is not None for v in state),Y=int(result['complete']),T_total_s=total,T_after_decision_s=total-decision['time'] if total is not None and decision else None,stages=result['stages'],checks=result['checks'],analysis_only=dict(max_original_path_error_m=result['max_original_path_error_m']),decision_prefix=result.get('decision_prefix'),prefix=dict(path=str(run/'trajectory.npz'),decision_time_s=decision['time'] if decision else None,replay='uninterrupted full-prefix execution, no snapshot restore'),disturbance_protocol=spec.get('external_force_pulse','D0'),source_config=case['config']))
 sequence=[spec['connection_candidate']]+([spec['connection_midpoint_candidate']] if spec.get('connection_midpoint_candidate') else [])
 rows[-1].update(decision_stage='exit',target_semantics='fixed_two_stage_sequence' if len(sequence)>1 else 'Q_hold',candidate_sequence=sequence)
 if decision is not None:rows[-1]['candidate']=decision['candidate']
 if spec.get('connection_selector'):rows[-1]['target_semantics']='adaptive_remaining_task'
 midpoint_path=run/'midpoint_decision.json'
 if midpoint_path.exists():
  event=json.loads(midpoint_path.read_text());mid_state,mid_named=decision_features(event,run,protocol,spec)
  midpoint_rows.append(dict(rows[-1],case_id=case['case_id']+':midpoint',decision_stage='midpoint',target_semantics='Q_hold',
      parent_candidate=rows[-1]['candidate'],candidate=event['candidate'],candidate_sequence=[event['candidate']],
      decision_prefix=result.get('midpoint_prefix'),
      state=mid_state,named_state=mid_named,decision_reached=True,supervised_input_complete=all(v is not None for v in mid_state),
      T_after_decision_s=total-event['time'] if total is not None else None,
      prefix=dict(path=str(run/'trajectory.npz'),decision_time_s=event['time'],replay='Actual reached midpoint in uninterrupted execution; no synthetic state substitution')))
 inventory.append(dict(case_id=case['case_id'],status='evaluated',decision_reached=decision is not None))
(a.output/'feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n');(a.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(a.output/'inventory.json').write_text(json.dumps(dict(declared_cases=len(m['cases']),evaluated_cases=len(rows),independent_origins=len({r['origin_id'] for r in rows}),role='Pilot DEV only; no train/validation/test split or fitted model',cases=inventory),indent=2)+'\n')
(a.output/'midpoint_hold_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in midpoint_rows))
print(json.dumps(dict(exported_rows=len(rows),midpoint_rows=len(midpoint_rows),independent_origins=len({r['origin_id'] for r in rows}))))

# Reuse the established eight-feature Gaussian baseline, without pretending the
# richer schema or the nominal pilot constitutes a trained neural evaluator.
from collections import defaultdict
base=json.loads(Path('learning/configs/connection_capability_dev.json').read_text())
names=[f['name'] for f in base['features']]
case_info={c['case_id']:c for c in m['cases']}
paired=defaultdict(list)
for row in rows:
    skill=json.dumps(dict(policies=row['policy_sha256'],model=row['model_contract_sha256'],zero_lower=row['zero_lower'],dt=row['physics_dt'],controller_contract=row['controller_contract']),sort_keys=True)
    disturbance=json.dumps(row['disturbance_protocol'],sort_keys=True)
    paired[(row['origin_id'],case_info[row['case_id']]['distance_m'],row['protocol_id'],skill,disturbance)].append(row)
kernel=[];groups=[]
for (origin,distance,protocol,skill,disturbance),group in paired.items():
    ready=len(group)==6 and all(r['decision_reached'] and r['target_semantics']=='Q_hold' and (r['decision_prefix'] or {}).get('corridor_intact_at_decision') is not False for r in group)
    groups.append(dict(origin_id=origin,distance_m=distance,protocol_id=protocol,skill_id=skill,disturbance_protocol=json.loads(disturbance),executed_rows=len(group),kernel_ready=ready,reason='complete_decision_group' if ready else 'upstream_failure_prior_corridor_violation_incomplete_group_or_non_hold_target'))
    if not ready:continue
    for row in group:
        candidate=row['candidate'];cid=f"v{candidate['speed_m_s']}_a{candidate['alpha']}"
        kernel.append(dict(origin_id=origin,decision_id=f'{origin}:approach_{distance}:{disturbance}',disturbance_protocol=json.loads(disturbance),
            task_id=f"{row['protocol_id']}:approach_{distance}",
            skill_id=skill,
            candidate_id=cid,state=[row['named_state'][name] for name in names],
            complete=row['Y'],success_cost=row['T_after_decision_s'],
            source_case_id=row['case_id'],scope='Uninterrupted decision record, not a serialized full simulator snapshot'))
base.update(candidates=[f'v{v}_a{alpha}' for v in (1.8,2.) for alpha in (0.,.5,1.)],
            role='Fixed eight-feature kernel DEV baseline for jump connection; no fitting or calibrated-probability claim',
            cost='Time after decision to full-task completion, success conditional',
            validation='Hold out whole upstream origin across all distances and candidates; upstream failures retained in full records and coverage inventory')
(a.output/'kernel_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in kernel))
(a.output/'kernel_config.json').write_text(json.dumps(base,indent=2)+'\n')
(a.output/'kernel_coverage.json').write_text(json.dumps(dict(groups=groups,full_records='records.jsonl',excluded_from_kernel_not_from_task_denominator=True),indent=2)+'\n')

# The same grouped evaluator can query actual reached midpoints, excluding the
# entire upstream origin rather than leaking its other first-action branches.
mid_kernel=[]
for row in midpoint_rows:
    if not row['supervised_input_complete'] or (row['decision_prefix'] or {}).get('corridor_intact_at_decision') is not True:
        continue
    skill=json.dumps(dict(policies=row['policy_sha256'],model=row['model_contract_sha256'],zero_lower=row['zero_lower'],dt=row['physics_dt'],controller_contract=row['controller_contract']),sort_keys=True)
    decision_id=json.dumps(dict(origin=row['origin_id'],map=row['map_sha256'],parent=row['parent_candidate'],disturbance=row['disturbance_protocol']),sort_keys=True)
    candidate=row['candidate']
    mid_kernel.append(dict(origin_id=row['origin_id'],decision_id=decision_id,
        task_id='midpoint',skill_id=skill,candidate_id=f"v{candidate['speed_m_s']}_a{candidate['alpha']}",
        state=row['state'],complete=row['Y'],success_cost=row['T_after_decision_s'],source_case_id=row['case_id']))
mid_config=json.loads(Path('learning/configs/jump_connection_capability_dev.json').read_text())
mid_config['candidates']=[f"v{c['speed_m_s']}_a{c['alpha']}" for c in mid_config['candidates']]
mid_config['role']='Actual reached midpoint Q_hold; whole-origin held-out pilot DEV, no fitted model or independent TEST'
(a.output/'midpoint_kernel_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mid_kernel))
(a.output/'midpoint_kernel_config.json').write_text(json.dumps(mid_config,indent=2)+'\n')
