"""Export decision-time inputs and whole-task labels without future-state leakage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
fields=[('speed','m/s'),('roll','rad'),('roll_rate','rad/s'),('cross_track','m'),('heading_error','rad'),('steer','rad'),('steer_rate','rad/s'),('previous_speed_reference','m/s'),('lateral_velocity_heading_frame','m/s'),('world_yaw_rate','rad/s'),('previous_alpha','1'),('previous_alpha_duration','s'),('remaining_path_to_release','m'),('path_curvature','1/m'),('corridor_half_width','m'),('pitch','rad'),('body_angular_y','rad/s'),('hip','rad'),('knee','rad'),('hip_velocity','rad/s'),('knee_velocity','rad/s'),('platform_height','m'),('remaining_path_to_platform','m')]
schema=dict(schema_version='rb_jump_exit_features_v1',features=[dict(name=n,unit=u) for n,u in fields],candidate_features=[dict(name='speed_m_s',unit='m/s'),dict(name='alpha',unit='1')],deployment_sources='IMU, localization/velocity estimation, joint encoders, command history and known map. Simulation state substitutes measurements in this DEV; deployment estimation not validated.',missing_joint_velocity='Older protocol lacks live joint ordering: retain null, never assume donor or URDF order. These rows cannot train the full feature set.',previous_alpha_duration='At first turn-exit decision only: elapsed episode time under unchanged upstream alpha.',excluded_inputs=['handoff_state','future_contacts','future_disturbance','max_path_error','terminal_outcome'],normalization='Not fitted; fit only on training roots if supervised learning is justified')
rows=[];inventory=[];identities={}
for case in m['cases']:
 spec=json.loads(Path(case['config']).read_text());run=Path(spec['output']);result=run/'chain_summary.json'
 if not result.exists():
  inventory.append(dict(case_id=case['case_id'],status='not_evaluated'));continue
 result=json.loads(result.read_text());protocol=json.loads((run/'protocol.json').read_text());decision_path=run/'connection_decision.json'
 decision=json.loads(decision_path.read_text()) if decision_path.exists() else None
 ground=json.loads((run/'ground_controller.json').read_text())
 plant_cfg=json.loads(Path(spec['plant_config']).read_text());contract=Path(plant_cfg['asset']).with_name('contract.json')
 key=str(contract)
 if key not in identities:identities[key]=hashlib.sha256(contract.read_bytes()).hexdigest()
 map_identity=json.loads((Path(spec['scene_terrain'])/'identity.json').read_text())
 state=None;named=None
 if decision is not None:
  root=np.asarray(decision['root'])[0];measurement=np.asarray(decision['measurement'])[0]
  qx,qy,qz,qw=root[3:7];yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
  route=np.load(run/'ground_route.npz');xy=route['xy'];edges=np.diff(xy,axis=0);u=np.clip(((root[:2]-xy[:-1])*edges).sum(-1)/(edges*edges).sum(-1),0,1);points=xy[:-1]+u[:,None]*edges;i=int(np.argmin(((points-root[:2])**2).sum(-1)))
  progress=float(route['s'][i]+u[i]*(route['s'][i+1]-route['s'][i]));heading=float(np.interp(progress,route['s'],route['heading']));delta=root[:2]-points[i]
  release_index=int(np.argmin(np.linalg.norm(xy-[spec['handoff_x'],0.],axis=1)));edge_index=int(np.argmin(np.linalg.norm(xy-[spec['platform_front_x'],0.],axis=1)))
  model=json.loads(Path(spec['model_config']).read_text());mount=np.asarray(model['mount_rotation_wxyz'])[[1,2,3,0]];rotation=Rotation.from_quat(root[3:7])*Rotation.from_quat(mount)
  # Only recorded live articulation ordering can bind joint velocity slots.
  live=protocol.get('joint_names')
  if live is None:
   trajectory=np.load(run/'trajectory.npz');trace=trajectory['trace'][:,0];sample=trace[np.argmin(abs(trace[:,0]-decision['time']))]
   hip=float(sample[9]);knee=float(sample[10]);hip_v=knee_v=None
  else:
   h=live.index('hip_joint');k=live.index('knee_joint');hip=decision['position'][0][h];knee=decision['position'][0][k];hip_v=decision['velocity'][0][h];knee_v=decision['velocity'][0][k]
  values=[float(np.cos(yaw)*root[7]+np.sin(yaw)*root[8]),float(measurement[0]),float(measurement[1]),float(-np.sin(heading)*delta[0]+np.cos(heading)*delta[1]),float(np.arctan2(np.sin(heading-yaw),np.cos(heading-yaw))),float(measurement[2]),float(measurement[3]),decision['previous_reference'][0][0],float(-np.sin(yaw)*root[7]+np.cos(yaw)*root[8]),float(root[12]),decision['previous_alpha'][0],decision['time'],float(route['s'][release_index]-progress),float(np.interp(progress,route['s'],route['curvature'])),spec['completion']['corridor_half_width_m'],float(rotation.as_euler('xyz')[1]),float(rotation.inv().apply(root[10:13])[1]),hip,knee,hip_v,knee_v,.16,float(route['s'][edge_index]-progress)]
  named=dict(zip([n for n,_ in fields],values));state=values
 total=result['success_time_s']
 rows.append(dict(protocol_id=spec['protocol_id'],feature_schema=schema['schema_version'],origin_id=spec['origin_id'],case_id=case['case_id'],data_role='pilot_DEV',model_contract_sha256=identities[key],map_sha256=map_identity['sha256'],policy_sha256=dict(ground=ground['upper_sha256'],jump=protocol['policies'][0]['artifact_sha256']),zero_lower=spec['zero_lower'],physics_dt=protocol['physics_dt'],candidate=spec['connection_candidate'],state=state,named_state=named,decision_reached=decision is not None,supervised_input_complete=state is not None and all(v is not None for v in state),Y=int(result['complete']),T_total_s=total,T_after_decision_s=total-decision['time'] if total is not None and decision else None,stages=result['stages'],checks=result['checks'],analysis_only=dict(max_original_path_error_m=result['max_original_path_error_m']),prefix=dict(path=str(run/'trajectory.npz'),decision_time_s=decision['time'] if decision else None,replay='uninterrupted full-prefix execution, no snapshot restore'),disturbance_protocol='D0',source_config=case['config']))
 inventory.append(dict(case_id=case['case_id'],status='evaluated',decision_reached=decision is not None))
(a.output/'feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n');(a.output/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(a.output/'inventory.json').write_text(json.dumps(dict(declared_cases=len(m['cases']),evaluated_cases=len(rows),independent_origins=len({r['origin_id'] for r in rows}),role='Pilot DEV only; no train/validation/test split or fitted model',cases=inventory),indent=2)+'\n')
print(json.dumps(dict(exported_rows=len(rows),independent_origins=len({r['origin_id'] for r in rows}))))
