"""Causal features and supervised data for candidate tolerance/task prediction.

Historical H75 q exports have a different model/physics/data contract and are
not loaded here. Source observations and explicit hypothetical interventions
form inputs; later physical outcomes are targets only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA='sttw_conditional_outcome42_v1'
STATE_INDICES=tuple(range(300,320))+tuple(range(337,351))


def outcome_features(observation,latent,source_yaw,case,interface):
    obs=np.asarray(observation,np.float32);z=np.asarray(latent,np.float32)
    if obs.shape!=(351,) or z.shape!=(2,) or not np.isfinite(np.r_[obs,z,source_yaw]).all():
        raise ValueError('finite current native351 observation, candidate latent and source yaw required')
    if abs(float(obs[336])-case['alpha'])>1e-6:raise ValueError('queried candidate alpha differs')
    start=case['source_task_start_s'];prep=case['preparation_s']
    if not 0<=start<10 or prep<0 or (case['force_N'] and abs(case['onset_s']-start-prep)>1e-6):
        raise ValueError('explicit source clock and hypothetical preparation required')
    direction=np.asarray(case.get('force_direction_world',[0.,1.,0.]),float)
    world=case['force_N']*direction;yaw=float(source_yaw)
    body=np.array([np.cos(yaw)*world[0]+np.sin(yaw)*world[1],-np.sin(yaw)*world[0]+np.cos(yaw)*world[1]])
    a=np.tanh(z);target=np.array([a[0]*(interface['speed_positive_scale_m_s'] if a[0]>=0 else interface['speed_negative_scale_m_s']),a[1]*interface['steer_scale_rad']])
    query=np.r_[case['alpha'],target[0],target[1]/.2,body/32.,prep/.6,case['duration_s']/.4,(10.-start)/10.]
    features=np.r_[obs[list(STATE_INDICES)],query].astype(np.float32)
    if features.shape!=(42,) or not np.isfinite(features).all():raise ValueError('conditional feature contract differs')
    return features


def prepare_dataset(run,output):
    """Freeze all declared TRAIN/DEV groups after successful independent analysis."""
    run=Path(run).resolve();output=Path(output).resolve()
    plan=json.loads((run/'plan.json').read_text());rows=json.loads((run/'analysis/results.json').read_text())
    status=json.loads((run/'evaluation/status.json').read_text());receipt=json.loads((run/'analysis/receipt.json').read_text())
    shape=json.loads((run/'evaluation/shape_receipt.json').read_text());partition=plan['protocol']['dataset_partition']
    if (status['phase']!='completed' or status['completed_cases']!=len(rows) or len(rows)!=plan['cases']
        or not receipt['execution_reward_reconstructed'] or not shape.get('all_nonalpha_source_leaves_exact')
        or not shape.get('paired_reset_all_leaves_exact') or not shape.get('conditional_complete_state_copy')):
        raise ValueError('complete authenticated conditional branch analysis required')
    inputs={};sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path in (run/'plan.json',run/'analysis/results.json',run/'analysis/receipt.json',run/'evaluation/shape_receipt.json',run/'source_replay_audit.json'):
        inputs[str(path)]=sha(path)
    if inputs[str(run/'plan.json')]!=json.loads((run/'seal.json').read_text())['plan_sha256']:raise ValueError('conditional plan changed')
    xs=[];ys=[];masks=[];roles=[];metadata=[];source_observations={}
    for row in rows:
        if row['endpoint']['physical_outcome'] not in ('survived','contact') or row['endpoint']['goal_arrived'] is None:
            raise ValueError('engineering/unknown outcome cannot become a negative target')
        with np.load(row['raw_file']) as z:raw=dict(z)
        n=row['active_lower_ticks'];obs=raw['query_observation'][0]
        neutral=np.delete(obs,336)
        if row['route'] in source_observations:
            np.testing.assert_allclose(neutral,source_observations[row['route']],atol=1e-6,rtol=0,err_msg='same-state source query differs beyond alpha')
        else:source_observations[row['route']]=neutral
        xs.append(outcome_features(obs,raw['composed_latent'][0],raw['pre_upper_yaw'][0],row,plan['config']['residual_interface']))
        with np.load(row['diagnostic_task_file']) as z:cross=np.asarray(z['original_cross_track_m'])
        survived=row['endpoint']['physical_outcome']=='survived';goal=survived and row['endpoint']['goal_arrived']
        first=row['endpoint']['first_arrival_s'];stop=min(n,round((first-row['source_task_start_s'])/.005)) if goal else n
        if stop<=0 or len(cross)!=n:raise ValueError('aligned original task cost window required')
        path_rmse=float(np.sqrt(np.mean(cross[:stop]**2)));peak=row['full_upright_max_abs_speed_m_s']
        if peak is None or not np.isfinite([path_rmse,peak]).all():raise ValueError('finite physical regression targets required')
        ys.append([float(survived),float(goal),np.log1p(path_rmse),np.log1p(peak)])
        # Do not reward early failure with an artificially small partial-window error.
        masks.append([float(survived),float(survived)])
        dev=row['force_N']==partition['development_force_N'] and row['preparation_s']==partition['development_preparation_s']
        roles.append('development' if dev else 'training')
        metadata.append(dict(case_id=row['case_id'],group=row['route']+'/'+row['event'],route=row['route'],alpha=row['alpha'],force_N=row['force_N'],preparation_s=row['preparation_s'],source_task_start_s=row['source_task_start_s'],path_rmse_to_goal_or_window_m=path_rmse,full_upright_max_abs_speed_m_s=peak,no_fall=survived,task_completed=bool(goal),regression_targets_valid=survived,role=roles[-1]))
    if roles.count('training')!=partition['training_rows'] or roles.count('development')!=partition['development_rows']:
        raise ValueError('predeclared TRAIN/DEV counts differ')
    for group in {m['group'] for m in metadata}:
        members=[m for m in metadata if m['group']==group]
        if len({m['role'] for m in members})!=1 or sorted(m['alpha'] for m in members)!=plan['protocol']['alphas']:
            raise ValueError('source/challenge alpha siblings cannot split across data roles')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'dataset.npz',features=np.asarray(xs),targets=np.asarray(ys,np.float32),regression_mask=np.asarray(masks,np.float32),roles=np.asarray(roles))
    manifest=dict(schema=SCHEMA,source_run=str(run),source_plan_sha256=inputs[str(run/'plan.json')],source_model=plan['models']['H75'],feature_dimension=42,state_indices=list(STATE_INDICES),query_features=['candidate_alpha','candidate_speed_correction_m_s','candidate_steer_correction_div_0p2rad','hypothetical_body_force_x_div32N','hypothetical_body_force_y_div32N','preparation_div0p6s','duration_div0p4s','remaining_task_time_div10s'],targets=['no_fall_to_original_task10','survived_and_entered_original_goal','log1p_original_path_rmse_to_goal_or_window','log1p_peak_upright_abs_speed'],partition=partition,records=metadata,inputs_sha256=inputs,dataset_sha256=sha(output/'dataset.npz'),limitations='Related four-state TRAIN/DEV, deterministic branch labels. Inputs contain hypothetical queries, never actual future sensor values. Only declared original task<=10s is labeled; extra raw suffix is preserved separately. No independent TEST, probability calibration or continuous safety guarantee.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
