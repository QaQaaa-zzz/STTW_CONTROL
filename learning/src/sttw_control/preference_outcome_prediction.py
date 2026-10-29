"""Causal features and supervised data for candidate tolerance/task prediction.

Historical H75 q exports have a different model/physics/data contract and are
not loaded here. Source observations and explicit hypothetical interventions
form inputs; later physical outcomes are targets only.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np


SCHEMA='sttw_conditional_outcome42_v1'
RETURN_SCHEMA='sttw_conditional_outcome43_return_v1'
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
    if (direction.shape!=(3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1.)>1e-6
        or direction[2]!=0. or not np.isfinite([case['force_N'],case['duration_s'],start,prep]).all() or case['duration_s']<=0):
        raise ValueError('finite signed force, positive duration and unit horizontal world direction required')
    world=case['force_N']*direction;yaw=float(source_yaw)
    body=np.array([np.cos(yaw)*world[0]+np.sin(yaw)*world[1],-np.sin(yaw)*world[0]+np.cos(yaw)*world[1]])
    a=np.tanh(z);target=np.array([a[0]*(interface['speed_positive_scale_m_s'] if a[0]>=0 else interface['speed_negative_scale_m_s']),a[1]*interface['steer_scale_rad']])
    query=np.r_[case['alpha'],target[0],target[1]/.2,body/32.,prep/.6,case['duration_s']/.4,(10.-start)/10.]
    features=np.r_[obs[list(STATE_INDICES)],query].astype(np.float32)
    if features.shape!=(42,) or not np.isfinite(features).all():raise ValueError('conditional feature contract differs')
    return features


def return_outcome_features(observation,latent,source_yaw,case,interface,original_user_alpha):
    if original_user_alpha is None or not np.isfinite(original_user_alpha) or not 0<=original_user_alpha<=1:raise ValueError('explicit original user alpha required for returning-policy43')
    return np.r_[outcome_features(observation,latent,source_yaw,case,interface),original_user_alpha].astype(np.float32)


def prepare_dataset(run,output,*,partition_override=None,release_note=None,policy_schema=SCHEMA):
    """Freeze all declared TRAIN/DEV groups after successful independent analysis."""
    run=Path(run).resolve();output=Path(output).resolve()
    plan=json.loads((run/'plan.json').read_text());rows=json.loads((run/'analysis/results.json').read_text())
    rule=plan['protocol'].get('causal_preference_return')
    if bool(rule)!=(policy_schema==RETURN_SCHEMA) or policy_schema not in (SCHEMA,RETURN_SCHEMA):raise ValueError('preference-return labels need an explicit user-conditioned policy schema; hold-alpha42 is incompatible')
    status=json.loads((run/'evaluation/status.json').read_text());receipt=json.loads((run/'analysis/receipt.json').read_text())
    shape=json.loads((run/'evaluation/shape_receipt.json').read_text());parent_partition=plan['protocol'].get('dataset_partition')
    partition=parent_partition if partition_override is None else partition_override
    if partition is None or (partition_override is not None and not release_note):raise ValueError('declared partition or explicit release note required')
    if (status['phase']!='completed' or status['completed_cases']!=len(rows) or len(rows)!=plan['cases']
        or not receipt['execution_reward_reconstructed'] or not shape.get('all_nonalpha_source_leaves_exact')
        or not shape.get('paired_reset_all_leaves_exact') or not shape.get('conditional_complete_state_copy')):
        raise ValueError('complete authenticated conditional branch analysis required')
    inputs={};sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    input_paths=[run/'plan.json',run/'analysis/results.json',run/'analysis/receipt.json',run/'evaluation/shape_receipt.json']
    if rule:input_paths.append(run/'source/tolerance_coordination.py')
    declared_groups=None
    if plan['protocol'].get('excluded_completed_cases') or plan['protocol'].get('range_planner_boundary'):
        input_paths.append(run/'inventory.json');declared_groups={}
        for case in json.loads((run/'inventory.json').read_text()):
            declared_groups.setdefault(case['route']+'/'+case['event'],[]).append(case['alpha'])
    source_alphas={c['route']:c['state_source']['source_alpha'] for c in plan['contexts']}
    has_unchanged=any(r['force_N']==0. and r['alpha']==source_alphas[r['route']] for r in rows)
    if has_unchanged:input_paths.append(run/'source_replay_audit.json')
    for path in input_paths:
        inputs[str(path)]=sha(path)
    if inputs[str(run/'plan.json')]!=json.loads((run/'seal.json').read_text())['plan_sha256']:raise ValueError('conditional plan changed')
    xs=[];ys=[];masks=[];roles=[];metadata=[];source_observations={}
    for row in rows:
        if row['endpoint']['physical_outcome'] not in ('survived','contact') or row['endpoint']['goal_arrived'] is None:
            raise ValueError('engineering/unknown outcome cannot become a negative target')
        if sha(row['raw_file'])!=row['raw_sha256']:raise ValueError('physical source trajectory changed after analysis')
        inputs[row['raw_file']]=row['raw_sha256'];inputs[row['diagnostic_task_file']]=sha(row['diagnostic_task_file'])
        with np.load(row['raw_file']) as z:raw=dict(z)
        n=row['active_lower_ticks'];obs=raw['query_observation'][0]
        neutral=np.delete(obs,336)
        if row['route'] in source_observations:
            np.testing.assert_allclose(neutral,source_observations[row['route']],atol=1e-6,rtol=0,err_msg='same-state source query differs beyond alpha')
        else:source_observations[row['route']]=neutral
        if rule and not row.get('preference_return_audit',{}).get('independent_sensor_permission_clock_audit'):raise ValueError('independently audited feedback return required')
        xs.append(return_outcome_features(obs,raw['composed_latent'][0],raw['pre_upper_yaw'][0],row,plan['config']['residual_interface'],row['original_user_alpha']) if rule else outcome_features(obs,raw['composed_latent'][0],raw['pre_upper_yaw'][0],row,plan['config']['residual_interface']))
        with np.load(row['diagnostic_task_file']) as z:cross=np.asarray(z['original_cross_track_m'])
        survived=row['endpoint']['physical_outcome']=='survived';goal=survived and row['endpoint']['goal_arrived']
        first=row['endpoint']['first_arrival_s'];stop=min(n,round((first-row['source_task_start_s'])/.005)) if goal else n
        if stop<=0 or len(cross)!=n:raise ValueError('aligned original task cost window required')
        path_rmse=float(np.sqrt(np.mean(cross[:stop]**2)));peak=row['full_upright_max_abs_speed_m_s']
        if peak is None or not np.isfinite([path_rmse,peak]).all():raise ValueError('finite physical regression targets required')
        ys.append([float(survived),float(goal),np.log1p(path_rmse),np.log1p(peak)])
        # Do not reward early failure with an artificially small partial-window error.
        masks.append([float(survived),float(survived)])
        dev=partition.get('all_development',False) or (row['force_N']==partition.get('development_force_N') and row['preparation_s']==partition.get('development_preparation_s'))
        roles.append('development' if dev else 'training')
        metadata.append(dict(case_id=row['case_id'],group=row['route']+'/'+row['event'],route=row['route'],alpha=row['alpha'],force_N=row['force_N'],preparation_s=row['preparation_s'],source_task_start_s=row['source_task_start_s'],path_rmse_to_goal_or_window_m=path_rmse,full_upright_max_abs_speed_m_s=peak,no_fall=survived,task_completed=bool(goal),regression_targets_valid=survived,role=roles[-1]))
        if rule:metadata[-1].update(physical_case_id=row['case_id'],original_user_alpha=row['original_user_alpha'],label_origin='executed_feedback_return')
    if roles.count('training')!=partition['training_rows'] or roles.count('development')!=partition['development_rows']:
        raise ValueError('predeclared TRAIN/DEV counts differ')
    for group in {m['group'] for m in metadata}:
        members=[m for m in metadata if m['group']==group]
        expected=sorted(declared_groups[group]) if declared_groups is not None else plan['protocol']['alphas']
        if len({m['role'] for m in members})!=1 or sorted(m['alpha'] for m in members)!=expected:
            raise ValueError('source/challenge alpha siblings cannot split across data roles')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'dataset.npz',features=np.asarray(xs),targets=np.asarray(ys,np.float32),regression_mask=np.asarray(masks,np.float32),roles=np.asarray(roles))
    manifest=dict(schema=SCHEMA,source_run=str(run),source_plan_sha256=inputs[str(run/'plan.json')],source_model=plan['models']['H75'],feature_dimension=42,state_indices=list(STATE_INDICES),query_features=['candidate_alpha','candidate_speed_correction_m_s','candidate_steer_correction_div_0p2rad','hypothetical_body_force_x_div32N','hypothetical_body_force_y_div32N','preparation_div0p6s','duration_div0p4s','remaining_task_time_div10s'],targets=['no_fall_to_original_task10','survived_and_entered_original_goal','log1p_original_path_rmse_to_goal_or_window','log1p_peak_upright_abs_speed'],partition=partition,records=metadata,inputs_sha256=inputs,dataset_sha256=sha(output/'dataset.npz'),unchanged_alpha_replay='checked' if has_unchanged else 'not applicable: no unchanged-alpha zero-force branch; complete source-copy proof retained',limitations='Related-state TRAIN/DEV, deterministic branch labels. Inputs contain hypothetical queries, never actual future sensor values. Only declared original task<=10s is labeled; extra raw suffix is preserved separately. No independent TEST, probability calibration or continuous safety guarantee.')
    if partition_override is not None:manifest.update(parent_partition=parent_partition,release_note=release_note)
    if rule:
        manifest.update(schema=RETURN_SCHEMA,feature_dimension=43,continuation_policy=dict(kind='sensor_gated_return_v1',parameters={k:v for k,v in rule.items() if k!='user_alpha'},implementation_sha256=sha(run/'source/tolerance_coordination.py')))
        manifest['query_features']=manifest['query_features']+['original_user_alpha']
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def prepare_return_dataset(run,output,**kwargs):
    return prepare_dataset(run,output,policy_schema=RETURN_SCHEMA,**kwargs)


def merge_outcome_datasets(datasets,output,*,release_note):
    """New TRAIN-only refit stage; preserve every parent's original data roles."""
    if not release_note:raise ValueError('explicit release of known development labels required')
    output=Path(output).resolve();parents=[];records=[];parts=[];inputs={};seen=set();reference=None
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for directory in datasets:
        directory=Path(directory).resolve();file=directory/'manifest.json';meta=json.loads(file.read_text());datafile=directory/'dataset.npz'
        if meta['schema'] not in (SCHEMA,RETURN_SCHEMA) or sha(datafile)!=meta['dataset_sha256']:raise ValueError('parent dataset identity differs')
        for source,expected in meta['inputs_sha256'].items():
            if sha(source)!=expected:raise ValueError('parent evidence changed '+source)
        contract={k:meta[k] for k in ('schema','feature_dimension','state_indices','query_features','targets')}
        contract['middle_actor_sha256']=meta['source_model']['actor_sha256']
        physical_contract=meta.get('physical_contract_sha256')
        if physical_contract is None:
            source=Path(meta['source_run']);source_plan=json.loads((source/'plan.json').read_text());receipt=json.loads((source/'analysis/receipt.json').read_text())
            payload=dict(config=source_plan['config'],initial_source=source_plan['initial_source'],physics_identity=receipt['physics_identity'])
            physical_contract=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        contract['physical_contract_sha256']=physical_contract
        contract['continuation_policy']=meta.get('continuation_policy')
        if reference is None:reference=contract
        elif reference!=contract:raise ValueError('cannot merge different feature or frozen-middle identities')
        with np.load(datafile) as z:data=dict(z)
        if len(meta['records'])!=len(data['features']):raise ValueError('parent labels/records count differs')
        for i,row in enumerate(meta['records']):
            if row['case_id'] in seen:raise ValueError('duplicate physical case across datasets')
            seen.add(row['case_id']);records.append(dict(row,role='training',parent_role=str(data['roles'][i]),parent_dataset=str(directory)))
        parents.append(dict(dataset=str(directory),partition=meta['partition'],rows=len(data['features'])))
        parts.append(data);inputs.update({str(file):sha(file),str(datafile):sha(datafile)})
    if not parts:raise ValueError('nonempty parent datasets required')
    merged={key:np.concatenate([part[key] for part in parts]) for key in ('features','targets','regression_mask')}
    merged['roles']=np.full(len(records),'training');output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'dataset.npz',**merged)
    manifest=dict(meta,schema=reference['schema'],source_run=None,physical_contract_sha256=reference['physical_contract_sha256'],source_datasets=parents,records=records,
        partition=dict(training_rows=len(records),development_rows=0,unit='Explicit release of all known parent rows for this refit only'),
        release_note=release_note,inputs_sha256=inputs,dataset_sha256=sha(output/'dataset.npz'),
        limitations='Related states and already observed outcomes; all rows TRAIN. Parent DEV evidence remains unchanged. No independent validation or online control claim.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def failure_before_return_deadline(contact_s,start_s,minimum_preparation_s,margin_s=.05):
    return contact_s is not None and np.isfinite(contact_s) and contact_s<start_s+minimum_preparation_s-margin_s


def derive_return_training_dataset(held_dataset,actual_return_dataset,output):
    """Reuse identical programs and pre-return failure prefixes, never futures.

    A held trajectory is exactly the same alpha program when its original-user
    destination equals its candidate. A contact before any return is possible
    also remains a negative query for another destination. These are derived
    supervision queries sharing physical evidence, not new simulation episodes.
    """
    held_dataset=Path(held_dataset).resolve();actual_return_dataset=Path(actual_return_dataset).resolve();output=Path(output).resolve()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    held=json.loads((held_dataset/'manifest.json').read_text());actual=json.loads((actual_return_dataset/'manifest.json').read_text())
    if held['schema']!=SCHEMA or actual['schema']!=RETURN_SCHEMA or held['source_model']['actor_sha256']!=actual['source_model']['actor_sha256']:raise ValueError('explicit held42 and executed-return43 datasets with same Actor required')
    inputs={}
    for directory,meta in ((held_dataset,held),(actual_return_dataset,actual)):
        if sha(directory/'dataset.npz')!=meta['dataset_sha256']:raise ValueError('parent outcome dataset changed')
        for path,expected in meta['inputs_sha256'].items():
            if sha(path)!=expected:raise ValueError('parent evidence changed '+path)
        inputs.update({str(directory/'manifest.json'):sha(directory/'manifest.json'),str(directory/'dataset.npz'):sha(directory/'dataset.npz')})
    run=Path(actual['source_run']);plan=json.loads((run/'plan.json').read_text());rule=plan['protocol']['causal_preference_return'];policy=actual['continuation_policy']
    receipt=json.loads((run/'analysis/receipt.json').read_text());payload=dict(config=plan['config'],initial_source=plan['initial_source'],physics_identity=receipt['physics_identity'])
    contract=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if held['physical_contract_sha256']!=contract:raise ValueError('held/return physical contracts differ')
    if policy['parameters']!={k:v for k,v in rule.items() if k!='user_alpha'}:raise ValueError('return policy parameter identity differs')
    evidence={};visited=set()
    def collect(directory):
        directory=Path(directory)
        if str(directory) in visited:return
        visited.add(str(directory));meta=json.loads((directory/'manifest.json').read_text())
        if meta.get('source_run'):
            file=Path(meta['source_run'])/'analysis/results.json';expected=meta['inputs_sha256'][str(file)]
            if sha(file)!=expected:raise ValueError('failure-time evidence changed')
            inputs[str(file)]=expected
            evidence.update({r['case_id']:r for r in json.loads(file.read_text())})
        else:
            for parent in meta['source_datasets']:
                file=Path(parent['dataset'])/'manifest.json'
                if sha(file)!=meta['inputs_sha256'][str(file)]:raise ValueError('parent source mapping changed')
                collect(parent['dataset'])
    collect(held_dataset)
    with np.load(held_dataset/'dataset.npz') as z:old=dict(z)
    with np.load(actual_return_dataset/'dataset.npz') as z:executed=dict(z)
    xs=[];ys=[];masks=[];records=[]
    for i,row in enumerate(held['records']):
        physical=evidence[row['case_id']];candidate=float(row['alpha']);user=float(rule['user_alpha'])
        requests=[(candidate,'held_equivalent_user_equals_candidate')]
        if candidate!=user and old['targets'][i,0]==0. and failure_before_return_deadline(physical.get('contact_pre_tick_s'),row['source_task_start_s'],rule['minimum_preparation_s']):requests.append((user,'before_any_return_contact'))
        for destination,origin in requests:
            xs.append(np.r_[old['features'][i],destination]);ys.append(old['targets'][i]);masks.append(old['regression_mask'][i])
            records.append(dict(row,case_id=row['case_id']+f'/return_policy_user{destination:g}/{origin}',physical_case_id=row['case_id'],
                original_user_alpha=destination,source_original_user_alpha=physical.get('original_user_alpha',physical.get('user_alpha',candidate)),
                role='training',label_origin=origin,parent_dataset=str(held_dataset)))
    for i,row in enumerate(actual['records']):
        xs.append(executed['features'][i]);ys.append(executed['targets'][i]);masks.append(executed['regression_mask'][i]);records.append(dict(row,role='training',parent_dataset=str(actual_return_dataset)))
    if len({r['case_id'] for r in records})!=len(records):raise ValueError('duplicate return-policy query identity')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'dataset.npz',features=np.asarray(xs,np.float32),targets=np.asarray(ys,np.float32),regression_mask=np.asarray(masks,np.float32),roles=np.full(len(records),'training'))
    counts={origin:sum(r['label_origin']==origin for r in records) for origin in sorted({r['label_origin'] for r in records})}
    manifest=dict(actual,source_run=None,physical_contract_sha256=contract,source_held_dataset=str(held_dataset),source_return_dataset=str(actual_return_dataset),
        records=records,partition=dict(training_rows=len(records),development_rows=0,unit='Policy-conditioned supervision queries; shared physical cases are not independent episodes'),
        unique_physical_cases=len({r['physical_case_id'] for r in records}),label_origin_counts=counts,inputs_sha256=inputs,dataset_sha256=sha(output/'dataset.npz'),
        causal_derivation=dict(negative_prefix_margin_s=.05,minimum_preparation_s=rule['minimum_preparation_s'],identity_property='Destination equals preparation alpha: no alpha change at any phase',physical_reexecution=False),
        limitations='Known related physical cases and explicitly derived equivalent-action supervision. New feedback policy has only the executed return cases above. No independent TEST, calibrated probabilities or continuous alpha safety guarantee.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def predict_outcomes(export,features):
    """CPU NumPy inference from the exact frozen multi-head ensemble export."""
    x=np.asarray(features,np.float32)
    if x.ndim not in (1,2) or x.shape[-1]!=len(export['mean']) or x.shape[-1] not in (42,43) or not np.isfinite(x).all():raise ValueError('finite matching conditional42/43 feature vector/matrix required')
    x=np.clip((x-export['mean'])/export['scale'],-5.,5.)
    h=np.einsum('...i,sij->s...j',x,export['w0'])+export['b0'][:,None,:] if x.ndim==2 else np.einsum('i,sij->sj',x,export['w0'])+export['b0']
    h=np.where(h>=0,h,np.expm1(np.minimum(h,0.)))
    h=np.einsum('s...i,sij->s...j',h,export['w1'])+(export['b1'][:,None,:] if x.ndim==2 else export['b1'])
    h=np.where(h>=0,h,np.expm1(np.minimum(h,0.)))
    logits=np.einsum('s...i,sij->s...j',h,export['w2'])+(export['b2'][:,None,:] if x.ndim==2 else export['b2'])
    probabilities=1./(1.+np.exp(-np.clip(logits[...,:2],-60.,60.)))
    continuous=np.expm1(np.clip(logits[...,2:],0.,10.))
    return np.concatenate([probabilities,continuous],axis=-1).mean(axis=0)


def plan_alpha_intervals(export,observation,candidate_alphas,candidate_latents,source_yaw,task_time_s,hypotheses,interface,*,no_fall_threshold,task_threshold,max_path_error_m,interval_function=None,original_user_alpha=None):
    """Return risk/task/path-constrained candidate ranges without a future-event input.

    Hypotheses are the caller-declared disturbance family, not a feed of actual
    scheduled gusts. Predictions remain empirical and domain-limited; an empty
    set is preserved instead of inventing a safe fallback alpha.
    """
    if interval_function is None:
        from .tolerance_coordination import feasible_alpha_intervals
    else:feasible_alpha_intervals=interval_function
    grid=np.asarray(candidate_alphas,float);latents=np.asarray(candidate_latents,np.float32)
    if grid.ndim!=1 or latents.shape!=(len(grid),2) or not hypotheses:raise ValueError('candidate latents and nonempty hypothetical disturbance family required')
    features=[]
    for alpha,latent in zip(grid,latents):
        for hypothesis in hypotheses:
            case=dict(hypothesis,alpha=float(alpha),source_task_start_s=float(task_time_s))
            case['onset_s']=task_time_s+case['preparation_s'] if case['force_N'] else 0.
            if case['force_N'] and case['onset_s']+case['duration_s']>10.:raise ValueError('hypothetical event extends beyond supported task horizon')
            query=np.asarray(observation,np.float32).copy();query[336]=alpha
            features.append(return_outcome_features(query,latent,source_yaw,case,interface,original_user_alpha) if len(export['mean'])==43 else outcome_features(query,latent,source_yaw,case,interface))
    forecast=predict_outcomes(export,np.stack(features)).reshape(len(grid),len(hypotheses),4)
    aggregate=np.c_[forecast[:,:,0].min(axis=1),forecast[:,:,1].min(axis=1),forecast[:,:,2].max(axis=1),forecast[:,:,3].max(axis=1)]
    result=feasible_alpha_intervals(grid,aggregate[:,0],aggregate[:,1],aggregate[:,2],no_fall_threshold=no_fall_threshold,task_threshold=task_threshold,max_path_error_m=max_path_error_m)
    result.update(candidate_outcomes=aggregate.tolist(),hypothetical_disturbances=hypotheses,task_time_s=float(task_time_s),aggregation='Minimum predicted no-fall/task scores and maximum path/speed predictions across declared hypothetical queries; not certified bounds.',model_scope='Estimated ranges from a small related-domain predictor; no calibrated risk or continuous-interior guarantee. No actual future event schedule is read.')
    return result


class FrozenAlphaRangePlanner:
    """State-query adapter: frozen middle queries plus the small outcome network.

    Loads only model/config artifacts, never trajectories or future outcome
    labels. The caller supplies the current native observation and a declared
    hypothetical disturbance family. This class does not command actuators.
    """
    def __init__(self,directory):
        import pickle
        import jax
        import jax.numpy as j
        from .path_command_policy import PathCommandActor
        self.directory=Path(directory).resolve();self.manifest=json.loads((self.directory/'manifest.json').read_text())
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();middle=self.manifest['source_model']
        for file,expected in ((self.directory/'parameters.npz',self.manifest['parameters_sha256']),(Path(middle['actor']),middle['actor_sha256']),(Path(middle['training_manifest']),middle['training_manifest_sha256'])):
            if sha(file)!=expected:raise ValueError('range planner model identity changed')
        config=json.loads(Path(middle['training_manifest']).read_text())['config']
        if config['network']['actor_dim']!=351:raise ValueError('native351 state schema required')
        self.interface=config['residual_interface']
        with np.load(self.directory/'parameters.npz') as z:self.parameters=dict(z)
        expected_dim=43 if self.manifest.get('schema')==RETURN_SCHEMA else 42
        if len(self.parameters['mean'])!=expected_dim or (expected_dim==43 and self.manifest.get('continuation_policy',{}).get('kind')!='sensor_gated_return_v1'):raise ValueError('outcome feature/policy schema differs')
        with Path(middle['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
        actor=PathCommandActor(architecture=config['network']['architecture'])
        self.middle_query=jax.jit(lambda x:actor.apply(params,x))

    def query(self,observation,source_yaw,task_time_s,hypotheses,*,user_alpha=None,candidate_alphas=(0.,.125,.25,.375,.5,.625,.75,.875,1.),no_fall_threshold=.8,task_threshold=.5,max_path_error_m=4.):
        import jax.numpy as j
        started=time.monotonic();obs=np.asarray(observation,np.float32)
        if obs.shape!=(351,) or not np.isfinite(obs).all() or not 0<=obs[336]<=1:raise ValueError('current native351 observation with valid conditioning alpha required')
        user=float(obs[336]) if user_alpha is None else float(user_alpha)
        if not np.isfinite(user) or not 0<=user<=1:raise ValueError('finite original user alpha in[0,1] required')
        # Preserve an explicitly requested user preference even off the standard grid.
        grid=np.unique(np.r_[candidate_alphas,user]);queries=np.repeat(obs[None,:],len(grid),axis=0);queries[:,336]=grid
        latents=np.asarray(self.middle_query(j.asarray(queries)))
        result=plan_alpha_intervals(self.parameters,obs,grid,latents,source_yaw,task_time_s,hypotheses,self.interface,no_fall_threshold=no_fall_threshold,task_threshold=task_threshold,max_path_error_m=max_path_error_m,original_user_alpha=user)
        if self.manifest.get('schema')==RETURN_SCHEMA:result['continuation_policy']=self.manifest['continuation_policy']
        result.update(user_alpha=user,conditioning_alpha_before_query=float(obs[336]),middle_id=self.manifest['source_model']['registry_id'],outcome_parameters_sha256=self.manifest['parameters_sha256'],query_seconds=time.monotonic()-started,scope='Query API only; caller controls execution and restores preference. Related-state alpha interpolation validated, not universal control/safety certification.')
        return result


def release_development_for_refit(dataset,output):
    """Explicit new-stage release; original failed TRAIN/DEV experiment stays intact."""
    dataset=Path(dataset).resolve();output=Path(output).resolve();meta=json.loads((dataset/'manifest.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(dataset/'dataset.npz')!=meta['dataset_sha256']:raise ValueError('parent dataset changed')
    with np.load(dataset/'dataset.npz') as z:data=dict(z)
    count=len(data['features']);data['roles']=np.full(count,'training')
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/'dataset.npz',**data)
    meta.update(parent_dataset=str(dataset),parent_partition=meta['partition'],development_released=True,
        release_note='Prior related DEV failure is known and these labels are released for a new fit. No DEV claim for refit; freeze predictions before new physical interior-alpha probes.',
        partition=dict(training_rows=count,development_rows=0,unit='all prior complete source/challenge groups released together'),dataset_sha256=sha(output/'dataset.npz'))
    for row in meta['records']:row['role']='training'
    meta['inputs_sha256'].update({str(dataset/'manifest.json'):sha(dataset/'manifest.json'),str(dataset/'dataset.npz'):sha(dataset/'dataset.npz')})
    (output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');return meta


def freeze_query_predictions(protocol,training,output):
    """Predict new candidate alphas before collecting their physical labels."""
    import pickle
    import jax
    import jax.numpy as j
    from .native_goal_run import low_speed_lateral_load_inventory
    from .path_command_policy import PathCommandActor
    protocol=Path(protocol).resolve();training=Path(training).resolve();output=Path(output).resolve()
    cfg=json.loads(protocol.read_text());cases=low_speed_lateral_load_inventory(cfg)
    fitted=json.loads((training/'manifest.json').read_text());dataset=Path(fitted['dataset']);meta=json.loads((dataset/'manifest.json').read_text())
    root=Path(cfg.get('query_source_run') or meta['source_run']);parent=json.loads((root/'plan.json').read_text());rows=json.loads((root/'analysis/results.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('prospective query source changed')
    if sha(training/'parameters.npz')!=fitted['parameters_sha256'] or sha(parent['models']['H75']['actor'])!=parent['models']['H75']['actor_sha256'] or parent['models']['H75']['actor_sha256']!=fitted['source_model']['actor_sha256']:
        raise ValueError('frozen outcome model or middle actor changed')
    with np.load(training/'parameters.npz') as z:export=dict(z)
    with Path(parent['models']['H75']['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
    actor=PathCommandActor(architecture=parent['config']['network']['architecture'])
    source={};checks=[]
    for route in {c['route'] for c in cases}:
        context=next(c for c in cfg['route_contexts'] if c['route']==route)
        previous_context=next(c for c in parent['contexts'] if c['route']==route)
        if context.get('state_source')!=previous_context.get('state_source'):raise ValueError('prospective complete source identity differs')
        row=next(r for r in rows if r['route']==route)
        with np.load(row['raw_file']) as z:raw=dict(z)
        obs=raw['query_observation'][0];latent=np.asarray(actor.apply(params,j.asarray(obs)))
        np.testing.assert_allclose(latent,raw['composed_latent'][0],atol=3e-5,rtol=3e-5)
        source[route]=(obs,float(raw['pre_upper_yaw'][0]));checks.append(dict(route=route,source_run=str(root),source_plan_sha256=sha(root/'plan.json'),source_raw_sha256=sha(row['raw_file']),maximum_actor_latent_error=float(np.max(abs(latent-raw['composed_latent'][0])))))
    queries=[]
    for case in cases:
        obs=source[case['route']][0].copy();obs[336]=case['alpha'];queries.append(obs)
    latents=np.asarray(actor.apply(params,j.asarray(np.stack(queries))))
    features=np.stack([return_outcome_features(obs,z,source[c['route']][1],c,parent['config']['residual_interface'],c['original_user_alpha']) if fitted['schema']==RETURN_SCHEMA else outcome_features(obs,z,source[c['route']][1],c,parent['config']['residual_interface']) for obs,z,c in zip(queries,latents,cases)])
    prediction=predict_outcomes(export,features)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_suffix('.npz').exists():raise FileExistsError('frozen prospective prediction already exists')
    np.savez_compressed(output.with_suffix('.npz'),features=features,prediction=prediction,case_ids=np.asarray([c['case_id'] for c in cases]))
    import datetime
    result=dict(schema=fitted['schema'],recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=str(training),training_manifest_sha256=sha(training/'manifest.json'),parameters_sha256=fitted['parameters_sha256'],query_data=str(output.with_suffix('.npz')),query_data_sha256=sha(output.with_suffix('.npz')),cases=cases,actor_export_checks=checks,scope='Prospective predictions frozen before new physical query labels. Known source states/challenges, no new-scenario generalization claim.',interval_examples=dict(no_fall_thresholds=[.7,.8,.9],task_threshold=.5,path_rmse_limits_m=[.3,1.,2.,4.]))
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def audit_query_feature_alignment(frozen,observed,export):
    """Bound Actor roundoff without changing the pre-physics predictions.

    Candidate steer correction (feature36) is divided by .2rad; the extra
    allowance is at most4e-5rad. State/clock/force features retain old bounds.
    Probability/output drift and default feasibility must also remain stable.
    No observed outcome labels enter this numerical check.
    """
    frozen=np.asarray(frozen);observed=np.asarray(observed)
    if frozen.shape!=observed.shape or frozen.ndim!=2 or frozen.shape[1] not in (42,43):raise ValueError('matching conditional42/43 features required')
    error=abs(observed-frozen);bound=np.full(frozen.shape[1],5e-5);bound[36]=2e-4
    if not np.isfinite(error).all() or np.any(error>bound+3e-5*abs(frozen)):raise ValueError('observed source features exceed bounded Actor roundoff')
    expected=predict_outcomes(export,frozen);actual=predict_outcomes(export,observed);drift=abs(expected-actual)
    if np.any(drift[:,:2]>5e-3) or np.any(drift[:,2:]>1e-2):raise ValueError('Actor roundoff changes outcome predictions materially')
    if not np.array_equal(expected[:,:2]>=.5,actual[:,:2]>=.5):raise ValueError('Actor roundoff changes binary classification')
    mask=lambda p:(p[:,0]>=.8)&(p[:,1]>=.5)&(p[:,2]<=4.)
    if not np.array_equal(mask(expected),mask(actual)):raise ValueError('Actor roundoff changes default feasibility')
    return dict(maximum_feature_error=float(error.max()),candidate_steer_error_rad=float(error[:,36].max()*.2),
        maximum_probability_drift=float(drift[:,:2].max()),maximum_regression_drift=float(drift[:,2:].max()),
        default_feasibility_unchanged=True,binary_classification_unchanged=True,frozen_predictions_preserved=True,
        numerical_protocol='Post-sampling Actor batching amendment: only feature36 up to2e-4 normalized, <=.005 probability / .01 regression drift; all state features retain original bounds, all classification/qualification decisions must be unchanged. No outcome labels used.')


def evaluate_frozen_queries(frozen_manifest,dataset,output):
    """Score only fresh physical probes; old training points only annotate ranges."""
    from .tolerance_coordination import feasible_alpha_intervals
    frozen_manifest=Path(frozen_manifest).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    frozen=json.loads(frozen_manifest.read_text());meta=json.loads((dataset/'manifest.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    training=Path(frozen['training']);fitted=json.loads((training/'manifest.json').read_text())
    if meta['schema']!=fitted['schema'] or sha(dataset/'dataset.npz')!=meta['dataset_sha256']:raise ValueError('new probe dataset identity changed')
    if meta['schema']==RETURN_SCHEMA and meta.get('continuation_policy')!=fitted.get('continuation_policy'):raise ValueError('observed continuation differs from forecast policy')
    if sha(frozen['query_data'])!=frozen['query_data_sha256'] or sha(training/'manifest.json')!=frozen['training_manifest_sha256'] or sha(training/'parameters.npz')!=frozen['parameters_sha256']:
        raise ValueError('prospective model/predictions changed after freeze')
    with np.load(frozen['query_data']) as z:query=dict(z)
    with np.load(dataset/'dataset.npz') as z:data=dict(z)
    if [r['case_id'] for r in meta['records']]!=query['case_ids'].tolist() or not np.all(data['roles']=='development'):
        raise ValueError('all new physical probe identities/roles must match frozen queries')
    with np.load(training/'parameters.npz') as z:export=dict(z)
    alignment=audit_query_feature_alignment(query['features'],data['features'],export)
    metric=outcome_metrics(data['targets'],query['prediction'],data['regression_mask'])
    default=(query['prediction'][:,0]>=.8)&(query['prediction'][:,1]>=.5)&(query['prediction'][:,2]<=4.)
    actual=(data['targets'][:,0]==1)&(data['targets'][:,1]==1)&(np.expm1(data['targets'][:,2])<=4.)
    disturbed=np.asarray([r['force_N']!=0 for r in meta['records']])
    metric.update(qualified_new_candidates=int(default.sum()),qualified_new_disturbed_candidates=int((default&disturbed).sum()),false_feasible_new_cases=[r['case_id'] for i,r in enumerate(meta['records']) if default[i] and not actual[i]],false_no_fall_qualified_cases=[r['case_id'] for i,r in enumerate(meta['records']) if default[i] and data['targets'][i,0]==0],scope=f'New{len(meta["records"])} prospective probes only; no training rows included in these metrics. Example thresholds q.8/task.5/pathRMSE4m.')
    metric['qualified_new_speed_violations']=[r['case_id'] for i,r in enumerate(meta['records']) if default[i] and r['full_upright_max_abs_speed_m_s']>2.1]
    parent=Path(fitted['dataset']);parent_meta=json.loads((parent/'manifest.json').read_text())
    if sha(parent/'manifest.json')!=fitted['dataset_manifest_sha256']:raise ValueError('training reference metadata changed')
    with np.load(parent/'dataset.npz') as z:reference=dict(z)
    with np.load(training/'parameters.npz') as z:export=dict(z)
    reference_prediction=predict_outcomes(export,reference['features']);wanted={r['group'] for r in meta['records']};combined=[]
    users={r['group']:r.get('original_user_alpha') for r in meta['records']}
    for records,truth,prediction,role in ((parent_meta['records'],reference['targets'],reference_prediction,'training_reference'),(meta['records'],data['targets'],query['prediction'],'fresh_alpha_probe')):
        combined.extend(dict(record=r,target=t.tolist(),prediction=p.tolist(),role=role) for r,t,p in zip(records,truth,prediction) if r['group'] in wanted and (meta['schema']!=RETURN_SCHEMA or r['original_user_alpha']==users[r['group']]))
    groups=[]
    examples=frozen['interval_examples']
    for group in sorted(wanted):
        members=sorted([r for r in combined if r['record']['group']==group],key=lambda r:r['record']['alpha']);grid=[r['record']['alpha'] for r in members]
        if len(grid)!=len(set(grid)):raise ValueError('duplicate checked alpha points in same user query group')
        prediction=np.asarray([r['prediction'] for r in members]);truth=np.asarray([r['target'] for r in members]);ranges=[]
        for risk in examples['no_fall_thresholds']:
            for limit in examples['path_rmse_limits_m']:
                plan=feasible_alpha_intervals(grid,prediction[:,0],prediction[:,1],prediction[:,2],no_fall_threshold=risk,task_threshold=examples['task_threshold'],max_path_error_m=limit)
                observed=(truth[:,0]==1)&(truth[:,1]==1)&(np.expm1(truth[:,2])<=limit)
                plan.update(observed_qualified_mask=observed.tolist(),false_feasible_alpha=[a for a,p,t in zip(grid,plan['feasible_mask'],observed) if p and not t]);ranges.append(plan)
        groups.append(dict(group=group,candidates=members,plans=ranges,original_five_reference_complete={0.,.25,.5,.75,1.}.issubset(set(grid))))
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema=fitted['schema'],feature_alignment_audit=alignment,prospective_manifest=str(frozen_manifest),prospective_manifest_sha256=sha(frozen_manifest),new_dataset=str(dataset),new_dataset_manifest_sha256=sha(dataset/'manifest.json'),metrics=metric,groups=groups,adoption='pending_closed_loop_review',limits='Model scores are uncalibrated, known-state query validation only. Range tables show only checked points and explicitly mark incomplete original-five references; unqueried candidates and continuous interiors remain unverified. No road/safety guarantee; empty sets preserved.')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');(output/'metrics.json').write_text(json.dumps(metric,indent=2)+'\n')
    physical_run=Path(meta['source_run']);first=next(r for r in meta['records'] if r['force_N']!=0)
    event=first['group'].split('/')[-1];xy=physical_run/'analysis'/(first['route']+'_'+event+'_fixed_candidate_xy.png')
    lines=['# Prospective alpha-interior prediction check','',f'![New physical alpha trajectories]({xy})','',f'All actual XY / reward / component panels: [{physical_run.name}]({physical_run}/analysis/README.md).','','```json',json.dumps(metric,indent=2),'```','','Fresh-probe metrics above exclude all training rows. Range references below mix the original five training points with fresh queried points; they are not all held-out cases.','', '|Source / challenge|Estimated intervals at q>=.8, task>=.5, pathRMSE<=4m|False feasible sampled alpha|','|---|---|---|']
    for g in groups:
        p=next(p for p in g['plans'] if p['no_fall_threshold']==.8 and p['max_path_error_m']==4.)
        lines.append(f"|{g['group']}|{p['estimated_intervals']}|{p['false_feasible_alpha']}|")
    lines+=['','Tight path limits can yield an empty set. This API predicts approximate candidate ranges and never certifies their unsampled interiors. Hypothetical force/preparation inputs are not a stream of actual future gust times.','',result['limits']]
    (output/'README.md').write_text('\n'.join(lines)+'\n');return result


def outcome_metrics(target,prediction,mask):
    truth=np.asarray(target);p=np.asarray(prediction);valid=np.asarray(mask)[:,0]>0
    return dict(rows=len(truth),no_fall_brier=float(np.mean((p[:,0]-truth[:,0])**2)),
        task_brier=float(np.mean((p[:,1]-truth[:,1])**2)),
        no_fall_accuracy=float(np.mean((p[:,0]>=.5)==truth[:,0])),task_accuracy=float(np.mean((p[:,1]>=.5)==truth[:,1])),
        path_rmse_prediction_error_m=float(np.sqrt(np.mean((p[valid,2]-np.expm1(truth[valid,2]))**2))) if valid.any() else None,
        peak_speed_prediction_error_m_s=float(np.sqrt(np.mean((p[valid,3]-np.expm1(truth[valid,3]))**2))) if valid.any() else None)


def fit_outcomes(dataset,output,config,*,on_training_complete=None):
    """Fixed-last supervised CPU pilot; does not update any control network."""
    import torch
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    dataset=Path(dataset).resolve();output=Path(output).resolve();cfg=json.loads(Path(config).read_text())
    feature_dim=cfg['architecture']['input_dim'];schema=SCHEMA if feature_dim==42 else RETURN_SCHEMA
    if (feature_dim not in (42,43) or cfg['architecture']['hidden']!=[16,16] or cfg['architecture']['outputs']!=4
        or cfg['training']['device']!='cpu' or cfg['training']['selection']!='fixed_last'
        or cfg['budget']['physics_ticks']!=0
        or cfg['budget']['optimizer_updates']!=len(cfg['training']['seeds'])*cfg['training']['updates_per_seed']):
        raise ValueError('declared small multi-output CPU fit contract differs')
    meta=json.loads((dataset/'manifest.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if meta['schema']!=schema or meta['feature_dimension']!=feature_dim or meta['dataset_sha256']!=sha(dataset/'dataset.npz'):raise ValueError('prepared dataset identity changed')
    for p,h in meta['inputs_sha256'].items():
        if sha(p)!=h:raise ValueError('prepared source changed '+p)
    with np.load(dataset/'dataset.npz') as z:data=dict(z)
    train=data['roles']=='training';dev=data['roles']=='development'
    if not train.any():raise ValueError('nonempty training partition required')
    x=data['features'];y=data['targets'];mask=data['regression_mask']
    mean=x[train].mean(axis=0);scale=np.maximum(x[train].std(axis=0),.05)
    normalized=np.clip((x-mean)/scale,-5.,5.).astype(np.float32)
    output.mkdir(parents=True,exist_ok=False);(output/'checkpoints').mkdir();started=time.monotonic()
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    status=dict(state='running',stage='training',completed_updates=0,declared_updates=cfg['budget']['optimizer_updates'],physics_ticks=0)
    def save_status():
        tmp=output/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(output/'status.json')
    save_status();torch.set_num_threads(cfg['training'].get('cpu_threads',1));tx=torch.tensor(normalized);ty=torch.tensor(y);tm=torch.tensor(mask)
    weights=torch.tensor(cfg['training']['head_loss_weights']);models=[];history=[]
    for seed in cfg['training']['seeds']:
        torch.manual_seed(seed);model=nn.Sequential(nn.Linear(feature_dim,16),nn.ELU(),nn.Linear(16,16),nn.ELU(),nn.Linear(16,4))
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
        writer=SummaryWriter(str(output/'tensorboard'/f'seed{seed}'))
        for step in range(1,cfg['training']['updates_per_seed']+1):
            optimizer.zero_grad();logits=model(tx[train]);binary=nn.functional.binary_cross_entropy_with_logits(logits[:,:2],ty[train,:2],reduction='none')
            regression=nn.functional.smooth_l1_loss(logits[:,2:],ty[train,2:],reduction='none')
            loss=(binary*weights[:2]).sum(dim=1).mean()+((regression*tm[train])*weights[2:]).sum()/torch.clamp(tm[train].sum(),min=1.)
            if not torch.isfinite(loss):raise ValueError('nonfinite supervised loss; stop without retry')
            loss.backward();optimizer.step();status['completed_updates']+=1
            if step==1 or step%cfg['training']['log_every_updates']==0:
                rec=dict(seed=seed,step=step,loss=float(loss.detach()))
                if dev.any():
                    with torch.no_grad():d=model(tx[dev]);prob=torch.sigmoid(d[:,:2]).numpy()
                    rec.update(no_fall_dev_brier=float(np.mean((prob[:,0]-y[dev,0])**2)),task_dev_brier=float(np.mean((prob[:,1]-y[dev,1])**2)))
                history.append(rec)
                for key,value in rec.items():
                    if key not in ('seed','step'):writer.add_scalar(key,value,step)
                writer.flush();save_status()
        torch.save(dict(state_dict=model.state_dict(),seed=seed,schema=schema,mean=mean,scale=scale,dataset_sha256=meta['dataset_sha256']),output/'checkpoints'/f'seed{seed}.pt')
        writer.close();models.append(model)
    status.update(stage='training_complete',state='completed',training_seconds=time.monotonic()-started);save_status()
    if on_training_complete:
        on_training_complete(output);status['stage_notified']=True;save_status()
    export=dict(mean=mean,scale=scale)
    for layer_index,model_index in enumerate((0,2,4)):
        export[f'w{layer_index}']=np.stack([m[model_index].weight.detach().numpy().T for m in models]);export[f'b{layer_index}']=np.stack([m[model_index].bias.detach().numpy() for m in models])
    prediction=predict_outcomes(export,x)
    with torch.no_grad():
        logits=torch.stack([m(tx) for m in models]).numpy();reference=np.concatenate([1./(1.+np.exp(-np.clip(logits[...,:2],-60,60))),np.expm1(np.clip(logits[...,2:],0,10))],axis=-1).mean(axis=0)
    np.testing.assert_allclose(prediction,reference,atol=3e-5,rtol=3e-5,err_msg='NumPy export inference differs from training model')
    np.savez_compressed(output/'parameters.npz',**export);np.savez_compressed(output/'predictions.npz',prediction=prediction,targets=y,roles=data['roles'])
    metrics=dict(training=outcome_metrics(y[train],prediction[train],mask[train]),development=outcome_metrics(y[dev],prediction[dev],mask[dev]) if dev.any() else None)
    result=dict(schema=schema,dataset=str(dataset),dataset_manifest_sha256=sha(dataset/'manifest.json'),config=cfg,source_model=meta['source_model'],metrics=metrics,selection='fixed last per seed; DEV curves diagnostic, not checkpoint selection',history=history,parameters_sha256=sha(output/'parameters.npz'),adoption='pending',limits='Related-state pilot; uncalibrated probabilities, no independent TEST or continuous alpha guarantee. Path/speed regression trained only on full-primary-window survival.')
    if schema==RETURN_SCHEMA:result['continuation_policy']=meta['continuation_policy']
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');(output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for seed in cfg['training']['seeds']:
        records=[r for r in history if r['seed']==seed]
        for ax,key in zip(axes,['loss','no_fall_dev_brier','task_dev_brier']):
            if key not in records[0]:
                ax.text(.5,.5,'No DEV during fit; fresh-alpha evaluation pending',ha='center',va='center',wrap=True,transform=ax.transAxes);continue
            ax.plot([r['step'] for r in records],[r[key] for r in records],marker='.',label=f'seed{seed}')
            ax.set(xlabel='Supervised updates',ylabel=key);ax.grid(alpha=.2);ax.legend()
    fig.suptitle(f'Conditional outcome MLP / TRAIN{int(train.sum())} DEV{int(dev.sum())} / fixed last; DEV is related, not independent TEST')
    fig.savefig(output/'training_progress.png',dpi=140);plt.close(fig)
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    result['implementation_sha256']=sha(output/'source.py');(output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    status.update(stage='pipeline_complete',export_numpy_verified=True,total_seconds=time.monotonic()-started);save_status()
    return result
