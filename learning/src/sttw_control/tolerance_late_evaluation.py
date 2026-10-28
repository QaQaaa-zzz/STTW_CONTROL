"""Fixed task6 DEV extrapolation assessment; no fitting or controller selection."""
from pathlib import Path
import json
import numpy as np
from . import tolerance_coordination as coordination


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_inventory(inventory,records):
    user_late=len(inventory)==192;count=192 if user_late else 48;group_count=16 if user_late else 4
    if len(inventory)!=count or len(records)!=count:raise ValueError('exact declared late48/192 inventory required')
    byid={c['case_id']:c for c in inventory}
    if len(byid)!=count or {r['case_id'] for r in records}!=set(byid):raise ValueError('late case identity mismatch')
    groups={}
    for c in inventory:
        if c.get('split')!=('late_user_development' if user_late else 'late_state_development'):raise ValueError('late DEV split required')
        if c.get('bank_index') not in ((4,6) if user_late else (2,5)):raise ValueError('late banks2/5 or user banks4/6 only')
        if user_late and (c.get('source_user_alpha') not in (0.,.5) or c.get('alpha')!=c['source_user_alpha']):raise ValueError('late source user identity mismatch')
        if (c['candidate_alpha'] not in (0.,.5,1.) or c['preparation_s'] not in (.2,.6) or c['force_N'] not in (-64.,64.) or
            not np.isclose(c['duration_s'],.1) or not np.isclose(c['onset_s'],6.+c['preparation_s']) or
            not np.isclose(c['primary_window_start_s'],6.) or not np.isclose(c['primary_window_end_s'],c['onset_s']+.1+3.)):
            raise ValueError('frozen task6 hypothetical condition mismatch')
        groups.setdefault(c['source_group_id'],[]).append(c)
    expected={(a,p,f) for a in (0.,.5,1.) for p in (.2,.6) for f in (-64.,64.)}
    if len(groups)!=group_count or any(len(c)!=12 or {(r['candidate_alpha'],r['preparation_s'],r['force_N']) for r in c}!=expected for c in groups.values()):raise ValueError('four complete or16 user late source groups required')
    for r in records:
        c=byid[r['case_id']]
        if any(r[k]!=c[k] for k in ('source_group_id','bank_index','candidate_alpha','preparation_s','force_N')+ (('source_user_alpha',) if user_late else ())):raise ValueError('late analysis condition changed')
    return byid


def grouped_metrics(y,p,meta,metric_fn):
    result={'overall':metric_fn(y,p,meta),'per_bank':{},'per_source':{}}
    for key,field in [('per_bank','bank_index'),('per_source','source_group_id')]:
        for value in sorted({m[field] for m in meta}):
            ids=[i for i,m in enumerate(meta) if m[field]==value]
            result[key][str(value)]=metric_fn(y[ids],p[ids],[meta[i] for i in ids])
    if meta and all('source_user_alpha' in m for m in meta):
        result['per_source_user_alpha']={}
        for value in sorted({m['source_user_alpha'] for m in meta}):
            ids=[i for i,m in enumerate(meta) if m['source_user_alpha']==value];result['per_source_user_alpha'][str(value)]=metric_fn(y[ids],p[ids],[meta[i] for i in ids])
    return result


def rejection_report(y,p,meta):
    user=np.array([m['alpha']==m.get('source_user_alpha',1.) for m in meta]);out={}
    for threshold in (.8,.85):
        rejected=p<threshold
        out[str(threshold)]=dict(user_rows=int(user.sum()),rejected_rows=int(np.sum(user&rejected)),survived_but_rejected_rows=int(np.sum(user&rejected&(y==1))),fall_but_not_rejected_rows=int(np.sum(user&~rejected&(y==0))))
    source={}
    for group in sorted({m['source_group_id'] for m in meta}):
        ids=[i for i,m in enumerate(meta) if m['source_group_id']==group and m['alpha']==m.get('source_user_alpha',1.)]
        if len(ids)!=4:
            source[group]=dict(resolved_user_rows=len(ids),four_hypothesis_assessment='incomplete; invalid labels excluded')
            continue
        qmin=float(np.min(p[ids]));source[group]=dict(resolved_user_rows=4,qmin=qmin,reject_at_08=qmin<.8,reject_at_085=qmin<.85,actual_no_fall=int(y[ids].sum()),actual_all_four_no_fall=bool(np.all(y[ids]==1)))
    out['four_hypothesis_user_gate']=source
    return out


def evaluate_late(root,q_export,output):
    """Authenticate frozen q before labels; evaluate once into a fresh directory."""
    import torch
    root=Path(root).resolve();q_export=Path(q_export).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh late evaluation output required')
    params=coordination.load_export(q_export) # Verifies all six copied weights/source BEFORE labels.
    frozen=coordination.file_module(q_export/'frozen_predictor_source.py','_late_frozen_q')
    source=json.loads((q_export/'ranking_source_manifest.json').read_text())
    cfg=json.loads((q_export/'ranking_config.json').read_text())
    plan_path=root/'plan.json';seal=json.loads((root/'seal.json').read_text())
    if coordination.sha(plan_path)!=seal['plan_sha256']:raise ValueError('late source plan changed')
    plan=json.loads(plan_path.read_text())
    validate_plan(plan);user_late=plan.get('mode')=='h75_user_late_state192';count=192 if user_late else 48
    export_receipt=json.loads((q_export/'receipt.json').read_text());arm_order=export_receipt['arm_order']
    if user_late and (arm_order!=['temporal_rank_selected','temporal_rank_last'] or export_receipt.get('source_manifest_kind')!='temporal_mix_selected_and_last'):raise ValueError('frozen temporal last slot1 required')
    if not any(expected==coordination.sha(q_export/'receipt.json') for expected in plan['input_sha256'].values()):raise ValueError('q export receipt not bound by sealed late plan')
    source_meta,source_features,source_yaw=authenticate_late_sources(root)
    inventory=json.loads((root/'inventory.json').read_text())
    analysis=root/'analysis_late';records=json.loads((analysis/'results.json').read_text())
    byid=validate_inventory(inventory,records)
    with np.load(root/'evaluation/source_features.npz',allow_pickle=False) as z:sources=z['native_observation' if user_late else 'native_observation_user1'].copy()
    if sources.shape!=(count,351):raise ValueError('late native351 source features required')
    index={c['case_id']:i for i,c in enumerate(inventory)}
    inputs={str(p):coordination.sha(p) for p in [plan_path,root/'seal.json',root/'inventory.json',analysis/'results.json',root/'evaluation/source_features.npz',root/'sources/seal.json',root/'sources/records.json',root/'sources/source_features.npz',q_export/'receipt.json']}
    features=[];labels=[];meta=[];excluded=[];source_groups={};candidate_latents={}
    for record in records:
        case=byid[record['case_id']];lane=index[record['case_id']];raw_path=Path(record['raw_file'])
        si=case['source_index'];src=source_meta[si]
        if any(case[k]!=src[k] for k in ('source_group_id','bank_index','split')+(('source_user_alpha',) if user_late else ())):raise ValueError('late source metadata mismatch')
        np.testing.assert_array_equal(sources[lane],source_features[si])
        if coordination.sha(raw_path)!=record['raw_sha256']:raise ValueError('late raw changed')
        inputs[str(raw_path)]=record['raw_sha256']
        with np.load(raw_path,allow_pickle=False) as raw:
            if not raw['active_tick'][0] or abs(float(raw['time'][0])-6.)>1e-6:raise ValueError('task6 first pre-force query required')
            query=raw['query_observation'][0];latent=raw['composed_latent'][0];yaw=float(raw['pre_upper_yaw'][0])
        if abs(np.arctan2(np.sin(yaw-source_yaw[si]),np.cos(yaw-source_yaw[si])))>1e-6:raise ValueError('late source pre-query yaw mismatch')
        if user_late:x=late_feature_row(sources[lane],query,latent,yaw,case,plan['config']['residual_interface'],cfg,frozen)
        else:x,_=frozen.feature_row(sources[lane],query,latent,yaw,case,plan['config']['residual_interface'],cfg)
        group=case['source_group_id']
        if group in source_groups:np.testing.assert_array_equal(sources[lane],source_groups[group])
        source_groups[group]=sources[lane]
        key=(group,case['candidate_alpha'])
        if key in candidate_latents:np.testing.assert_allclose(latent,candidate_latents[key],atol=1e-6,rtol=1e-6)
        candidate_latents[key]=latent
        label=frozen.resolved_label(record['primary'])
        if label is None:
            excluded.append(dict(case_id=case['case_id'],reason=record['primary']['label_status']));continue
        features.append(x);labels.append(label);meta.append(dict(case_id=case['case_id'],source_group_id=group,bank_index=case['bank_index'],alpha=case['candidate_alpha'],preparation_s=case['preparation_s'],force_N=case['force_N'],**(dict(source_user_alpha=case['source_user_alpha'],full_H10=record['horizons']['10']) if user_late else {})))
    if not labels:raise ValueError('no resolved late labels')
    features=np.asarray(features,np.float32);labels=np.asarray(labels,np.float32)
    probabilities={};metrics={};rejections={}
    evaluated_arms=['temporal_rank_last'] if user_late else ['bce','bce_rank']
    for arm in evaluated_arms:
        for seed in (0,1,2):
            model=frozen.make_model(41);checkpoint=torch.load(q_export/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict'])
            probabilities[f'{arm}/seed{seed}']=frozen.predict(model,features)
        probabilities[f'{arm}/ensemble']=np.mean([probabilities[f'{arm}/seed{s}'] for s in (0,1,2)],axis=0)
    if not user_late:probabilities['original_training_prior']=np.full(len(labels),57/72)
    for name,p in probabilities.items():
        metrics[name]=grouped_metrics(labels,p,meta,frozen.metrics)
        rejections[name]=rejection_report(labels,p,meta)
    # Existing highest-precision JAX export must agree with the local Torch ensembles.
    jax_probabilities=np.asarray(coordination.infer_probabilities(params,features));errors={}
    for arm in evaluated_arms:
        ai=arm_order.index(arm)
        error=float(np.max(np.abs(jax_probabilities[ai]-probabilities[f'{arm}/ensemble'])));errors[arm]=error
        if not np.isfinite(error) or error>2e-6:raise ValueError(f'late ensemble Torch/JAX error {error}')
    output.mkdir(parents=True)
    write_json(output/'metrics.json',metrics);write_json(output/'user_rejection.json',rejections)
    if user_late:write_json(output/'full_H10_outcomes.json',[dict(case_id=r['case_id'],source_group_id=r['source_group_id'],source_user_alpha=r['source_user_alpha'],candidate_alpha=r['candidate_alpha'],primary=r['primary'],full_H10=r['horizons']['10']) for r in records])
    write_json(output/'predictions.json',[dict(m,label=int(y),probabilities={k:float(p[i]) for k,p in probabilities.items()}) for i,(m,y) in enumerate(zip(meta,labels))])
    np.savez(output/'features_predictions.npz',features=features,labels=labels,jax_ensemble_probabilities=jax_probabilities)
    write_json(output/'receipt.json',dict(inputs_sha256=inputs,evaluator_sha256=coordination.sha(__file__),frozen_predictor_source_sha256=source['implementation_sha256'],resolved_rows=len(labels),excluded_invalid=excluded,ensemble_probability_errors=errors,primary_model='temporal_rank_last slot1' if user_late else 'legacy BCE/rank',source_users=[0.,.5] if user_late else [1.],resolved_label_classes=np.unique(labels).astype(int).tolist(),candidate_ranking_identifiable=bool(metrics[evaluated_arms[0]+'/ensemble']['overall']['discordant_pair_ranking']['pairs']>0),role='Frozen q applied to declared complete task6 states; no fitting, selection or threshold tuning',source_task_s=6.,new_optimizer_updates=0,new_physics_steps=0,thresholds=[.8,.85]))
    write_json(output/'seal.json',{p.name:coordination.sha(p) for p in output.iterdir() if p.is_file()})
    return metrics


def validate_plan(plan):
    user_late=plan.get('mode')=='h75_user_late_state192'
    if plan.get('mode') not in ('h75_late_state_tolerance48','h75_user_late_state192'):raise ValueError('explicit task6 late48 plan required')
    protocol=plan['protocol']
    fixed=dict(expected_cases=48,source_task_s=6.,horizon_s=10.,new_horizon_s=4.,maximum_prefix_lower_ticks=0,maximum_lower_ticks=38400,maximum_total_lower_ticks=38400,optimizer_updates=0,bank_indices=[2,5],candidate_alphas=[0.,.5,1.],preparation_s=[.2,.6],forces_N=[-64.,64.],duration_s=.1,primary_after_pulse_end_s=3.)
    if user_late:fixed.update(expected_cases=192,maximum_lower_ticks=153600,maximum_total_lower_ticks=153600,bank_indices=[4,6])
    if any(protocol.get(k)!=v for k,v in fixed.items()):raise ValueError('frozen late48 protocol mismatch')


def authenticate_late_sources(root):
    root=Path(root);seal=json.loads((root/'sources/seal.json').read_text())
    if seal['plan_sha256']!=coordination.sha(root/'plan.json'):raise ValueError('late source seal plan changed')
    required=[root/'sources/records.json',root/'sources/source_features.npz',root/'sources/complete_carry.pkl',root/'sources/executed_paths.pkl',root/'sources/host_state.npz']
    if any(str(p) not in seal['files'] for p in required):raise ValueError('late source seal missing required artifacts')
    for file,expected in seal['files'].items():
        if coordination.sha(file)!=expected:raise ValueError('late complete source artifact changed')
    records=json.loads((root/'sources/records.json').read_text());plan=json.loads((root/'plan.json').read_text());user_late=plan.get('mode')=='h75_user_late_state192';count=16 if user_late else 4
    if (len(records)!=count or {r['source_index'] for r in records}!=set(range(count)) or any(r.get('source_task_tick')!=1200 or r.get('source_physical_tick')!=1900 or r.get('split')!=('late_user_development' if user_late else 'late_state_development') for r in records)):raise ValueError('four task6/physical1900 sources required')
    with np.load(root/'sources/source_features.npz',allow_pickle=False) as z:
        features=z['native_observation' if user_late else 'native_observation_user1'].copy();yaw=z['source_world_yaw'].copy()
    if features.shape!=(count,351) or yaw.shape!=(count,) or not np.isfinite(features).all() or not np.isfinite(yaw).all():raise ValueError('four finite source features/yaw required')
    if user_late:
        users=np.asarray([r['source_user_alpha'] for r in records]);np.testing.assert_allclose(features[:,336],users,atol=1e-6,rtol=0)
        if set(users)!={0.,.5}:raise ValueError('both declared source users required')
    return {r['source_index']:r for r in records},features,yaw


def late_feature_row(source,query,latent,yaw,case,interface,cfg,frozen):
    """Frozen41 features for actual source user; feature336 remains excluded."""
    import math
    source=np.asarray(source,np.float32);query=np.asarray(query,np.float32)
    if source.shape!=(351,) or query.shape!=(351,) or not np.isfinite(source).all() or not np.isfinite(query).all():raise ValueError('finite native351 source/query required')
    np.testing.assert_allclose(np.delete(source,336),np.delete(query,336),atol=1e-6,rtol=1e-6)
    if abs(float(source[336])-case['source_user_alpha'])>1e-6 or abs(float(query[336])-case['candidate_alpha'])>1e-6:raise ValueError('late source/query alpha mismatch')
    indices=cfg['features']['full_indices']
    if indices!=list(range(300,320))+list(range(337,351)):raise ValueError('frozen41 feature identity differs')
    target=frozen.native_target(latent,interface)
    shared=np.array([case['candidate_alpha'],target[0],target[1]/.2,case['force_N']*math.sin(yaw)/64.,case['force_N']*math.cos(yaw)/64.,case['duration_s']/.1,case['preparation_s']/.6],np.float32)
    features=np.concatenate([source[indices],shared])
    if features.shape!=(41,) or not np.isfinite(features).all():raise ValueError('finite frozen41 features required')
    return features
