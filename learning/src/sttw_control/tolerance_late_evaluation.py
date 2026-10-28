"""Fixed task6 DEV extrapolation assessment; no fitting or controller selection."""
from pathlib import Path
import json
import numpy as np
from . import tolerance_coordination as coordination


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_inventory(inventory,records):
    if len(inventory)!=48 or len(records)!=48:raise ValueError('exact late48 inventory required')
    byid={c['case_id']:c for c in inventory}
    if len(byid)!=48 or {r['case_id'] for r in records}!=set(byid):raise ValueError('late case identity mismatch')
    groups={}
    for c in inventory:
        if c.get('split')!='late_state_development':raise ValueError('late DEV split required')
        if c.get('bank_index') not in (2,5):raise ValueError('late banks2/5 only')
        if (c['candidate_alpha'] not in (0.,.5,1.) or c['preparation_s'] not in (.2,.6) or c['force_N'] not in (-64.,64.) or
            not np.isclose(c['duration_s'],.1) or not np.isclose(c['onset_s'],6.+c['preparation_s']) or
            not np.isclose(c['primary_window_start_s'],6.) or not np.isclose(c['primary_window_end_s'],c['onset_s']+.1+3.)):
            raise ValueError('frozen task6 hypothetical condition mismatch')
        groups.setdefault(c['source_group_id'],[]).append(c)
    expected={(a,p,f) for a in (0.,.5,1.) for p in (.2,.6) for f in (-64.,64.)}
    if len(groups)!=4 or any(len(c)!=12 or {(r['candidate_alpha'],r['preparation_s'],r['force_N']) for r in c}!=expected for c in groups.values()):raise ValueError('four complete late source groups required')
    for r in records:
        c=byid[r['case_id']]
        if any(r[k]!=c[k] for k in ('source_group_id','bank_index','candidate_alpha','preparation_s','force_N')):raise ValueError('late analysis condition changed')
    return byid


def grouped_metrics(y,p,meta,metric_fn):
    result={'overall':metric_fn(y,p,meta),'per_bank':{},'per_source':{}}
    for key,field in [('per_bank','bank_index'),('per_source','source_group_id')]:
        for value in sorted({m[field] for m in meta}):
            ids=[i for i,m in enumerate(meta) if m[field]==value]
            result[key][str(value)]=metric_fn(y[ids],p[ids],[meta[i] for i in ids])
    return result


def rejection_report(y,p,meta):
    user=np.array([m['alpha']==1. for m in meta]);out={}
    for threshold in (.8,.85):
        rejected=p<threshold
        out[str(threshold)]=dict(user_rows=int(user.sum()),rejected_rows=int(np.sum(user&rejected)),survived_but_rejected_rows=int(np.sum(user&rejected&(y==1))),fall_but_not_rejected_rows=int(np.sum(user&~rejected&(y==0))))
    source={}
    for group in sorted({m['source_group_id'] for m in meta}):
        ids=[i for i,m in enumerate(meta) if m['source_group_id']==group and m['alpha']==1.]
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
    validate_plan(plan)
    if not any(expected==coordination.sha(q_export/'receipt.json') for expected in plan['input_sha256'].values()):raise ValueError('q export receipt not bound by sealed late plan')
    source_meta,source_features,source_yaw=authenticate_late_sources(root)
    inventory=json.loads((root/'inventory.json').read_text())
    analysis=root/'analysis_late';records=json.loads((analysis/'results.json').read_text())
    byid=validate_inventory(inventory,records)
    with np.load(root/'evaluation/source_features.npz',allow_pickle=False) as z:sources=z['native_observation_user1'].copy()
    if sources.shape!=(48,351):raise ValueError('late native351 source features required')
    index={c['case_id']:i for i,c in enumerate(inventory)}
    inputs={str(p):coordination.sha(p) for p in [plan_path,root/'seal.json',root/'inventory.json',analysis/'results.json',root/'evaluation/source_features.npz',root/'sources/seal.json',root/'sources/records.json',root/'sources/source_features.npz',q_export/'receipt.json']}
    features=[];labels=[];meta=[];excluded=[];source_groups={};candidate_latents={}
    for record in records:
        case=byid[record['case_id']];lane=index[record['case_id']];raw_path=Path(record['raw_file'])
        si=case['source_index'];src=source_meta[si]
        if any(case[k]!=src[k] for k in ('source_group_id','bank_index','split')):raise ValueError('late source metadata mismatch')
        np.testing.assert_array_equal(sources[lane],source_features[si])
        if coordination.sha(raw_path)!=record['raw_sha256']:raise ValueError('late raw changed')
        inputs[str(raw_path)]=record['raw_sha256']
        with np.load(raw_path,allow_pickle=False) as raw:
            if not raw['active_tick'][0] or abs(float(raw['time'][0])-6.)>1e-6:raise ValueError('task6 first pre-force query required')
            query=raw['query_observation'][0];latent=raw['composed_latent'][0];yaw=float(raw['pre_upper_yaw'][0])
        if abs(np.arctan2(np.sin(yaw-source_yaw[si]),np.cos(yaw-source_yaw[si])))>1e-6:raise ValueError('late source pre-query yaw mismatch')
        x,_=frozen.feature_row(sources[lane],query,latent,yaw,case,plan['config']['residual_interface'],cfg)
        group=case['source_group_id']
        if group in source_groups:np.testing.assert_array_equal(sources[lane],source_groups[group])
        source_groups[group]=sources[lane]
        key=(group,case['candidate_alpha'])
        if key in candidate_latents:np.testing.assert_allclose(latent,candidate_latents[key],atol=1e-6,rtol=1e-6)
        candidate_latents[key]=latent
        label=frozen.resolved_label(record['primary'])
        if label is None:
            excluded.append(dict(case_id=case['case_id'],reason=record['primary']['label_status']));continue
        features.append(x);labels.append(label);meta.append(dict(case_id=case['case_id'],source_group_id=group,bank_index=case['bank_index'],alpha=case['candidate_alpha'],preparation_s=case['preparation_s'],force_N=case['force_N']))
    if not labels:raise ValueError('no resolved late labels')
    features=np.asarray(features,np.float32);labels=np.asarray(labels,np.float32)
    probabilities={};metrics={};rejections={}
    for arm in ('bce','bce_rank'):
        for seed in (0,1,2):
            model=frozen.make_model(41);checkpoint=torch.load(q_export/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict'])
            probabilities[f'{arm}/seed{seed}']=frozen.predict(model,features)
        probabilities[f'{arm}/ensemble']=np.mean([probabilities[f'{arm}/seed{s}'] for s in (0,1,2)],axis=0)
    probabilities['original_training_prior']=np.full(len(labels),57/72)
    for name,p in probabilities.items():
        metrics[name]=grouped_metrics(labels,p,meta,frozen.metrics)
        rejections[name]=rejection_report(labels,p,meta)
    # Existing highest-precision JAX export must agree with the local Torch ensembles.
    jax_probabilities=np.asarray(coordination.infer_probabilities(params,features));errors={}
    for ai,arm in enumerate(('bce','bce_rank')):
        error=float(np.max(np.abs(jax_probabilities[ai]-probabilities[f'{arm}/ensemble'])));errors[arm]=error
        if not np.isfinite(error) or error>2e-6:raise ValueError(f'late ensemble Torch/JAX error {error}')
    output.mkdir(parents=True)
    write_json(output/'metrics.json',metrics);write_json(output/'user_rejection.json',rejections)
    write_json(output/'predictions.json',[dict(m,label=int(y),probabilities={k:float(p[i]) for k,p in probabilities.items()}) for i,(m,y) in enumerate(zip(meta,labels))])
    np.savez(output/'features_predictions.npz',features=features,labels=labels,jax_ensemble_probabilities=jax_probabilities)
    write_json(output/'receipt.json',dict(inputs_sha256=inputs,evaluator_sha256=coordination.sha(__file__),frozen_predictor_source_sha256=source['implementation_sha256'],resolved_rows=len(labels),excluded_invalid=excluded,ensemble_probability_errors=errors,role='task1-trained frozen q extrapolated to task6 engineering DEV; no fitting, selection or threshold tuning',source_task_s=6.,new_optimizer_updates=0,new_physics_steps=0,thresholds=[.8,.85]))
    write_json(output/'seal.json',{p.name:coordination.sha(p) for p in output.iterdir() if p.is_file()})
    return metrics


def validate_plan(plan):
    if plan.get('mode')!='h75_late_state_tolerance48':raise ValueError('explicit task6 late48 plan required')
    protocol=plan['protocol']
    fixed=dict(expected_cases=48,source_task_s=6.,horizon_s=10.,new_horizon_s=4.,maximum_prefix_lower_ticks=0,maximum_lower_ticks=38400,maximum_total_lower_ticks=38400,optimizer_updates=0,bank_indices=[2,5],candidate_alphas=[0.,.5,1.],preparation_s=[.2,.6],forces_N=[-64.,64.],duration_s=.1,primary_after_pulse_end_s=3.)
    if any(protocol.get(k)!=v for k,v in fixed.items()):raise ValueError('frozen late48 protocol mismatch')


def authenticate_late_sources(root):
    root=Path(root);seal=json.loads((root/'sources/seal.json').read_text())
    if seal['plan_sha256']!=coordination.sha(root/'plan.json'):raise ValueError('late source seal plan changed')
    required=[root/'sources/records.json',root/'sources/source_features.npz',root/'sources/complete_carry.pkl',root/'sources/executed_paths.pkl',root/'sources/host_state.npz']
    if any(str(p) not in seal['files'] for p in required):raise ValueError('late source seal missing required artifacts')
    for file,expected in seal['files'].items():
        if coordination.sha(file)!=expected:raise ValueError('late complete source artifact changed')
    records=json.loads((root/'sources/records.json').read_text())
    if (len(records)!=4 or {r['source_index'] for r in records}!={0,1,2,3} or any(r.get('source_task_tick')!=1200 or r.get('source_physical_tick')!=1900 or r.get('split')!='late_state_development' for r in records)):raise ValueError('four task6/physical1900 sources required')
    with np.load(root/'sources/source_features.npz',allow_pickle=False) as z:
        features=z['native_observation_user1'].copy();yaw=z['source_world_yaw'].copy()
    if features.shape!=(4,351) or yaw.shape!=(4,) or not np.isfinite(features).all() or not np.isfinite(yaw).all():raise ValueError('four finite source features/yaw required')
    return {r['source_index']:r for r in records},features,yaw
