"""Frozen tolerance scores and bounded causal alpha coordination (no physics)."""
from pathlib import Path
from typing import NamedTuple, Any
import hashlib
import json
import shutil
import importlib.util
import numpy as np


def feasible_alpha_intervals(candidate_alphas,no_fall_probability,task_probability,path_error_prediction,*,
                             no_fall_threshold,task_threshold,max_path_error_m):
    """Threshold model outputs into possibly disconnected, approximate grid intervals.

    Endpoints describe consecutive qualifying queried samples. Interior alpha
    values remain unverified; no monotonicity, interpolation or safety guarantee
    is assumed. Missing/nonfinite model outputs are infeasible, never repaired.
    Empty output is an explicit result, not an implicit alpha fallback.
    """
    grid=np.asarray(candidate_alphas,float);q=np.asarray(no_fall_probability,float)
    task=np.asarray(task_probability,float);path=np.asarray(path_error_prediction,float)
    if (grid.ndim!=1 or len(grid)==0 or not np.isfinite(grid).all()
        or np.any(np.diff(grid)<=0) or np.any((grid<0)|(grid>1))):
        raise ValueError('finite strictly increasing alpha grid in[0,1] required')
    if any(x.shape!=grid.shape for x in (q,task,path)):raise ValueError('one prediction per queried alpha required')
    if (not np.isfinite([no_fall_threshold,task_threshold,max_path_error_m]).all()
        or not 0<=no_fall_threshold<=1 or not 0<=task_threshold<=1 or max_path_error_m<0):
        raise ValueError('finite probability thresholds and nonnegative path limit required')
    if any(np.any(np.isfinite(x)&((x<0)|(x>1))) for x in (q,task)) or np.any(np.isfinite(path)&(path<0)):
        raise ValueError('probabilities in[0,1] and nonnegative path error required')
    valid=np.isfinite(q)&np.isfinite(task)&np.isfinite(path)
    mask=valid&(q>=no_fall_threshold)&(task>=task_threshold)&(path<=max_path_error_m)
    indices=np.flatnonzero(mask);runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if len(indices) else []
    return dict(candidate_alphas=grid.tolist(),feasible_mask=mask.tolist(),valid_prediction_mask=valid.tolist(),
        estimated_intervals=[[float(grid[r[0]]),float(grid[r[-1]])] for r in runs],
        feasible_candidates=grid[mask].tolist(),empty=not bool(len(indices)),
        no_fall_threshold=float(no_fall_threshold),task_threshold=float(task_threshold),max_path_error_m=float(max_path_error_m),
        interpretation='Consecutive qualifying queried samples; continuous interior unverified. Model estimates, no safety guarantee.')


class OnlineCarry(NamedTuple):
    native: Any
    latched: Any
    prev_alpha: Any

    @property
    def state(self):
        return self.native.state


class DwellOnlineCarry(NamedTuple):
    native: Any
    latched: Any
    prev_alpha: Any
    hold_until_tick: Any

    @property
    def state(self):
        return self.native.state


def apply_alpha_dwell(proposed_alpha,proposed_latched,task_mask,permission,
                      prev_alpha,latched,hold_until_tick,current_tick,*,
                      candidate_alphas=(0.,.25,.5,.75,1.),dwell_ticks=40):
    """Optional causal hold, timed by native lower ticks (200Hz).

    Reason codes:0=no release,1=expiry,2=current task-mask infeasible,
    3=publication denied. A q decline alone never releases the hold.
    Expiry follows elapsed native time even during denied publication. Actual
    alpha changes restart the deadline, including a change to fallback .5.
    No initial hold is imposed; caller initializes deadline to current tick.
    """
    import jax.numpy as j
    if isinstance(dwell_ticks,bool) or not isinstance(dwell_ticks,int) or dwell_ticks<=0:raise ValueError('positive static dwell tick count required')
    grid=j.asarray(candidate_alphas,j.float32);idx=j.argmin(j.abs(grid-prev_alpha))
    active=current_tick<hold_until_tick
    feasible=task_mask[idx]
    held=permission&active&feasible
    alpha=j.where(permission,j.where(held,prev_alpha,proposed_alpha),prev_alpha)
    latch=j.where(permission,j.where(held,latched,proposed_latched),latched)
    changed=permission&(alpha!=prev_alpha)
    deadline=j.where(changed,current_tick+dwell_ticks,
                     j.where(permission&((active&~feasible)|~active),0,hold_until_tick))
    reason=j.where(~permission,3,j.where(active&~feasible,2,j.where((hold_until_tick>0)&~active,1,0)))
    return dict(alpha=alpha,latched=latch,hold_until_tick=deadline,
        held=held,remaining_ticks=j.maximum(deadline-current_tick,0),
        release_reason=reason,alpha_changed=changed)


def file_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_export(path):
    path=Path(path)
    if path.is_dir():path=path/'parameters.npz'
    receipt_path=path.parent/'receipt.json'
    receipt=json.loads(receipt_path.read_text())
    for name,expected in receipt['files_sha256'].items():
        if sha(path.parent/name)!=expected:raise ValueError('sealed predictor export changed')
    with np.load(path,allow_pickle=False) as data:params={k:data[k].copy() for k in data.files}
    if receipt.get('feature_mask_keep_indices') is not None:
        mask=receipt['feature_mask_keep_indices'];removed=[i for i in range(41) if i not in mask]
        if mask!=[4]+list(range(34,41)) or receipt.get('masked_export_opt_in') is not True or receipt.get('mask_implementation')!='first_layer_input_rows_zeroed' or np.any(params['w0'][...,removed,:]!=0):raise ValueError('masked export metadata or folded parameters differ')
    return params


def load_five_alpha_export(path, *, expected_export_sha256,
                           candidate_alphas=(0.,.25,.5,.75,1.), hypothetical_duration_s=.15):
    """Explicit transport-only loader for fixed-last five-alpha fit artifacts.

    Caller supplies an independently frozen export-manifest hash. Authenticate
    parameter/selection/checkpoint/config/data-receipt identities and the small
    source feature artifacts; do not rescan historical physics or adopt a model.
    Arm0 is a three-seed 41-16-16-1 ensemble. Legacy load_export is unchanged.
    """
    root=Path(path);root=root if root.is_dir() else root.parent
    export_path=root/'export_manifest.json'
    if not isinstance(expected_export_sha256,str) or len(expected_export_sha256)!=64 or sha(export_path)!=expected_export_sha256:
        raise ValueError('five-alpha export manifest identity differs')
    export=json.loads(export_path.read_text());selection_path=root/'selection_manifest.json'
    manifest=json.loads(selection_path.read_text());seal=json.loads((root/'selection_seal.json').read_text())
    grid=[0.,.25,.5,.75,1.]
    if (list(candidate_alphas)!=grid or hypothetical_duration_s!=.15 or
        export.get('candidate_alpha_grid')!=grid or export.get('source_user_alpha')!=.5 or
        export.get('arms')!=['bce_rank_last'] or manifest.get('kind')!='five_alpha_fixed_last' or
        manifest.get('candidate_alpha_grid')!=grid or manifest.get('feature_dimension')!=41):
        raise ValueError('five-alpha grid/arm/feature/duration contract differs')
    if export['parameters_sha256']!=sha(root/'parameters.npz') or export['selection_manifest_sha256']!=sha(selection_path) or seal['selection_manifest_sha256']!=sha(selection_path):
        raise ValueError('five-alpha parameters or selection seal differs')
    if sha(manifest['config'])!=manifest['config_sha256'] or sha(root/'data_receipt.json')!=manifest['data_receipt_sha256']:
        raise ValueError('five-alpha config/data receipt differs')
    cfg=json.loads(Path(manifest['config']).read_text());receipt=json.loads((root/'data_receipt.json').read_text())
    if (cfg['architecture']['hidden']!=[16,16] or cfg['training']['seeds']!=[0,1,2] or
        cfg['features']['full_dim']!=41 or cfg['features']['full_indices']!=list(range(300,320))+list(range(337,351)) or
        cfg['training'].get('fixed_last') is not True):raise ValueError('five-alpha network/config feature semantics differ')
    selected=manifest['selected']
    if len(selected)!=3 or [r['seed'] for r in selected]!=[0,1,2] or seal['selected_sha256']!={r['file']:r['sha256'] for r in selected}:
        raise ValueError('five-alpha three-seed checkpoint inventory differs')
    with np.load(root/'parameters.npz',allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
    shapes=dict(w0=(1,3,41,16),b0=(1,3,16),w1=(1,3,16,16),b1=(1,3,16),w2=(1,3,16,1),b2=(1,3,1))
    if set(params)!=set(shapes) or any(params[k].shape!=shape or params[k].dtype!=np.float32 or not np.isfinite(params[k]).all() for k,shape in shapes.items()):
        raise ValueError('five-alpha parameter layout/dtype/finiteness differs')
    # Verify the NPZ is the actual selected checkpoint export, not merely a
    # separately rehashed array file. Deserialization follows SHA authentication.
    import torch
    for seed,row in enumerate(selected):
        if sha(row['file'])!=row['sha256']:raise ValueError('five-alpha checkpoint identity differs')
        if any(row.get(k)!=v for k,v in dict(dim=41,arm='bce_rank',selector='last',step=cfg['training']['updates_per_model_seed']).items()):raise ValueError('five-alpha selection metadata differs')
        state=torch.load(row['file'],map_location='cpu',weights_only=True)
        if any(state.get(k)!=row[k] for k in ('dim','seed','step','arm','selector')):raise ValueError('five-alpha checkpoint metadata differs')
        for i,layer in enumerate((0,2,4)):
            for key,value in [(f'w{i}',state['state_dict'][f'{layer}.weight'].numpy().T),(f'b{i}',state['state_dict'][f'{layer}.bias'].numpy())]:
                if not np.array_equal(params[key][0,seed],value):raise ValueError('five-alpha exported weights differ from checkpoint')
    for analysis in cfg['dataset']['analyses']:
        analysis=Path(analysis);feature_file=analysis/'features_labels.npz';plan_file=analysis.parent/'plan.json'
        for file in (feature_file,plan_file):
            if receipt['inputs'].get(str(file))!=sha(file):raise ValueError('five-alpha feature/source-plan identity differs')
        protocol=json.loads(plan_file.read_text())['protocol']
        if protocol['duration_s']!=hypothetical_duration_s or protocol['candidate_alphas']!=grid or any(g['source_user_alpha']!=.5 for g in protocol['source_groups']):
            raise ValueError('five-alpha source duration/grid/user semantics differ')
        with np.load(feature_file,allow_pickle=False) as z:features=z['features']
        if (features.ndim!=2 or features.shape[1]!=41 or not np.isfinite(features).all() or
            not np.isin(features[:,34],grid).all() or not np.allclose(features[:,39],hypothetical_duration_s/.1,rtol=0,atol=1e-6)):
            raise ValueError('five-alpha saved feature duration/grid differs')
    return params


def infer_probabilities(params,features):
    """Return [arm, sample] by averaging three sigmoid probabilities."""
    import jax.numpy as j
    import jax.nn as nn
    x=j.asarray(features,j.float32)
    h=nn.relu(j.einsum('ni,asij->asnj',x,j.asarray(params['w0']),precision='highest')+j.asarray(params['b0'])[:,:,None,:])
    h=nn.relu(j.einsum('asni,asij->asnj',h,j.asarray(params['w1']),precision='highest')+j.asarray(params['b1'])[:,:,None,:])
    logits=(j.einsum('asni,asij->asnj',h,j.asarray(params['w2']),precision='highest')+j.asarray(params['b2'])[:,:,None,:])[...,0]
    return nn.sigmoid(logits).mean(axis=1)


def native_offsets(latents,interface):
    import jax.numpy as j
    a=j.tanh(latents)
    return j.stack([a[...,0]*j.where(a[...,0]>=0,interface['speed_positive_scale_m_s'],interface['speed_negative_scale_m_s']),a[...,1]*interface['steer_scale_rad']],axis=-1)


def candidate_features(observation,latents,yaw,interface,*,candidate_alphas=(0.,.5,1.),hypothetical_duration_s=.1):
    import jax.numpy as j
    if not np.isfinite(hypothetical_duration_s) or hypothetical_duration_s<=0:
        raise ValueError('positive finite hypothetical duration required')
    offsets=native_offsets(latents,interface)
    alpha=j.asarray(candidate_alphas,j.float32)
    count=alpha.shape[0]
    force=j.array([-64.,64.,-64.,64.],j.float32)
    preparation=j.array([.2,.2,.6,.6],j.float32)
    shared=j.stack([j.broadcast_to(alpha[:,None],(count,4)),j.broadcast_to(offsets[:,0,None],(count,4)),j.broadcast_to(offsets[:,1,None]/.2,(count,4)),j.broadcast_to(force[None,:]*j.sin(yaw)/64.,(count,4)),j.broadcast_to(force[None,:]*j.cos(yaw)/64.,(count,4)),j.full((count,4),hypothetical_duration_s/.1,j.float32),j.broadcast_to(preparation[None,:]/.6,(count,4))],axis=-1)
    indices=j.array(list(range(300,320))+list(range(337,351)))
    return j.concatenate([j.broadcast_to(observation[indices],(count,4,34)),shared],axis=-1)


def candidate_probabilities(params,observation,latents,yaw,interface,*,candidate_alphas=(0.,.5,1.),hypothetical_duration_s=.1):
    features=candidate_features(observation,latents,yaw,interface,candidate_alphas=candidate_alphas,hypothetical_duration_s=hypothetical_duration_s)
    return infer_probabilities(params,features.reshape(-1,41)).reshape((-1,len(candidate_alphas),4))


def preview_curvature(progress,path_s,path_curvature):
    """6m preview includes every knot and both interpolated endpoints."""
    import jax.numpy as j
    end=j.minimum(progress+6.,path_s[-1]);start=j.clip(progress,path_s[0],path_s[-1])
    knots=j.where((path_s>=start)&(path_s<=end),j.abs(path_curvature),0.)
    return j.maximum(j.max(knots),j.maximum(j.abs(j.interp(start,path_s,path_curvature)),j.abs(j.interp(end,path_s,path_curvature))))


def choose_alpha(targets,max_curvature,nominal,v_user,qmin,permission,latched,prev_alpha,*,use_q=True,user_alpha=1.,task_aware_fallback=False,guard_fallback=False,candidate_alphas=(0.,.5,1.),curvature_task_mode='hard',candidate_valid=None):
    """No time/event/route inputs; thresholds are engineering gates."""
    import jax.numpy as j
    if curvature_task_mode not in ('hard','soft_penalty'):raise ValueError('unknown curvature task mode')
    permission=j.asarray(permission,dtype=bool);latched=j.asarray(latched,dtype=bool)
    alpha=j.asarray(candidate_alphas,j.float32)
    count=alpha.shape[0]
    task_mask=targets[:,0]**2*max_curvature<=3.
    user_index=j.argmin(j.abs(alpha-user_alpha))
    cost=(.1+.9*user_alpha)*((targets[:,0]-v_user)/1.)**2+(1.-.9*user_alpha)*((targets[:,1]-nominal[1])/.2)**2+.05*(alpha-user_alpha)**2+.02*(alpha-prev_alpha)**2
    original_cost=cost
    if curvature_task_mode=='soft_penalty':
        historical_mask=task_mask
        valid=j.all(j.isfinite(targets),axis=-1)&j.isfinite(cost)&j.isfinite(max_curvature)&(max_curvature>=0)
        if candidate_valid is not None:valid=valid&j.asarray(candidate_valid,dtype=bool)
        acceleration=targets[:,0]**2*max_curvature
        excess=j.maximum(acceleration/3.-1.,0.)
        penalty=excess**2
        cost=original_cost+penalty
        valid=valid&j.isfinite(cost)
        task_mask=valid
    quality=j.where(use_q,qmin>=.8,j.ones(count,dtype=bool))
    restore=task_mask[user_index]&j.where(use_q,qmin[user_index]>=j.where(latched,.85,.8),True)
    eligible=task_mask&quality&(j.arange(count)!=user_index)
    degraded=(~restore)&(~j.any(eligible))
    fallback=j.where(j.any(eligible),alpha[j.argmin(j.where(eligible,cost,j.inf))],.5)
    task_candidates=task_mask&(qmin==j.max(j.where(task_mask,qmin,-j.inf)))
    task_best=alpha[j.argmin(j.where(task_candidates,cost,j.inf))]
    guard_selected=j.asarray(guard_fallback)&j.asarray(use_q)&degraded&j.any(task_mask)
    guard_best=j.where(task_mask[user_index],alpha[user_index],alpha[j.argmin(j.where(task_mask,cost,j.inf))])
    task_aware_selected=j.asarray(task_aware_fallback)&degraded&j.any(task_mask)&~guard_selected
    fallback=j.where(guard_selected,guard_best,j.where(task_aware_selected,task_best,fallback))
    proposed=j.where(restore,alpha[user_index],fallback)
    if curvature_task_mode=='soft_penalty':
        selected_valid=valid[j.argmin(j.abs(alpha-proposed))]
        reference_fault=~selected_valid
        permission=permission&selected_valid
    chosen=j.where(permission,proposed,prev_alpha)
    new_latch=j.where(permission,~restore,latched)
    result=dict(alpha=chosen,latched=new_latch,prev_alpha=chosen,task_mask=task_mask,proxy_cost=cost,degraded=permission&degraded,publication_withheld=~permission,task_aware_fallback=j.asarray(task_aware_fallback),task_aware_selected=permission&task_aware_selected,guard_fallback=j.asarray(guard_fallback),guard_fallback_selected=permission&guard_selected)
    if curvature_task_mode=='soft_penalty':result.update(historical_hard_task_mask=historical_mask,valid_target_mask=valid,original_proxy_cost=original_cost,proxy_accel=acceleration,curvature_excess=excess,curvature_excess_penalty=penalty,selected_reference_valid=selected_valid,candidate_reference_fault=reference_fault)
    return result


def coordinate(params,observation,latents,yaw,nominal,v_user,path_progress,path_s,path_curvature,permission,latched,prev_alpha,interface,*,use_q=True,arm=1,user_alpha=1.,task_aware_fallback=False,guard_fallback=False,candidate_alphas=(0.,.5,1.),hypothetical_duration_s=.1,curvature_task_mode='hard'):
    import jax.numpy as j
    probabilities=candidate_probabilities(params,observation,latents,yaw,interface,candidate_alphas=candidate_alphas,hypothetical_duration_s=hypothetical_duration_s)[arm]
    qmin=j.min(probabilities,axis=-1)
    unbounded=nominal[None,:]+native_offsets(latents,interface)
    lo,hi=interface['speed_reference_bounds_m_s'];steer=interface['steer_reference_bound_rad']
    targets=j.stack([j.clip(unbounded[:,0],lo,hi),j.clip(unbounded[:,1],-steer,steer)],axis=-1)
    curvature=preview_curvature(path_progress,path_s,path_curvature)
    valid=(j.all(j.isfinite(latents),axis=-1)&j.all(j.isfinite(unbounded),axis=-1)&j.all(j.isfinite(targets),axis=-1)&(targets[:,0]>=lo)&(targets[:,0]<=hi)&(j.abs(targets[:,1])<=steer)) if curvature_task_mode=='soft_penalty' else None
    result=choose_alpha(targets,curvature,nominal,v_user,qmin,permission,latched,prev_alpha,use_q=use_q,user_alpha=user_alpha,task_aware_fallback=task_aware_fallback,guard_fallback=guard_fallback,candidate_alphas=candidate_alphas,curvature_task_mode=curvature_task_mode,candidate_valid=valid)
    result.update(probabilities=probabilities,qmin=qmin,targets=targets,unbounded_targets=unbounded,target_clipped=targets!=unbounded,max_abs_curvature=curvature)
    return result


def export_predictors(manifest_path,output,*,allow_masked_export=False):
    """Authenticate and export six frozen networks; CPU numerical audit only."""
    import torch
    output=Path(output)
    if output.exists():raise ValueError('fresh export directory required')
    # Present helper authenticates archived source independently of this module.
    pilot=file_module(Path(__file__).with_name('tolerance_predictor_pilot.py'),'_coordination_predictor_auth')
    raw_manifest=json.loads(Path(manifest_path).read_text())
    masked=raw_manifest.get('format')=='sttw_frozen_speed_query_last_v1'
    if masked and not allow_masked_export:raise ValueError('masked export requires explicit opt-in')
    user_mix=masked or raw_manifest.get('format')=='sttw_frozen_user_mix_last_v1'
    generic=user_mix or raw_manifest.get('format')=='sttw_frozen_predictor_candidates_v1'
    if generic:
        manifest,source,frozen=verify_user_mix_candidate(manifest_path,allow_masked_export=allow_masked_export) if user_mix else verify_temporal_candidates(manifest_path)
        labels=manifest['ensemble_labels'];source_root=Path(manifest['training_directory'])
    else:
        manifest,source,frozen=pilot.verify_confirmation_candidates(manifest_path)
        labels=['bce','bce_rank'];source_root=Path(manifest_path).parent
    params={};models={}
    for ai,arm in enumerate(labels):
        for seed in (0,1,2):
            row=next(r for r in manifest['candidate_models'] if r['arm']==arm and r['seed']==seed)
            state=torch.load(row['file'],map_location='cpu',weights_only=True)['state_dict']
            model=frozen.make_model(41);model.load_state_dict(state);models[ai,seed]=model
            for layer,index in enumerate((0,2,4)):
                for prefix,suffix in [('w','weight'),('b','bias')]:
                    value=state[f'{index}.{suffix}'].detach().cpu().numpy()
                    if prefix=='w':value=value.T
                    key=f'{prefix}{layer}'
                    if key not in params:params[key]=np.empty((len(labels),3)+value.shape,np.float32)
                    params[key][ai,seed]=value
    cfg=json.loads(Path(source['config']).read_text())
    mask=cfg.get('feature_mask_keep_indices') if masked else None
    if mask is not None:params['w0']=fold_input_mask(params['w0'],mask)
    if generic:
        training_file=Path(manifest['prepared_directory'])/'training.npz'
        with np.load(training_file,allow_pickle=False) as z:features=z['full'].copy()
        data=dict(full=features,inputs={str(training_file):sha(training_file)})
    else:
        receipt=json.loads((source_root/'data_receipt.json').read_text())
        training_analysis=next(Path(p).parent for p in receipt['train']['inputs'] if p.endswith('/initial_carry_review.json'))
        data=frozen.load_dataset(training_analysis,'training',cfg)
    rng=np.random.default_rng(75);features=np.concatenate([data['full'],rng.uniform(-5,5,(128,41)).astype(np.float32)])
    import jax.numpy as j
    import jax.nn as nn
    if masked:
        query_features=np.asarray(candidate_features(j.asarray(rng.uniform(-5,5,351),j.float32),j.asarray(rng.uniform(-2,2,(3,2)),j.float32),j.asarray(.6,j.float32),dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2))).reshape(12,41)
        features=np.concatenate([features,query_features])
    # Check each of six models, not only the ensemble.
    h=nn.relu(j.einsum('ni,asij->asnj',j.asarray(features),j.asarray(params['w0']),precision='highest')+j.asarray(params['b0'])[:,:,None,:])
    h=nn.relu(j.einsum('asni,asij->asnj',h,j.asarray(params['w1']),precision='highest')+j.asarray(params['b1'])[:,:,None,:])
    actual=np.asarray(nn.sigmoid((j.einsum('asni,asij->asnj',h,j.asarray(params['w2']),precision='highest')+j.asarray(params['b2'])[:,:,None,:])[...,0]))
    errors={}
    for (arm,seed),model in models.items():
        expected=frozen.predict(model,features,feature_mask_keep_indices=mask) if masked else frozen.predict(model,features);error=float(np.max(np.abs(expected-actual[arm,seed])))
        errors[f'{arm}/seed{seed}']=error
        if not np.isfinite(error) or error>2e-6:raise ValueError(f'Torch/JAX frozen predictor mismatch: arm={arm} seed={seed} maximum_absolute_probability_error={error:.9g}, tolerance=2e-6, matmul_precision=highest')
    output.mkdir(parents=True);np.savez(output/'parameters.npz',**params)
    shutil.copyfile(manifest_path,output/'original_candidate_manifest.json')
    if generic:shutil.copyfile(Path(manifest_path).with_suffix('.seal.json'),output/'original_candidate_manifest.seal.json')
    shutil.copyfile(source_root/'selection_manifest.json',output/'ranking_source_manifest.json')
    shutil.copyfile(source['config'],output/'ranking_config.json')
    shutil.copyfile(source_root/'source/tolerance_predictor_pilot.py',output/'frozen_predictor_source.py')
    for row in manifest['candidate_models']:shutil.copyfile(row['file'],output/f"{row['arm']}_seed{row['seed']}.pt")
    np.savez(output/'numerical_inputs.npz',features=features,jax_probabilities=actual)
    report=dict(candidate_manifest_sha256=sha(manifest_path),ranking_source_manifest_sha256=manifest['ranking_source_manifest_sha256'],frozen_predictor_source_sha256=source['implementation_sha256'],coordination_source_sha256=sha(__file__),training_inputs_sha256=data['inputs'],real_training_rows=len(data['full']),deterministic_random_rows=128,absolute_tolerance=2e-6,matmul_precision='highest',maximum_probability_errors=errors,arm_order=labels,source_manifest_kind='speed_query_masked_fixed_last' if masked else 'user_mix_prior_fixed_last' if user_mix else 'temporal_mix_selected_and_last' if generic else 'legacy_ranking_selected',hypothesis_order=[dict(preparation_s=p,force_N=f) for p,f in ((.2,-64),(.2,64),(.6,-64),(.6,64))],new_optimizer_updates=0,new_physics_steps=0)
    if masked:report.update(feature_mask_keep_indices=mask,masked_export_opt_in=True,mask_implementation='first_layer_input_rows_zeroed',candidate_query_audit_rows=12,input_information='speed plus candidate/disturbance query; not pure speed')
    report['files_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def _sealed_files(directory,seal_name,required):
    directory=Path(directory).resolve();seal=json.loads((directory/seal_name).read_text())
    files=seal.get('artifacts_sha256',seal)
    for name in required:
        if name not in files or sha(directory/name)!=files[name]:raise ValueError('frozen temporal artifact changed: '+name)
    return files


def authenticate_temporal_training(training,prepared):
    training=Path(training).resolve();prepared=Path(prepared).resolve()
    required=['selection_manifest.json','source/tolerance_predictor_pilot.py','resolved_config.json','data_receipt.json','preparation_receipt.json']
    for seed in (0,1,2):required.extend([f'bce_rank/seed{seed}/balanced_brier_selected.pt',f'bce_rank/seed{seed}/last.pt',f'initial_seed{seed}.pt'])
    files=_sealed_files(training,'selection_seal.json',required)
    _sealed_files(prepared,'seal.json',['receipt.json','training.npz','training.json','development.npz','development.json'])
    source=json.loads((training/'selection_manifest.json').read_text());receipt=json.loads((prepared/'receipt.json').read_text())
    if (source['actual_optimizer_updates']!=6000 or source['new_physics_steps']!=0 or source['new_ppo_updates']!=0 or
        source['implementation_sha256']!=sha(training/'source/tolerance_predictor_pilot.py') or
        source['source_snapshot_sha256']!=source['implementation_sha256'] or
        source['config_sha256']!=sha(source['config']) or source['resolved_config_sha256']!=sha(training/'resolved_config.json') or
        source['data_receipt_sha256']!=sha(training/'data_receipt.json') or receipt['config_sha256']!=source['config_sha256'] or
        receipt['implementation_sha256']!=source['implementation_sha256'] or
        json.loads((training/'preparation_receipt.json').read_text())!=receipt):raise ValueError('frozen temporal preparation/source identity mismatch')
    expected={(a,s,'balanced_brier') for a in ('bce','bce_rank') for s in (0,1,2)}
    if len(source['selected'])!=6 or {(r['arm'],r['seed'],r['selector']) for r in source['selected']}!=expected:raise ValueError('six temporal selected models required')
    for row in source['selected']:
        if row['dim']!=41 or sha(row['file'])!=row['sha256']:raise ValueError('temporal selected checkpoint changed')
    frozen=file_module(training/'source/tolerance_predictor_pilot.py','_temporal_frozen_export')
    frozen.validate_temporal_config(json.loads(Path(source['config']).read_text()))
    with np.load(prepared/'training.npz',allow_pickle=False) as z:data=dict(full=z['full'].copy(),y=z['y'].copy(),**json.loads((prepared/'training.json').read_text()))
    with np.load(prepared/'development.npz',allow_pickle=False) as z:dev=dict(full=z['full'].copy(),y=z['y'].copy(),**json.loads((prepared/'development.json').read_text()))
    frozen.validate_temporal_datasets(data,dev)
    return source,files,frozen


def freeze_temporal_candidates(training,prepared,manifest_path,*,labels=('temporal_rank_selected','temporal_rank_last')):
    """Freeze named selected/last ensembles without changing their source recipe."""
    import torch
    training=Path(training).resolve();prepared=Path(prepared).resolve();manifest_path=Path(manifest_path).resolve()
    if manifest_path.exists() or manifest_path.with_suffix('.seal.json').exists():raise ValueError('fresh temporal candidate manifest only')
    if len(labels)!=2 or len(set(labels))!=2 or any(not name.replace('_','').isalnum() for name in labels):raise ValueError('two distinct safe ensemble labels required')
    source,files,_=authenticate_temporal_training(training,prepared);rows=[]
    for slot,label in enumerate(labels):
        for seed in (0,1,2):
            selected=next(r for r in source['selected'] if r['arm']=='bce_rank' and r['seed']==seed)
            path=Path(selected['file']) if slot==0 else training/f'bce_rank/seed{seed}/last.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            if checkpoint['arm']!='bce_rank' or checkpoint['seed']!=seed or checkpoint['step']!=(selected['step'] if slot==0 else 1000):raise ValueError('temporal checkpoint metadata mismatch')
            rows.append(dict(arm=label,slot=slot,seed=seed,dim=41,file=str(path),sha256=sha(path),step=checkpoint['step'],source_arm='bce_rank',source_selector='balanced_brier' if slot==0 else 'last',initial_sha256=selected['initial_sha256']))
    manifest=dict(format='sttw_frozen_predictor_candidates_v1',ensemble_labels=list(labels),candidate_models=rows,training_directory=str(training),prepared_directory=str(prepared),ranking_source_manifest_sha256=sha(training/'selection_manifest.json'),training_selection_seal_sha256=sha(training/'selection_seal.json'),prepared_seal_sha256=sha(prepared/'seal.json'),role='seen mixed-time engineering DEV candidates; selected vs fixed final step, no new selection',new_optimizer_updates=0,new_physics_steps=0)
    manifest_path.parent.mkdir(parents=True,exist_ok=True);manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    manifest_path.with_suffix('.seal.json').write_text(json.dumps(dict(candidate_manifest_sha256=sha(manifest_path)),indent=2)+'\n')
    return manifest


def verify_temporal_candidates(manifest_path):
    manifest_path=Path(manifest_path).resolve();manifest=json.loads(manifest_path.read_text())
    seal=json.loads(manifest_path.with_suffix('.seal.json').read_text())
    if sha(manifest_path)!=seal['candidate_manifest_sha256']:raise ValueError('temporal candidate manifest changed')
    labels=manifest['ensemble_labels'];rows=manifest['candidate_models']
    if (manifest.get('format')!='sttw_frozen_predictor_candidates_v1' or len(labels)!=2 or len(set(labels))!=2 or any(not name.replace('_','').isalnum() for name in labels) or len(rows)!=6 or
        {(r['arm'],r['slot'],r['seed']) for r in rows}!={(a,i,s) for i,a in enumerate(labels) for s in (0,1,2)}):raise ValueError('two complete named temporal ensembles required')
    training=Path(manifest['training_directory']);prepared=Path(manifest['prepared_directory'])
    if (sha(training/'selection_manifest.json')!=manifest['ranking_source_manifest_sha256'] or
        sha(training/'selection_seal.json')!=manifest['training_selection_seal_sha256'] or sha(prepared/'seal.json')!=manifest['prepared_seal_sha256']):raise ValueError('temporal source seal changed')
    source,_,frozen=authenticate_temporal_training(training,prepared)
    for row in rows:
        selected=next(r for r in source['selected'] if r['arm']=='bce_rank' and r['seed']==row['seed'])
        expected=Path(selected['file']) if row['slot']==0 else training/f"bce_rank/seed{row['seed']}/last.pt"
        if (row['source_arm']!='bce_rank' or row['source_selector']!=('balanced_brier' if row['slot']==0 else 'last') or
            Path(row['file'])!=expected or sha(expected)!=row['sha256'] or row['initial_sha256']!=selected['initial_sha256'] or row['dim']!=41 or row['step']!=(selected['step'] if row['slot']==0 else 1000)):
            raise ValueError('temporal candidate weight identity changed')
    return manifest,source,frozen


def speedcap_latent(alpha1_latent,nominal,max_curvature,interface):
    """Cap only native alpha1 speed target; retain its steering latent exactly."""
    import jax.numpy as j
    offsets=native_offsets(alpha1_latent,interface)
    lo,hi=interface['speed_reference_bounds_m_s']
    original=j.clip(nominal[0]+offsets[0],lo,hi)
    cap=j.where(max_curvature>0,j.sqrt(3./j.maximum(max_curvature,1e-12)),j.inf)
    requested=j.minimum(original,cap)
    desired=requested-nominal[0]
    scale=j.where(desired>=0,interface['speed_positive_scale_m_s'],interface['speed_negative_scale_m_s'])
    bounded=j.clip(desired/scale,-1.+1e-6,1.-1e-6)
    z=alpha1_latent.at[0].set(j.arctanh(bounded))
    reachable=j.clip(nominal[0]+bounded*scale,lo,hi)
    return z,dict(cap_m_s=cap,requested_target_m_s=requested,reachable_target_m_s=reachable,h75_target_m_s=original,active=j.abs(reachable-original)>1e-6)


def authenticate_user_mix_training(training,prepared,*,allow_masked_export=False):
    """Authenticate the three fixed-final checkpoints without selecting by DEV."""
    import torch
    training=Path(training).resolve();prepared=Path(prepared).resolve()
    required=['selection_manifest.json','source/tolerance_predictor_pilot.py','resolved_config.json','data_receipt.json','preparation_receipt.json']
    for seed in (0,1,2):required.extend([f'bce_rank/seed{seed}/last_selected.pt',f'bce_rank/seed{seed}/last.pt',f'initial_seed{seed}.pt'])
    files=_sealed_files(training,'selection_seal.json',required)
    _sealed_files(prepared,'seal.json',['receipt.json','training.npz','training.json','development.npz','development.json'])
    source=json.loads((training/'selection_manifest.json').read_text());receipt=json.loads((prepared/'receipt.json').read_text())
    if (source['actual_optimizer_updates']!=3000 or source['new_physics_steps']!=0 or source['new_ppo_updates']!=0 or
        source['implementation_sha256']!=sha(training/'source/tolerance_predictor_pilot.py') or source['source_snapshot_sha256']!=source['implementation_sha256'] or
        source['config_sha256']!=sha(source['config']) or source['resolved_config_sha256']!=sha(training/'resolved_config.json') or
        source['data_receipt_sha256']!=sha(training/'data_receipt.json') or receipt['config_sha256']!=source['config_sha256'] or
        receipt['implementation_sha256']!=source['implementation_sha256'] or json.loads((training/'preparation_receipt.json').read_text())!=receipt):raise ValueError('fixed user mix source/preparation identity differs')
    if len(source['selected'])!=3 or {(r['arm'],r['seed'],r['selector'],r['step']) for r in source['selected']}!={('bce_rank',s,'last',1000) for s in (0,1,2)}:raise ValueError('three fixed last1000 models required')
    frozen=file_module(training/'source/tolerance_predictor_pilot.py','_user_mix_frozen_export')
    cfg=frozen.validate_user_mix_config(json.loads(Path(source['config']).read_text()))
    if allow_masked_export:
        if cfg.get('feature_mask_keep_indices')!=[4]+list(range(34,41)):raise ValueError('only fixed speed+query input mask may opt in')
    else:require_unmasked_online_inputs(cfg)
    if cfg.get('bce_group_weights')!={'early':.75,'late':.125,'userlate':.125}:raise ValueError('preregistered prior candidate required')
    data=[]
    for role in ('training','development'):
        with np.load(prepared/(role+'.npz'),allow_pickle=False) as z:data.append(dict(full=z['full'].copy(),y=z['y'].copy(),**json.loads((prepared/(role+'.json')).read_text())))
    frozen.validate_user_mix_datasets(*data)
    for row in source['selected']:
        seed=row['seed'];last=training/f'bce_rank/seed{seed}/last.pt'
        if row['dim']!=41 or sha(row['file'])!=row['sha256'] or row['initial_sha256']!=sha(training/f'initial_seed{seed}.pt'):raise ValueError('fixed user checkpoint/init changed')
        checkpoint=torch.load(last,map_location='cpu',weights_only=True);selected=torch.load(row['file'],map_location='cpu',weights_only=True)
        if allow_masked_export and (checkpoint.get('feature_mask_keep_indices')!=cfg['feature_mask_keep_indices'] or selected.get('feature_mask_keep_indices')!=cfg['feature_mask_keep_indices'] or receipt.get('feature_mask_keep_indices')!=cfg['feature_mask_keep_indices']):raise ValueError('masked checkpoint/preparation metadata differs')
        if checkpoint['step']!=1000 or checkpoint['arm']!='bce_rank' or checkpoint['seed']!=seed or any(not torch.equal(v,selected['state_dict'][k]) for k,v in checkpoint['state_dict'].items()):raise ValueError('selected metadata differs from actual fixed last')
    return source,files,frozen


def freeze_user_mix_last(training,prepared,manifest_path,label='prior_rank_last',*,allow_masked_export=False):
    training=Path(training).resolve();prepared=Path(prepared).resolve();manifest_path=Path(manifest_path).resolve()
    if label!=('speed_query_rank_last' if allow_masked_export else 'prior_rank_last'):raise ValueError('explicit fixed ensemble label required')
    if manifest_path.exists() or manifest_path.with_suffix('.seal.json').exists():raise ValueError('fresh fixed-last manifest required')
    source,_,_=authenticate_user_mix_training(training,prepared,allow_masked_export=allow_masked_export)
    rows=[dict(arm=label,slot=0,seed=seed,dim=41,file=str(training/f'bce_rank/seed{seed}/last.pt'),sha256=sha(training/f'bce_rank/seed{seed}/last.pt'),step=1000,source_arm='bce_rank',source_selector='last',initial_sha256=sha(training/f'initial_seed{seed}.pt')) for seed in (0,1,2)]
    manifest=dict(format='sttw_frozen_speed_query_last_v1' if allow_masked_export else 'sttw_frozen_user_mix_last_v1',ensemble_labels=[label],candidate_models=rows,training_directory=str(training),prepared_directory=str(prepared),ranking_source_manifest_sha256=sha(training/'selection_manifest.json'),training_selection_seal_sha256=sha(training/'selection_seal.json'),prepared_seal_sha256=sha(prepared/'seal.json'),role='fixed final prior-matched user coverage candidate; engineering DEV only',new_optimizer_updates=0,new_physics_steps=0)
    if allow_masked_export:manifest.update(feature_mask_keep_indices=[4]+list(range(34,41)),mask_implementation='first_layer_input_rows_zeroed',masked_export_opt_in=True)
    manifest_path.parent.mkdir(parents=True,exist_ok=True);manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');manifest_path.with_suffix('.seal.json').write_text(json.dumps(dict(candidate_manifest_sha256=sha(manifest_path)),indent=2)+'\n')
    return manifest


def verify_user_mix_candidate(manifest_path,*,allow_masked_export=False):
    manifest_path=Path(manifest_path).resolve();manifest=json.loads(manifest_path.read_text())
    if sha(manifest_path)!=json.loads(manifest_path.with_suffix('.seal.json').read_text())['candidate_manifest_sha256']:raise ValueError('fixed user candidate manifest changed')
    masked=manifest.get('format')=='sttw_frozen_speed_query_last_v1'
    if masked and not allow_masked_export:raise ValueError('masked candidate requires explicit export opt-in')
    if masked and (manifest.get('feature_mask_keep_indices')!=[4]+list(range(34,41)) or manifest.get('mask_implementation')!='first_layer_input_rows_zeroed' or manifest.get('masked_export_opt_in') is not True):raise ValueError('explicit masked manifest identity differs')
    label='speed_query_rank_last' if masked else 'prior_rank_last'
    if manifest.get('format')!=('sttw_frozen_speed_query_last_v1' if masked else 'sttw_frozen_user_mix_last_v1') or manifest['ensemble_labels']!=[label] or len(manifest['candidate_models'])!=3:raise ValueError('single fixed prior ensemble required')
    training=Path(manifest['training_directory']);prepared=Path(manifest['prepared_directory']);source,_,frozen=authenticate_user_mix_training(training,prepared,allow_masked_export=masked and allow_masked_export)
    for file,key in [(training/'selection_manifest.json','ranking_source_manifest_sha256'),(training/'selection_seal.json','training_selection_seal_sha256'),(prepared/'seal.json','prepared_seal_sha256')]:
        if sha(file)!=manifest[key]:raise ValueError('fixed prior source seal changed')
    if {(r['arm'],r['slot'],r['seed'],r['step']) for r in manifest['candidate_models']}!={(label,0,s,1000) for s in (0,1,2)}:raise ValueError('fixed prior candidate slots differ')
    for row in manifest['candidate_models']:
        if row['file']!=str(training/f"bce_rank/seed{row['seed']}/last.pt") or sha(row['file'])!=row['sha256']:raise ValueError('fixed prior checkpoint changed')
    return manifest,source,frozen


def require_unmasked_online_inputs(config):
    if config.get('feature_mask_keep_indices') is not None:raise ValueError('offline feature-mask ablation cannot be exported as an unmasked online predictor')


def fold_input_mask(first_layer_weights,keep_indices):
    """Fold the explicitly declared input zero mask into exported input rows."""
    if list(keep_indices)!=[4]+list(range(34,41)) or np.asarray(first_layer_weights).shape[-2]!=41:raise ValueError('declared speed+query mask and41 input rows required')
    folded=np.array(first_layer_weights,copy=True)
    removed=[i for i in range(41) if i not in keep_indices];folded[...,removed,:]=0
    return folded
