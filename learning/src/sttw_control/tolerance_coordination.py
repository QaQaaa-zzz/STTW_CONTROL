"""Frozen tolerance scores and bounded causal alpha coordination (no physics)."""
from pathlib import Path
from typing import NamedTuple, Any
import hashlib
import json
import shutil
import importlib.util
import numpy as np


class OnlineCarry(NamedTuple):
    native: Any
    latched: Any
    prev_alpha: Any

    @property
    def state(self):
        return self.native.state


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
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


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


def candidate_features(observation,latents,yaw,interface):
    import jax.numpy as j
    offsets=native_offsets(latents,interface)
    alpha=j.array([0.,.5,1.],j.float32)
    force=j.array([-64.,64.,-64.,64.],j.float32)
    preparation=j.array([.2,.2,.6,.6],j.float32)
    shared=j.stack([j.broadcast_to(alpha[:,None],(3,4)),j.broadcast_to(offsets[:,0,None],(3,4)),j.broadcast_to(offsets[:,1,None]/.2,(3,4)),j.broadcast_to(force[None,:]*j.sin(yaw)/64.,(3,4)),j.broadcast_to(force[None,:]*j.cos(yaw)/64.,(3,4)),j.ones((3,4),j.float32),j.broadcast_to(preparation[None,:]/.6,(3,4))],axis=-1)
    indices=j.array(list(range(300,320))+list(range(337,351)))
    return j.concatenate([j.broadcast_to(observation[indices],(3,4,34)),shared],axis=-1)


def candidate_probabilities(params,observation,latents,yaw,interface):
    return infer_probabilities(params,candidate_features(observation,latents,yaw,interface).reshape(12,41)).reshape(2,3,4)


def preview_curvature(progress,path_s,path_curvature):
    """6m preview includes every knot and both interpolated endpoints."""
    import jax.numpy as j
    end=j.minimum(progress+6.,path_s[-1]);start=j.clip(progress,path_s[0],path_s[-1])
    knots=j.where((path_s>=start)&(path_s<=end),j.abs(path_curvature),0.)
    return j.maximum(j.max(knots),j.maximum(j.abs(j.interp(start,path_s,path_curvature)),j.abs(j.interp(end,path_s,path_curvature))))


def choose_alpha(targets,max_curvature,nominal,v_user,qmin,permission,latched,prev_alpha,*,use_q=True,user_alpha=1.):
    """No time/event/route inputs; thresholds are engineering gates."""
    import jax.numpy as j
    permission=j.asarray(permission,dtype=bool);latched=j.asarray(latched,dtype=bool)
    alpha=j.array([0.,.5,1.],j.float32)
    task_mask=targets[:,0]**2*max_curvature<=3.
    user_index=j.argmin(j.abs(alpha-user_alpha))
    cost=(.1+.9*user_alpha)*((targets[:,0]-v_user)/1.)**2+(1.-.9*user_alpha)*((targets[:,1]-nominal[1])/.2)**2+.05*(alpha-user_alpha)**2+.02*(alpha-prev_alpha)**2
    quality=j.where(use_q,qmin>=.8,j.ones(3,dtype=bool))
    restore=task_mask[user_index]&j.where(use_q,qmin[user_index]>=j.where(latched,.85,.8),True)
    eligible=task_mask&quality&(j.arange(3)!=user_index)
    degraded=(~restore)&(~j.any(eligible))
    fallback=j.where(j.any(eligible),alpha[j.argmin(j.where(eligible,cost,j.inf))],.5)
    proposed=j.where(restore,alpha[user_index],fallback)
    chosen=j.where(permission,proposed,prev_alpha)
    new_latch=j.where(permission,~restore,latched)
    return dict(alpha=chosen,latched=new_latch,prev_alpha=chosen,task_mask=task_mask,proxy_cost=cost,degraded=permission&degraded,publication_withheld=~permission)


def coordinate(params,observation,latents,yaw,nominal,v_user,path_progress,path_s,path_curvature,permission,latched,prev_alpha,interface,*,use_q=True,arm=1,user_alpha=1.):
    import jax.numpy as j
    probabilities=candidate_probabilities(params,observation,latents,yaw,interface)[arm]
    qmin=j.min(probabilities,axis=-1)
    unbounded=nominal[None,:]+native_offsets(latents,interface)
    lo,hi=interface['speed_reference_bounds_m_s'];steer=interface['steer_reference_bound_rad']
    targets=j.stack([j.clip(unbounded[:,0],lo,hi),j.clip(unbounded[:,1],-steer,steer)],axis=-1)
    curvature=preview_curvature(path_progress,path_s,path_curvature)
    result=choose_alpha(targets,curvature,nominal,v_user,qmin,permission,latched,prev_alpha,use_q=use_q,user_alpha=user_alpha)
    result.update(probabilities=probabilities,qmin=qmin,targets=targets,unbounded_targets=unbounded,target_clipped=targets!=unbounded,max_abs_curvature=curvature)
    return result


def export_predictors(manifest_path,output):
    """Authenticate and export six frozen networks; CPU numerical audit only."""
    import torch
    output=Path(output)
    if output.exists():raise ValueError('fresh export directory required')
    # Present helper authenticates archived source independently of this module.
    pilot=file_module(Path(__file__).with_name('tolerance_predictor_pilot.py'),'_coordination_predictor_auth')
    raw_manifest=json.loads(Path(manifest_path).read_text())
    generic=raw_manifest.get('format')=='sttw_frozen_predictor_candidates_v1'
    if generic:
        manifest,source,frozen=verify_temporal_candidates(manifest_path)
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
                    if key not in params:params[key]=np.empty((2,3)+value.shape,np.float32)
                    params[key][ai,seed]=value
    cfg=json.loads(Path(source['config']).read_text())
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
    # Check each of six models, not only the ensemble.
    h=nn.relu(j.einsum('ni,asij->asnj',j.asarray(features),j.asarray(params['w0']),precision='highest')+j.asarray(params['b0'])[:,:,None,:])
    h=nn.relu(j.einsum('asni,asij->asnj',h,j.asarray(params['w1']),precision='highest')+j.asarray(params['b1'])[:,:,None,:])
    actual=np.asarray(nn.sigmoid((j.einsum('asni,asij->asnj',h,j.asarray(params['w2']),precision='highest')+j.asarray(params['b2'])[:,:,None,:])[...,0]))
    errors={}
    for (arm,seed),model in models.items():
        expected=frozen.predict(model,features);error=float(np.max(np.abs(expected-actual[arm,seed])))
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
    report=dict(candidate_manifest_sha256=sha(manifest_path),ranking_source_manifest_sha256=manifest['ranking_source_manifest_sha256'],frozen_predictor_source_sha256=source['implementation_sha256'],coordination_source_sha256=sha(__file__),training_inputs_sha256=data['inputs'],real_training_rows=len(data['full']),deterministic_random_rows=128,absolute_tolerance=2e-6,matmul_precision='highest',maximum_probability_errors=errors,arm_order=labels,source_manifest_kind='temporal_mix_selected_and_last' if generic else 'legacy_ranking_selected',hypothesis_order=[dict(preparation_s=p,force_N=f) for p,f in ((.2,-64),(.2,64),(.6,-64),(.6,64))],new_optimizer_updates=0,new_physics_steps=0)
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
