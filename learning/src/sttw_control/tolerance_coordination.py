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


def choose_alpha(targets,max_curvature,nominal,v_user,qmin,permission,latched,prev_alpha,*,use_q=True):
    """No time/event/route inputs; thresholds are engineering gates."""
    import jax.numpy as j
    permission=j.asarray(permission,dtype=bool);latched=j.asarray(latched,dtype=bool)
    alpha=j.array([0.,.5,1.],j.float32)
    task_mask=targets[:,0]**2*max_curvature<=3.
    cost=((targets[:,0]-v_user)/1.)**2+.1*((targets[:,1]-nominal[1])/.2)**2+.05*(alpha-1.)**2+.02*(alpha-prev_alpha)**2
    quality=j.where(use_q,qmin>=.8,j.ones(3,dtype=bool))
    restore=task_mask[2]&j.where(use_q,qmin[2]>=j.where(latched,.85,.8),True)
    eligible=task_mask[:2]&quality[:2]
    degraded=(~restore)&(~j.any(eligible))
    fallback=j.where(j.any(eligible),alpha[j.argmin(j.where(eligible,cost[:2],j.inf))],.5)
    proposed=j.where(restore,1.,fallback)
    chosen=j.where(permission,proposed,prev_alpha)
    new_latch=j.where(permission,~restore,latched)
    return dict(alpha=chosen,latched=new_latch,prev_alpha=chosen,task_mask=task_mask,proxy_cost=cost,degraded=permission&degraded,publication_withheld=~permission)


def coordinate(params,observation,latents,yaw,nominal,v_user,path_progress,path_s,path_curvature,permission,latched,prev_alpha,interface,*,use_q=True,arm=1):
    import jax.numpy as j
    probabilities=candidate_probabilities(params,observation,latents,yaw,interface)[arm]
    qmin=j.min(probabilities,axis=-1)
    unbounded=nominal[None,:]+native_offsets(latents,interface)
    lo,hi=interface['speed_reference_bounds_m_s'];steer=interface['steer_reference_bound_rad']
    targets=j.stack([j.clip(unbounded[:,0],lo,hi),j.clip(unbounded[:,1],-steer,steer)],axis=-1)
    curvature=preview_curvature(path_progress,path_s,path_curvature)
    result=choose_alpha(targets,curvature,nominal,v_user,qmin,permission,latched,prev_alpha,use_q=use_q)
    result.update(probabilities=probabilities,qmin=qmin,targets=targets,unbounded_targets=unbounded,target_clipped=targets!=unbounded,max_abs_curvature=curvature)
    return result


def export_predictors(manifest_path,output):
    """Authenticate and export six frozen networks; CPU numerical audit only."""
    import torch
    output=Path(output)
    if output.exists():raise ValueError('fresh export directory required')
    # Present helper authenticates archived source independently of this module.
    pilot=file_module(Path(__file__).with_name('tolerance_predictor_pilot.py'),'_coordination_predictor_auth')
    manifest,source,frozen=pilot.verify_confirmation_candidates(manifest_path)
    params={};models={}
    for ai,arm in enumerate(('bce','bce_rank')):
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
    receipt=json.loads((Path(manifest_path).parent/'data_receipt.json').read_text())
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
    shutil.copyfile(Path(manifest_path).parent/'selection_manifest.json',output/'ranking_source_manifest.json')
    shutil.copyfile(source['config'],output/'ranking_config.json')
    shutil.copyfile(Path(manifest_path).parent/'source/tolerance_predictor_pilot.py',output/'frozen_predictor_source.py')
    for row in manifest['candidate_models']:shutil.copyfile(row['file'],output/f"{row['arm']}_seed{row['seed']}.pt")
    np.savez(output/'numerical_inputs.npz',features=features,jax_probabilities=actual)
    report=dict(candidate_manifest_sha256=sha(manifest_path),ranking_source_manifest_sha256=manifest['ranking_source_manifest_sha256'],frozen_predictor_source_sha256=source['implementation_sha256'],coordination_source_sha256=sha(__file__),training_inputs_sha256=data['inputs'],real_training_rows=len(data['full']),deterministic_random_rows=128,absolute_tolerance=2e-6,matmul_precision='highest',maximum_probability_errors=errors,arm_order=['bce','bce_rank'],hypothesis_order=[dict(preparation_s=p,force_N=f) for p,f in ((.2,-64),(.2,64),(.6,-64),(.6,64))],new_optimizer_updates=0,new_physics_steps=0)
    report['files_sha256']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
