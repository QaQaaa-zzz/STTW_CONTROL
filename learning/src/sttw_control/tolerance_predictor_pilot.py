"""Frozen small CPU predictor pilot; no controller selection or physics.

Train/dev labels enter fit. Held-out labels enter only evaluate after six selected
files and their manifest have been authenticated. All probabilities are descriptive.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def validate_config(cfg):
    f=cfg['features'];t=cfg['training']
    if (f['full_indices']!=list(range(300,320))+list(range(337,351)) or f['speed_indices']!=[304] or
        f['full_dim']!=41 or f['speed_dim']!=8 or cfg['architecture']['hidden']!=[16,16] or
        t['seeds']!=[0,1,2] or t['updates_per_model_seed']!=1000 or t['maximum_total_optimizer_updates']!=6000 or
        t['learning_rate']!=.001 or t['weight_decay']!=.001 or t['development_every_updates']!=100 or
        t['optimizer']!='AdamW' or t['device']!='CPU' or cfg['budget']['new_physics_steps']!=0):
        raise ValueError('frozen pilot configuration mismatch')
    return cfg


_NATIVE_AUDIT = r'''
import sys,json,pickle,importlib.util
from pathlib import Path
import numpy as np
r=Path(sys.argv[1]);p=json.loads((r/'plan.json').read_text());sys.path.insert(0,p['native_runtime_root'])
for name,file in [('_sttw_goal_continuation','native_goal_continuation.py'),('_sttw_online_scalar','geometric_preference_schedule.py')]:
    f=r/'source'/file
    if f.exists():
        spec=importlib.util.spec_from_file_location(name,f);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
with (r/'evaluation/initial_carry.pkl').open('rb') as f:carry=pickle.load(f)
if hasattr(carry,'native'):carry=carry.native
s=carry.state
spec=json.loads((Path(p['initial_source']['upper_run'])/'alpha0/frozen_config_stage52.json').read_text())
scales=np.array([x['scale'] for x in spec['network']['frame_fields']],np.float32)
frames=np.asarray(s.history.frames)[:,-1,:]
np.savez(sys.argv[2],latest_frame_normalized=np.clip(frames/scales,-5,5),latest_mask=np.asarray(s.history.mask)[:,-1],task_tick=np.asarray(s.tick),physical_tick=np.asarray(s.physical.physical_tick),source_yaw=np.asarray(s.physical.yaw_wrapped))
'''


def source_audit(root):
    """Read saved state in an isolated native namespace on CPU; no env/step."""
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['JAX_PLATFORMS']='cpu';env['CUDA_VISIBLE_DEVICES']=''
    with tempfile.TemporaryDirectory(prefix='sttw-predictor-audit-') as tmp:
        out=Path(tmp)/'source.npz'
        subprocess.run([sys.executable,'-c',_NATIVE_AUDIT,str(root),str(out)],env=env,check=True,timeout=120,capture_output=True)
        with np.load(out,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def native_target(latent,interface):
    a=np.tanh(np.asarray(latent,dtype=np.float32))
    return np.array([a[0]*(interface['speed_positive_scale_m_s'] if a[0]>=0 else interface['speed_negative_scale_m_s']),a[1]*interface['steer_scale_rad']],np.float32)


def feature_row(source,query,latent,yaw,case,interface,cfg,*,expected_source_alpha=1.):
    """Source-only state plus explicit hypothetical intervention; no outcome."""
    source=np.asarray(source,np.float32);query=np.asarray(query,np.float32)
    if source.shape!=(351,) or query.shape!=(351,) or not np.isfinite(source).all() or not np.isfinite(query).all():
        raise ValueError('finite native351 source/query required')
    np.testing.assert_allclose(np.delete(source,336),np.delete(query,336),atol=1e-6,rtol=1e-6)
    if not np.isfinite(expected_source_alpha) or not 0<=expected_source_alpha<=1:
        raise ValueError('invalid expected source alpha')
    if abs(float(source[336])-expected_source_alpha)>1e-6 or abs(float(query[336])-case['candidate_alpha'])>1e-6:raise ValueError('source/query alpha mismatch')
    target=native_target(latent,interface)
    shared=np.array([case['candidate_alpha'],target[0]/1.,target[1]/.2,
        case['force_N']*math.sin(yaw)/64.,case['force_N']*math.cos(yaw)/64.,case['duration_s']/.1,case['preparation_s']/.6],np.float32)
    full=np.concatenate([source[cfg['features']['full_indices']],shared]);speed=np.concatenate([source[cfg['features']['speed_indices']],shared])
    if full.shape!=(41,) or speed.shape!=(8,) or not np.isfinite(full).all():raise ValueError('pilot feature dimensions/nonfinite')
    return full,speed



def resolved_label(primary):
    status=primary['label_status']
    if status=='invalid' or status.startswith('invalid_'):return None
    if status not in ('fall','no_fall'):raise ValueError('unresolved primary label')
    value=float(primary['no_fall_through_window'])
    if value!=float(status=='no_fall'):raise ValueError('primary label/status contradiction')
    return value

def load_dataset(analysis,role,cfg,*,released=False,audit_fn=source_audit,confirmation=False,feature_fn=feature_row):
    """Only a development-only analysis is permitted during fitting."""
    analysis=Path(analysis).resolve()
    if role=='new_confirmation' and (not confirmation or analysis.name!='analysis_confirmation'):raise ValueError('explicit confirmation-only artifact required')
    if role=='evaluation' and not released:raise ValueError('held-out labels require sealed selection release')
    if role=='development' and analysis.name!='analysis_development':raise ValueError('development-only artifact required; do not open combined labels')
    root=analysis.parent;plan=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('dataset plan identity changed')
    reviewed=json.loads((analysis/'initial_carry_review.json').read_text())
    if reviewed.get('initial_carry_sha256')!=digest(root/'evaluation/initial_carry.pkl') or not reviewed.get('all_complete_source_copy_leaves_exact'):
        raise ValueError('saved source-state provenance does not match analysis')
    if role=='new_confirmation':validate_confirmation_plan(plan)
    records=json.loads((analysis/'results.json').read_text())
    bank={'training':3,'development':0,'evaluation':1,'new_confirmation':None}[role]
    if role=='development' and any(r.get('bank_index')!=0 for r in records):raise ValueError('held-out contamination in development artifact')
    if role=='new_confirmation':
        if len(records)!=144 or len({r['case_id'] for r in records})!=144 or any(r.get('bank_index') not in (2,5) for r in records):raise ValueError('exact confirmation144 banks2/5 required')
    else:
        records=[r for r in records if r.get('bank_index',3)==bank]
        if len(records)!=72 or len({r['case_id'] for r in records})!=72:raise ValueError('exact bank72 inventory required')
    declared=json.loads((root/'inventory.json').read_text());byid={r['case_id']:i for i,r in enumerate(declared)}
    if role=='new_confirmation':validate_confirmation_inventory(declared,records)
    audit=audit_fn(root)
    with np.load(root/'evaluation/source_features.npz',allow_pickle=False) as data:sources=data['native_observation_user1']
    full=[];speed=[];labels=[];meta=[];inputs={str(analysis/'initial_carry_review.json'):digest(analysis/'initial_carry_review.json'),str(root/'plan.json'):digest(root/'plan.json'),str(analysis/'results.json'):digest(analysis/'results.json'),str(root/'evaluation/initial_carry.pkl'):digest(root/'evaluation/initial_carry.pkl'),str(root/'evaluation/source_features.npz'):digest(root/'evaluation/source_features.npz')}
    if role=='new_confirmation':
        inputs[str(root/'inventory.json')]=digest(root/'inventory.json');inputs[str(root/'seal.json')]=digest(root/'seal.json')
    feature_groups={};candidate_queries={};excluded=[]
    for record in records:
        lane=byid[record['case_id']];case=declared[lane]
        if audit['task_tick'][lane]!=200 or audit['physical_tick'][lane]!=900 or audit['latest_mask'][lane]!=1:raise ValueError('source latest-frame task1 alignment mismatch')
        np.testing.assert_allclose(sources[lane,300:320],audit['latest_frame_normalized'][lane],atol=1e-6,rtol=1e-6)
        primary=record['primary']
        label=resolved_label(primary)
        if label is None:
            excluded.append(dict(case_id=record['case_id'],reason=primary['label_status']));continue
        raw=Path(record['raw_file'])
        if digest(raw)!=record['raw_sha256']:raise ValueError('raw trajectory changed')
        inputs[str(raw)]=record['raw_sha256']
        with np.load(raw,allow_pickle=False) as data:
            if not data['active_tick'][0] or abs(float(data['time'][0])-1.)>1e-6 or not data['reference_target_permitted'][0]:raise ValueError('first prebranch candidate query missing')
            query=data['query_observation'][0];latent=data['composed_latent'][0];yaw=float(data['pre_upper_yaw'][0])
        if abs(math.atan2(math.sin(yaw-float(audit['source_yaw'][lane])),math.cos(yaw-float(audit['source_yaw'][lane]))))>1e-6:raise ValueError('first pre-force yaw mismatch')
        x,s=feature_fn(sources[lane],query,latent,yaw,case,plan['config']['residual_interface'],cfg)
        group=case.get('source_group_id',f"bank3/source{case['source_index']}")
        if group in feature_groups:np.testing.assert_array_equal(sources[lane],feature_groups[group])
        feature_groups[group]=sources[lane]
        pair=(group,case['candidate_alpha'])
        if pair in candidate_queries:np.testing.assert_allclose(latent,candidate_queries[pair],atol=1e-6,rtol=1e-6)
        candidate_queries[pair]=latent
        full.append(x);speed.append(s);labels.append(label);meta.append(dict(case_id=case['case_id'],source_group_id=group,bank_index=case['bank_index'] if role=='new_confirmation' else bank,alpha=case['candidate_alpha'],preparation_s=case['preparation_s'],force_N=case['force_N']))
    if not labels:raise ValueError('no resolved pilot labels')
    return dict(full=np.asarray(full,np.float32),speed=np.asarray(speed,np.float32),y=np.asarray(labels,np.float32),meta=meta,inputs=inputs,excluded_invalid=excluded,audit=dict(latest_frame_is_source_tick200=True,first_query_before_first_force_step=True,source_groups=len(feature_groups),candidate_query_invariant_to_hypothetical_event=True,normalization='native fixed scales/clip; no fitted scaler'))


def metrics(y,p,meta):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if y.shape!=p.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid probabilities')
    q=np.clip(p,1e-7,1-1e-7);pred=p>=.5
    out=dict(n=len(y),positives=int(y.sum()),brier=float(np.mean((p-y)**2)),logloss=float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),confusion=dict(tp=int(np.sum(pred&(y==1))),fp=int(np.sum(pred&(y==0))),tn=int(np.sum(~pred&(y==0))),fn=int(np.sum(~pred&(y==1)))))
    if len(np.unique(y))<2:out.update(roc_auc=None,pr_auc=None,discrimination='not identifiable: single class')
    else:
        pos=p[y==1];neg=p[y==0];out['roc_auc']=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))
        thresholds=np.sort(np.unique(p))[::-1];recall=[0.];precision=[1.]
        for threshold in thresholds:
            hit=p>=threshold;tp=np.sum(hit&(y==1));recall.append(float(tp/len(pos)));precision.append(float(tp/np.sum(hit)))
        out['pr_auc']=float(np.trapezoid(precision,recall));out['pr_auc_definition']='trapezoid threshold PR curve including (recall0,precision1)'
    groups={}
    for i,m in enumerate(meta):groups.setdefault((m['source_group_id'],m['preparation_s'],m['force_N']),[]).append(i)
    score=0.;pairs=0;discordant=0
    for ids in groups.values():
        if len(set(y[ids]))<2:continue
        discordant+=1
        for a in ids:
            for b in ids:
                if y[a]==1 and y[b]==0:pairs+=1;score+=float(p[a]>p[b])+.5*float(p[a]==p[b])
    out['discordant_pair_ranking']=dict(discordant_groups=discordant,pairs=pairs,score_sum=score,mean=None if not pairs else score/pairs)
    return out


def make_model(dim):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(dim,16),torch.nn.ReLU(),torch.nn.Linear(16,16),torch.nn.ReLU(),torch.nn.Linear(16,1))


def predict(model,x,*,feature_mask_keep_indices=None):
    import torch
    model.eval()
    x=apply_feature_mask(x,feature_mask_keep_indices)
    with torch.no_grad():return torch.sigmoid(model(torch.as_tensor(x,dtype=torch.float32)).squeeze(-1)).numpy()


def fit(config,train_analysis,dev_analysis,output):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(1)
    config=Path(config).resolve();cfg=validate_config(json.loads(config.read_text()));output=Path(output).resolve()
    if output.exists():raise ValueError('new pilot output only; no implicit retry')
    train=load_dataset(train_analysis,'training',cfg);dev=load_dataset(dev_analysis,'development',cfg)
    if len(np.unique(train['y']))!=2:raise ValueError('pilot training needs both resolved classes; no automatic resampling')
    output.mkdir(parents=True);writer=SummaryWriter(str(output/'tensorboard'));selected=[];updates=0;per_seed={};inflight=False
    writer.add_scalar('pilot/optimizer_updates',0,0);writer.flush()
    write_json(output/'status.json',dict(phase='initialized',actual_optimizer_updates=0,per_seed=per_seed))
    write_json(output/'data_receipt.json',dict(train={k:v for k,v in train.items() if k not in ('full','speed','y')},development={k:v for k,v in dev.items() if k not in ('full','speed','y')},config_sha256=digest(config)))
    try:
        for name,key,dim in [('full_state_mlp','full',41),('speed_state_ablation','speed',8)]:
            for seed in cfg['training']['seeds']:
                torch.manual_seed(seed);model=make_model(dim);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                x=torch.as_tensor(train[key]);y=torch.as_tensor(train['y']);best=None;rows=[]
                for step in range(1,1001):
                    model.train();optimizer.zero_grad();logit=model(x).squeeze(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,y)
                    if not torch.isfinite(loss):raise ValueError('nonfinite training loss; stop affected fit')
                    loss.backward()
                    if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise ValueError('nonfinite gradient')
                    inflight=True
                    optimizer.step();updates+=1;inflight=False
                    per_seed[f'{name}/seed{seed}']=step
                    write_json(output/'status.json',dict(phase='fitting',model=name,seed=seed,step=step,actual_optimizer_updates=updates,per_seed=per_seed))
                    if updates>6000:raise ValueError('pilot optimizer budget exceeded')
                    row=dict(step=step,train_bce=float(loss.detach()),development_brier=None);writer.add_scalar(f'{name}/seed{seed}/train_bce',row['train_bce'],step)
                    if step%100==0:
                        probabilities=predict(model,dev[key]);brier=float(np.mean((probabilities-dev['y'])**2));row['development_brier']=brier
                        if not math.isfinite(brier):raise ValueError('nonfinite development score')
                        writer.add_scalar(f'{name}/seed{seed}/development_brier',brier,step)
                        if best is None or brier<best['development_brier']:
                            best=dict(step=step,development_brier=brier,state_dict=copy.deepcopy(model.state_dict()))
                    rows.append(row)
                    with (output/f'{name}_seed{seed}_history.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
                path=output/f'{name}_seed{seed}_selected.pt';torch.save(dict(state_dict=best.pop('state_dict'),model=name,seed=seed,dim=dim,**best),path)
                write_json(output/f'{name}_seed{seed}_history.json',rows)
                selected.append(dict(model=name,key=key,seed=seed,dim=dim,file=str(path),sha256=digest(path),**best))
                writer.flush()
        prior=float(np.mean(train['y']))
        manifest=dict(implementation_sha256=digest(__file__),config=str(config),config_sha256=digest(config),selected=selected,training_prior=prior,actual_optimizer_updates=updates,data_receipt_sha256=digest(output/'data_receipt.json'),new_physics_steps=0,new_ppo_updates=0)
        write_json(output/'selection_manifest.json',manifest)
        write_json(output/'selection_seal.json',dict(selection_manifest_sha256=digest(output/'selection_manifest.json'),selected_sha256={r['file']:r['sha256'] for r in selected}))
        for role,data in [('training',train),('development',dev)]:
            report,_=score_selected(manifest,data)
            write_json(output/(role+'_metrics.json'),report)
        write_json(output/'status.json',dict(phase='completed',actual_optimizer_updates=updates,per_seed=per_seed,selection_sealed=True))
    except BaseException as exc:
        write_json(output/'status.json',dict(phase='error',actual_optimizer_updates=updates,per_seed=per_seed,optimizer_step_may_be_partial=inflight,error=repr(exc),selection_sealed=(output/'selection_seal.json').exists()))
        raise
    finally:writer.close()
    return manifest


def verify_selection(output):
    output=Path(output);seal=json.loads((output/'selection_seal.json').read_text());manifest=json.loads((output/'selection_manifest.json').read_text())
    if digest(output/'selection_manifest.json')!=seal['selection_manifest_sha256'] or len(manifest['selected'])!=6:raise ValueError('six-model sealed selection required')
    if {(r['model'],r['seed']) for r in manifest['selected']}!={(m,s) for m in ('full_state_mlp','speed_state_ablation') for s in (0,1,2)}:raise ValueError('selected model identities mismatch')
    if manifest['implementation_sha256']!=digest(__file__):raise ValueError('sealed predictor implementation changed')
    for r in manifest['selected']:
        if digest(r['file'])!=r['sha256'] or seal['selected_sha256'].get(r['file'])!=r['sha256']:raise ValueError('selected checkpoint changed')
    if digest(manifest['config'])!=manifest['config_sha256'] or digest(output/'data_receipt.json')!=manifest['data_receipt_sha256']:raise ValueError('selected config/data receipt changed')
    return manifest


def evaluate(output,heldout_analysis):
    import torch
    output=Path(output);manifest=verify_selection(output)  # MUST precede label read.
    cfg=validate_config(json.loads(Path(manifest['config']).read_text()))
    if (output/'heldout_metrics.json').exists():raise ValueError('held-out evaluation already recorded; no overwrite')
    data=load_dataset(heldout_analysis,'evaluation',cfg,released=True)
    report,predictions=score_selected(manifest,data)
    write_json(output/'heldout_metrics.json',report);write_json(output/'heldout_predictions.json',[dict(m,label=int(y),probabilities={k:float(v[i]) for k,v in predictions.items()}) for i,(m,y) in enumerate(zip(data['meta'],data['y']))])
    write_json(output/'heldout_data_receipt.json',dict(inputs=data['inputs'],audit=data['audit'],excluded_invalid=data['excluded_invalid'],selection_seal_sha256=digest(output/'selection_seal.json')))
    return report


def score_selected(manifest,data):
    import torch
    report={};predictions={}
    for r in manifest['selected']:
        checkpoint=torch.load(r['file'],map_location='cpu',weights_only=True);model=make_model(r['dim']);model.load_state_dict(checkpoint['state_dict'])
        probability=predict(model,data[r['key']]);name=f"{r['model']}/seed{r['seed']}";predictions[name]=probability;report[name]=metrics(data['y'],probability,data['meta'])
    for name in ('full_state_mlp','speed_state_ablation'):
        probability=np.mean([predictions[f'{name}/seed{s}'] for s in (0,1,2)],axis=0);predictions[name+'/ensemble']=probability;report[name+'/ensemble']=metrics(data['y'],probability,data['meta'])
    prior=np.full(len(data['y']),manifest['training_prior']);report['training_prior']=metrics(data['y'],prior,data['meta']);predictions['training_prior']=prior
    return report,predictions


def validate_ranking_config(cfg):
    validate_config(cfg)
    if (cfg['models']!=['full_state_mlp'] or cfg['architecture']['activation']!='ReLU' or
        cfg['training']['full_batch'] is not True or cfg['training']['precision']!='float32' or
        cfg['ranking']!=dict(arms=['bce','bce_rank'],weight=.5,selectors=['brier','rank'],pairing='source_group_id/preparation_s/force_N; positive-negative only') or
        cfg['budget']!=dict(new_physics_steps=0,new_ppo_updates=0,maximum_optimizer_updates=6000,automatic_extensions=False) or
        cfg['evaluation'].get('release')!='none; development-only experiment' or 'evaluation' in cfg['data']):
        raise ValueError('frozen ranking configuration mismatch')
    return cfg


def discordant_pairs(y,meta):
    """Positive/negative row indices only within the exact hypothetical condition."""
    groups={}
    for i,m in enumerate(meta):groups.setdefault((m['source_group_id'],m['preparation_s'],m['force_N']),[]).append(i)
    pairs=[(a,b) for ids in groups.values() for a in ids for b in ids if y[a]==1 and y[b]==0]
    return np.asarray(pairs,dtype=np.int64).reshape(-1,2)


def ranking_loss(logits,pairs):
    import torch
    if len(pairs)==0:return logits.sum()*0.
    indices=torch.as_tensor(pairs,dtype=torch.long,device=logits.device)
    return torch.nn.functional.softplus(-(logits[indices[:,0]]-logits[indices[:,1]])).mean()


def selector_key(selector,brier,rank,step):
    if not math.isfinite(brier) or (rank is not None and not math.isfinite(rank)):raise ValueError('nonfinite selection score')
    if selector=='brier':return (brier,step)
    if selector=='rank':return (-(rank if rank is not None else -1.),brier,step)
    raise ValueError('unknown selector')


def ranking_fit(config,train_analysis,dev_analysis,output,*,_prepared=None,on_training_complete=None):
    """Fixed scratch CPU comparison; never reads or releases bank1 labels."""
    import torch
    from torch.utils.tensorboard import SummaryWriter
    started=time.monotonic();timings=dict(initialization_s=0.,sampling_s=0.,optimization_s=0.,validation_s=0.)
    config=Path(config).resolve();temporal=_prepared is not None
    raw_cfg=json.loads(config.read_text());user_mix='user_mix' in raw_cfg
    cfg=(validate_user_mix_config if user_mix else validate_temporal_config if temporal else validate_ranking_config)(raw_cfg)
    # Enforce the development boundary before even opening the training dataset.
    if not temporal and Path(dev_analysis).name!='analysis_development':raise ValueError('development-only artifact required')
    output=Path(output).resolve()
    if output.exists():raise ValueError('new ranking output only; no implicit retry')
    torch.set_num_threads(1)
    if temporal:train,dev=_prepared['training'],_prepared['development']
    else:train=load_dataset(train_analysis,'training',cfg);dev=load_dataset(dev_analysis,'development',cfg)
    expected=192 if user_mix else 96 if temporal else 72
    if len(train['y'])!=expected or len(dev['y'])!=expected or len(np.unique(train['y']))!=2:raise ValueError('exact resolved training/development inventory required')
    mask=cfg.get('feature_mask_keep_indices')
    mask_metadata={} if mask is None else dict(feature_mask_keep_indices=mask,input_information='speed plus candidate/disturbance query; offline-only, not pure speed')
    if mask is not None:validate_matched_full_prior(cfg)
    selectors=cfg['ranking']['selectors']
    if temporal:(validate_user_mix_datasets if user_mix else validate_temporal_datasets)(train,dev)
    output.mkdir(parents=True);updates=0;inflight=False;selected=[];progress={}
    write_json(output/'data_receipt.json',dict(train={k:v for k,v in train.items() if k not in ('full','speed','y')},development={k:v for k,v in dev.items() if k not in ('full','speed','y')}))
    write_json(output/'resolved_config.json',cfg)
    snapshot=output/'source';snapshot.mkdir();(snapshot/'tolerance_predictor_pilot.py').write_bytes(Path(__file__).read_bytes())
    pairs=discordant_pairs(train['y'],train['meta']);x=torch.as_tensor(apply_feature_mask(train['full'],mask));y=torch.as_tensor(train['y'])
    sample_weights=torch.as_tensor(user_mix_bce_weights(train['meta'],cfg['bce_group_weights'])) if user_mix and 'bce_group_weights' in cfg else None
    if temporal and any(train['meta'][i]['period']!='early' for pair in pairs for i in pair):raise ValueError('late ranking pairs forbidden by frozen all-positive dataset')
    timings['initialization_s']=time.monotonic()-started
    if temporal:write_json(output/'preparation_receipt.json',_prepared['receipt'])
    write_json(output/'status.json',dict(phase='initialized',actual_optimizer_updates=0,per_seed=progress))
    try:
        for seed in (0,1,2):
            torch.manual_seed(seed);initial=copy.deepcopy(make_model(41).state_dict())
            if mask is not None:
                matched=torch.load(Path(cfg['matched_full_prior_training'])/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
                if set(initial)!=set(matched) or any(not torch.equal(v,matched[k]) for k,v in initial.items()):raise ValueError('same-seed initial tensors differ from matched full prior')
                initial=copy.deepcopy(matched)
            init_file=output/f'initial_seed{seed}.pt';torch.save(initial,init_file)
            for arm in cfg['ranking']['arms']:
                initialized=time.monotonic()
                model=make_model(41);model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                folder=output/arm/f'seed{seed}';folder.mkdir(parents=True);writer=SummaryWriter(str(Path(_prepared['tensorboard_root'])/arm/f'seed{seed}' if temporal else folder/'tensorboard'));best={};rows=[]
                writer.add_scalar('train/optimizer_updates',0,0);writer.flush()
                timings['initialization_s']+=time.monotonic()-initialized
                try:
                    for step in range(1,1001):
                        if updates>=cfg['budget']['maximum_optimizer_updates']:raise ValueError('ranking optimizer budget exhausted')
                        optimized=time.monotonic()
                        model.train();optimizer.zero_grad();logits=model(x).squeeze(-1)
                        bce,weighted_bce=bce_objective(logits,y,sample_weights);rank=ranking_loss(logits,pairs)
                        loss=weighted_bce+(.5 if arm=='bce_rank' else 0.)*rank
                        if not torch.isfinite(loss):raise ValueError('nonfinite ranking loss')
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite ranking gradient')
                        inflight=True;optimizer.step();updates+=1;inflight=False
                        timings['optimization_s']+=time.monotonic()-optimized
                        row=dict(step=step,bce=float(bce.detach()),weighted_bce=float(weighted_bce.detach()),rank_loss=float(rank.detach()),total_loss=float(loss.detach()),development_brier=None,development_rank=None)
                        for key in ('bce','weighted_bce','rank_loss','total_loss'):writer.add_scalar('train/'+key,row[key],step)
                        if step%100==0:
                            validated=time.monotonic();dev_p=predict(model,dev['full'],feature_mask_keep_indices=mask)
                            score=metrics(dev['y'],dev_p,dev['meta']);db=score['brier'];dr=score['discordant_pair_ranking']['mean']
                            selection_score=db
                            if temporal:
                                selection_score,period_scores=(user_mix_selection if user_mix else temporal_selection)(dev['y'],dev_p,dev['meta'])
                                period_scores.update(temporal_diagnostics(dev['y'],dev_p,dev['meta']))
                                row.update(development_selection_score=selection_score,**period_scores)
                                for metric,value in dict(selection_score=selection_score,**period_scores).items():writer.add_scalar('development/'+metric,value,step)
                            writer.add_scalar('train/optimizer_updates',step,step)
                            row.update(development_brier=db,development_rank=dr);writer.add_scalar('development/brier',db,step)
                            if dr is not None:writer.add_scalar('development/rank',dr,step)
                            for selector in selectors:
                                if selector=='last' and step!=1000:continue
                                key=selector_key('brier' if selector in ('balanced_brier','last') else selector,selection_score,dr,step)
                                if selector not in best or key<best[selector]['key']:
                                    best[selector]=dict(key=key,step=step,development_brier=db,development_rank=dr,development_selection_score=selection_score,state_dict=copy.deepcopy(model.state_dict()))
                            timings['validation_s']+=time.monotonic()-validated
                            progress[f'{arm}/seed{seed}']=step
                            write_json(output/'status.json',dict(phase='fitting',arm=arm,seed=seed,step=step,actual_optimizer_updates=updates,per_seed=progress,timings=timings));writer.flush()
                        rows.append(row)
                        with (folder/'history.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
                    write_json(folder/'history.json',rows)
                    if temporal:torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=seed,step=1000,role='last; not selected',**mask_metadata),folder/'last.pt')
                    for selector,record in best.items():
                        state=record.pop('state_dict');record.pop('key');path=folder/f'{selector}_selected.pt'
                        torch.save(dict(state_dict=state,arm=arm,seed=seed,selector=selector,dim=41,**record,**mask_metadata),path)
                        selected.append(dict(arm=arm,seed=seed,selector=selector,dim=41,file=str(path),sha256=digest(path),initial_sha256=digest(init_file),**record,**mask_metadata))
                finally:writer.close()
        manifest=dict(experiment_id=cfg['experiment_id'],implementation_sha256=digest(__file__),source_snapshot_sha256=digest(snapshot/'tolerance_predictor_pilot.py'),config=str(config),config_sha256=digest(config),resolved_config_sha256=digest(output/'resolved_config.json'),data_receipt_sha256=digest(output/'data_receipt.json'),selected=selected,actual_optimizer_updates=updates,new_physics_steps=0,new_ppo_updates=0,heldout_release=False)
        if mask is not None:manifest.update(feature_mask_keep_indices=mask,matched_full_prior_training=cfg['matched_full_prior_training'],matched_full_prior_manifest_sha256=cfg['matched_full_prior_manifest_sha256'],input_information=mask_metadata['input_information'])
        if user_mix:manifest.update(checkpoint_rule='fixed last1000 for all three seeds; no development selection',data_role='all seen engineering DEV; no independent TEST')
        write_json(output/'selection_manifest.json',manifest)
        if temporal:
            write_json(output/'status.json',dict(phase='training_completed',stage='training_completed',actual_optimizer_updates=updates,per_seed=progress,timings=timings))
            write_json(output.parent/'status.json',dict(phase='analysis_running',stage='training_completed',actual_optimizer_updates=updates))
            if on_training_complete is not None:
                try:
                    on_training_complete(dict(stage='training_completed',actual_optimizer_updates=updates,output=str(output)))
                    write_json(output/'training_completion_notification.json',dict(callback_called=True,callback_error=None))
                except Exception as notice_error:write_json(output/'training_completion_notification.json',dict(callback_called=True,callback_error=repr(notice_error)))
        analysis_started=time.monotonic()
        for role,data in [('training',train),('development',dev)]:
            scores={};predictions={}
            for r in selected:
                checkpoint=torch.load(r['file'],map_location='cpu',weights_only=True);model=make_model(41);model.load_state_dict(checkpoint['state_dict'])
                name=f"{r['arm']}/{r['selector']}/seed{r['seed']}";p=predict(model,data['full'],feature_mask_keep_indices=checkpoint.get('feature_mask_keep_indices'));predictions[name]=p;scores[name]=metrics(data['y'],p,data['meta'])
            for arm in cfg['ranking']['arms']:
                for selector in selectors:
                    name=f'{arm}/{selector}/ensemble';p=np.mean([predictions[f'{arm}/{selector}/seed{s}'] for s in (0,1,2)],axis=0);predictions[name]=p;scores[name]=metrics(data['y'],p,data['meta'])
            if temporal:
                probabilities=user_mix_references(data,_prepared['q_export'],full_prior_training=cfg.get('matched_full_prior_training')) if user_mix else temporal_references(data,_prepared['q_export'])
                probabilities.update(predictions)
                for arm in cfg['ranking']['arms']:
                    for seed in (0,1,2):
                        last=torch.load(output/arm/f'seed{seed}'/'last.pt',map_location='cpu',weights_only=True);model=make_model(41);model.load_state_dict(last['state_dict'])
                        probabilities[f'{arm}/last/seed{seed}']=predict(model,data['full'],feature_mask_keep_indices=last.get('feature_mask_keep_indices'))
                    probabilities[f'{arm}/last/ensemble']=np.mean([probabilities[f'{arm}/last/seed{s}'] for s in (0,1,2)],axis=0)
                write_json(output/f'{role}_temporal_metrics.json',temporal_metrics(data,probabilities))
                write_json(output/f'{role}_temporal_predictions.json',[dict(m,label=int(v),probabilities={k:float(p[i]) for k,p in probabilities.items()}) for i,(m,v) in enumerate(zip(data['meta'],data['y']))])
            write_json(output/f'{role}_metrics.json',scores)
            write_json(output/f'{role}_predictions.json',[dict(m,label=int(v),probabilities={k:float(p[i]) for k,p in predictions.items()}) for i,(m,v) in enumerate(zip(data['meta'],data['y']))])
        timings['analysis_s']=time.monotonic()-analysis_started
        timings['total_s']=time.monotonic()-started
        write_json(output/'timings.json',timings)
        artifacts=[p for p in output.rglob('*') if p.is_file() and p.name!='status.json']
        write_json(output/'selection_seal.json',dict(artifacts_sha256={str(p.relative_to(output)):digest(p) for p in artifacts}))
        write_json(output/'status.json',dict(phase='completed',actual_optimizer_updates=updates,per_seed=progress,selection_sealed=True,heldout_release=False,timings=timings,stage='training_completed' if temporal else 'completed'))
        if temporal:write_json(output.parent/'status.json',dict(phase='completed',stage='completed',actual_optimizer_updates=updates,analysis_completed=True))
        return manifest
    except BaseException as exc:
        write_json(output/'status.json',dict(phase='error',actual_optimizer_updates=updates,per_seed=progress,optimizer_step_may_be_partial=inflight,error=repr(exc),heldout_release=False));raise


def validate_confirmation_plan(plan):
    protocol=plan['protocol']
    groups=protocol['source_groups']
    if (plan.get('mode')!='h75_samestate_bank144' or protocol.get('source_initialization')!='prepared_bank_all_prefixes' or
        protocol.get('bank_indices')!=[2,5] or len(groups)!=8 or
        len({g['source_group_id'] for g in groups})!=8 or
        any(g.get('split')!='new_confirmation' or g.get('bank_index') not in (2,5) for g in groups)):
        raise ValueError('new confirmation plan identity mismatch')


def verify_confirmation_candidates(manifest_path):
    """Authenticate the preselected six models and frozen implementation before data."""
    import importlib.util
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    manifest=json.loads(manifest_path.read_text());source_path=root/'selection_manifest.json'
    if digest(source_path)!=manifest['ranking_source_manifest_sha256']:raise ValueError('ranking source manifest changed')
    source=json.loads(source_path.read_text());snapshot=root/'source/tolerance_predictor_pilot.py'
    if digest(snapshot)!=source['implementation_sha256'] or digest(snapshot)!=source['source_snapshot_sha256']:raise ValueError('frozen ranking source changed')
    expected={(a,s,k) for a,k in [('bce','rank'),('bce_rank','brier')] for s in (0,1,2)}
    candidates=manifest['candidate_models']
    if (len(candidates)!=6 or {(r['arm'],r['seed'],r['selector']) for r in candidates}!=expected or
        manifest['bank_indices']!=[2,5] or manifest['training_prior']!=57/72 or manifest['new_optimizer_updates']!=0):raise ValueError('frozen confirmation candidates mismatch')
    for r in candidates:
        if r not in source['selected'] or r['dim']!=41 or digest(r['file'])!=r['sha256']:raise ValueError('confirmation checkpoint changed')
    if digest(source['config'])!=source['config_sha256']:raise ValueError('ranking config changed')
    spec=importlib.util.spec_from_file_location('_frozen_confirmation_predictor',snapshot);frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    return manifest,source,frozen


def grouped_confirmation_metrics(y,p,meta,metric_fn=metrics):
    result={'overall':metric_fn(y,p,meta),'per_bank':{},'per_source':{}}
    for key,field in [('per_bank','bank_index'),('per_source','source_group_id')]:
        for value in sorted({m[field] for m in meta}):
            ids=[i for i,m in enumerate(meta) if m[field]==value]
            result[key][str(value)]=metric_fn(np.asarray(y)[ids],np.asarray(p)[ids],[meta[i] for i in ids])
    return result


def confirmation_evaluate(manifest_path,analysis,output):
    import torch
    manifest,source,frozen=verify_confirmation_candidates(manifest_path) # MUST precede label access.
    output=Path(output).resolve()
    if output.exists():raise ValueError('new confirmation output only; no overwrite')
    cfg=json.loads(Path(source['config']).read_text())
    data=load_dataset(analysis,'new_confirmation',cfg,confirmation=True,audit_fn=frozen.source_audit,feature_fn=frozen.feature_row)
    if len(data['meta'])+len(data['excluded_invalid'])!=144:raise ValueError('confirmation inventory incomplete')
    predictions={};report={}
    for r in manifest['candidate_models']:
        model=frozen.make_model(41);model.load_state_dict(torch.load(r['file'],map_location='cpu',weights_only=True)['state_dict'])
        name=f"{r['arm']}/{r['selector']}/seed{r['seed']}";predictions[name]=frozen.predict(model,data['full'])
    for arm,selector in [('bce','rank'),('bce_rank','brier')]:
        predictions[f'{arm}/{selector}/ensemble']=np.mean([predictions[f'{arm}/{selector}/seed{s}'] for s in (0,1,2)],axis=0)
    predictions['original_training_prior']=np.full(len(data['y']),57/72)
    for name,p in predictions.items():report[name]=grouped_confirmation_metrics(data['y'],p,data['meta'],frozen.metrics)
    output.mkdir(parents=True)
    write_json(output/'metrics.json',report)
    write_json(output/'predictions.json',[dict(m,label=int(v),probabilities={k:float(p[i]) for k,p in predictions.items()}) for i,(m,v) in enumerate(zip(data['meta'],data['y']))])
    write_json(output/'data_receipt.json',dict(inputs=data['inputs'],audit=data['audit'],excluded_invalid=data['excluded_invalid'],candidate_manifest_sha256=digest(manifest_path),frozen_source_sha256=source['implementation_sha256'],evaluator_sha256=digest(__file__),new_optimizer_updates=0,new_physics_steps=0))
    write_json(output/'seal.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    return report


def validate_confirmation_inventory(declared,records):
    if (len(declared)!=144 or len({c['case_id'] for c in declared})!=144 or
        {c['case_id'] for c in declared}!={r['case_id'] for r in records} or
        len({c['source_group_id'] for c in declared})!=8 or
        any(c.get('split')!='new_confirmation' or c.get('bank_index') not in (2,5) for c in declared)):
        raise ValueError('confirmation inventory split/bank mismatch')
    byid={c['case_id']:c for c in declared}
    if any(r['bank_index']!=byid[r['case_id']]['bank_index'] for r in records):raise ValueError('confirmation record bank mismatch')


def validate_temporal_config(cfg):
    validate_config(cfg)
    if (cfg['models']!=['full_state_mlp'] or cfg['architecture']['activation']!='ReLU' or
        cfg['training']['full_batch'] is not True or cfg['training']['precision']!='float32' or
        cfg['ranking']!=dict(arms=['bce','bce_rank'],weight=.5,selectors=['balanced_brier'],pairing='source_group_id/preparation_s/force_N; positive-negative only') or
        cfg['temporal_mix']!=dict(train_early_bank=3,train_late_bank=2,dev_early_bank=0,dev_late_bank=5,early_rows=72,late_rows=24,late_labels='all positive; no ranking pairs',selection='0.5*early_dev_brier + 0.5*late_dev_brier; earlier step ties',bce_weighting='equal per sample among all96; no class weights',role='seen engineering DEV; not independent TEST') or
        cfg['budget']!=dict(new_physics_steps=0,new_ppo_updates=0,maximum_optimizer_updates=6000,automatic_extensions=False) or
        cfg['evaluation'].get('release')!='none; development-only experiment' or 'evaluation' in cfg['data']):
        raise ValueError('frozen temporal mix configuration mismatch')
    return cfg


def temporal_selection(y,p,meta):
    values={}
    for period in ('early','late'):
        ids=[i for i,m in enumerate(meta) if m['period']==period]
        if not ids:raise ValueError('both time periods required for selection')
        value=float(np.mean((np.asarray(p)[ids]-np.asarray(y)[ids])**2))
        if not math.isfinite(value):raise ValueError('nonfinite temporal selection score')
        values[period+'_brier']=value
    return .5*(values['early_brier']+values['late_brier']),values


def validate_temporal_datasets(train,dev):
    for data,early_bank,late_bank in [(train,3,2),(dev,0,5)]:
        if data['full'].shape!=(96,41) or data['y'].shape!=(96,) or len(data['meta'])!=96 or not np.isfinite(data['full']).all() or not np.isin(data['y'],[0,1]).all():raise ValueError('finite temporal96/full41 required')
        early=[i for i,m in enumerate(data['meta']) if m['period']=='early'];late=[i for i,m in enumerate(data['meta']) if m['period']=='late']
        if len(early)!=72 or len(late)!=24 or any(data['meta'][i]['bank_index']!=early_bank for i in early) or any(data['meta'][i]['bank_index']!=late_bank for i in late):raise ValueError('temporal ancestor bank split mismatch')
        if not np.all(data['y'][late]==1) or len(np.unique(data['y'][early]))!=2:raise ValueError('frozen early mixed/late positive labels required')
        if len({m['case_id'] for m in data['meta']})!=96:raise ValueError('duplicate temporal case')
    if {m['source_group_id'] for m in train['meta']}&{m['source_group_id'] for m in dev['meta']}:raise ValueError('shared train/development source ancestor')


def load_late_features(evaluation):
    evaluation=Path(evaluation).resolve()
    seal=json.loads((evaluation/'seal.json').read_text())
    for name,expected in seal.items():
        if digest(evaluation/name)!=expected:raise ValueError('late evaluation seal changed')
    for name in ('features_predictions.npz','predictions.json','receipt.json'):
        if name not in seal:raise ValueError('late evaluation seal missing data')
    receipt=json.loads((evaluation/'receipt.json').read_text())
    if receipt['source_task_s']!=6. or receipt['resolved_rows']!=48 or receipt['excluded_invalid'] or receipt['new_optimizer_updates']!=0:raise ValueError('complete task6 late48 evaluation required')
    rows=json.loads((evaluation/'predictions.json').read_text())
    with np.load(evaluation/'features_predictions.npz',allow_pickle=False) as z:features=z['features'].copy();labels=z['labels'].copy()
    if features.shape!=(48,41) or labels.shape!=(48,) or len(rows)!=48 or not np.isfinite(features).all() or not np.all(labels==1):raise ValueError('frozen positive late48 required')
    meta=[]
    for row,y in zip(rows,labels):
        if row['bank_index'] not in (2,5) or row['label']!=y:raise ValueError('late features/labels metadata mismatch')
        meta.append({k:row[k] for k in ('case_id','source_group_id','bank_index','alpha','preparation_s','force_N')})
    if len({m['case_id'] for m in meta})!=48:raise ValueError('duplicate late sample identity')
    expected={(a,p,f) for a in (0.,.5,1.) for p in (.2,.6) for f in (-64.,64.)}
    for group in {m['source_group_id'] for m in meta}:
        group_rows=[m for m in meta if m['source_group_id']==group]
        if len(group_rows)!=12 or {(m['alpha'],m['preparation_s'],m['force_N']) for m in group_rows}!=expected:raise ValueError('complete late group12 inventory required')
    for bank in (2,5):
        ids=[i for i,m in enumerate(meta) if m['bank_index']==bank]
        if len(ids)!=24 or len({meta[i]['source_group_id'] for i in ids})!=2:raise ValueError('late ancestor split inventory mismatch')
    return dict(full=features,y=labels,meta=meta),dict(evaluation=str(evaluation),seal_sha256=digest(evaluation/'seal.json'),receipt_sha256=digest(evaluation/'receipt.json'))


def prepare_temporal_mix(config,train_analysis,dev_analysis,late_evaluation,q_export,output):
    from torch.utils.tensorboard import SummaryWriter
    from sttw_control.tolerance_coordination import load_export
    started=time.monotonic();config=Path(config).resolve();cfg=validate_temporal_config(json.loads(config.read_text()))
    if Path(dev_analysis).name!='analysis_development':raise ValueError('development-only artifact required before data access')
    output=Path(output).resolve()
    if output.exists():raise ValueError('fresh temporal preparation only')
    load_export(q_export) # Authenticate the unmodified old comparator bundle.
    early_train=load_dataset(train_analysis,'training',cfg);early_dev=load_dataset(dev_analysis,'development',cfg)
    late,late_receipt=load_late_features(late_evaluation)
    datasets={}
    for role,early,bank in [('training',early_train,2),('development',early_dev,5)]:
        ids=[i for i,m in enumerate(late['meta']) if m['bank_index']==bank]
        datasets[role]=dict(full=np.concatenate([early['full'],late['full'][ids]]),y=np.concatenate([early['y'],late['y'][ids]]),meta=[dict(m,period='early') for m in early['meta']]+[dict(late['meta'][i],period='late') for i in ids],inputs=early['inputs'],excluded_invalid=early['excluded_invalid'],audit=early['audit'])
    validate_temporal_datasets(datasets['training'],datasets['development'])
    output.mkdir(parents=True)
    for role,data in datasets.items():
        np.savez(output/(role+'.npz'),full=data['full'],y=data['y'])
        write_json(output/(role+'.json'),{k:v for k,v in data.items() if k not in ('full','y')})
    tensorboard=output.parent/'tensorboard'
    for arm in ('bce','bce_rank'):
        for seed in (0,1,2):
            with SummaryWriter(str(tensorboard/arm/f'seed{seed}')) as writer:writer.add_scalar('train/optimizer_updates',0,0);writer.flush()
    receipt=dict(config=str(config),config_sha256=digest(config),implementation_sha256=digest(__file__),q_export=str(Path(q_export).resolve()),q_receipt_sha256=digest(Path(q_export)/'receipt.json'),tensorboard_root=str(tensorboard),late=late_receipt,preparation_s=time.monotonic()-started,new_optimizer_updates=0,new_physics_steps=0,roles=cfg['temporal_mix'])
    write_json(output/'receipt.json',receipt)
    write_json(output/'seal.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write_json(output.parent/'status.json',dict(phase='prepared',stage='prepared',actual_optimizer_updates=0,tensorboard_root=str(tensorboard)))
    return receipt


def temporal_mix_fit(config,prepared,output,*,on_training_complete=None):
    config=Path(config).resolve();validate_temporal_config(json.loads(config.read_text()));prepared=Path(prepared).resolve()
    seal=json.loads((prepared/'seal.json').read_text())
    for name,expected in seal.items():
        if digest(prepared/name)!=expected:raise ValueError('temporal preparation changed')
    if any(name not in seal for name in ('training.npz','development.npz','training.json','development.json','receipt.json')):raise ValueError('temporal preparation seal incomplete')
    receipt=json.loads((prepared/'receipt.json').read_text())
    if receipt['config_sha256']!=digest(config) or receipt['implementation_sha256']!=digest(__file__):raise ValueError('prepared temporal configuration/source changed')
    if digest(Path(receipt['q_export'])/'receipt.json')!=receipt['q_receipt_sha256']:raise ValueError('old comparator identity changed')
    data=dict(receipt=receipt,q_export=receipt['q_export'],tensorboard_root=receipt['tensorboard_root'])
    for role in ('training','development'):
        with np.load(prepared/(role+'.npz'),allow_pickle=False) as z:data[role]=dict(full=z['full'].copy(),y=z['y'].copy(),**json.loads((prepared/(role+'.json')).read_text()))
    validate_temporal_datasets(data['training'],data['development'])
    return ranking_fit(config,None,None,output,_prepared=data,on_training_complete=on_training_complete)


def temporal_references(data,q_export):
    import torch
    from sttw_control.tolerance_coordination import load_export
    load_export(q_export);q_export=Path(q_export);predictions={}
    for arm in ('bce','bce_rank'):
        probabilities=[]
        for seed in (0,1,2):
            model=make_model(41);model.load_state_dict(torch.load(q_export/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);probabilities.append(predict(model,data['full']))
        predictions['old_frozen_'+arm+'/ensemble']=np.mean(probabilities,axis=0)
    predictions['original_training_prior']=np.full(len(data['y']),57/72)
    return predictions


def temporal_metrics(data,probabilities):
    from sttw_control.tolerance_late_evaluation import rejection_report
    report={}
    for name,p in probabilities.items():
        report[name]={}
        for period in sorted({m['period'] for m in data['meta']}):
            ids=[i for i,m in enumerate(data['meta']) if m['period']==period];meta=[data['meta'][i] for i in ids];y=data['y'][ids];q=p[ids]
            gate=np.array([m['preparation_s'] in (.2,.6) for m in meta],bool)
            report[name][period]=dict(proper_and_ranking=metrics(y,q,meta),user_rejection=rejection_report(y[gate],q[gate],[m for m,keep in zip(meta,gate) if keep]),early_fall_missed_at_05=int(np.sum((y==0)&(q>=.5))) if period=='early' else None,fall_not_rejected_at_08=int(np.sum((y==0)&(q>=.8))),fall_not_rejected_at_085=int(np.sum((y==0)&(q>=.85))),falls=int(np.sum(y==0)))
    return report


def temporal_diagnostics(y,p,meta):
    y=np.asarray(y);p=np.asarray(p);values={}
    for period in sorted({m['period'] for m in meta}):
        mask=np.array([m['period']==period for m in meta]);user=mask&np.array([m['alpha']==m.get('source_user_alpha',1.) and m['preparation_s'] in (.2,.6) for m in meta])
        values[period+'_falls']=int(np.sum(mask&(y==0)))
        values[period+'_falls_missed_05']=int(np.sum(mask&(y==0)&(p>=.5)))
        values[period+'_user_rows']=int(user.sum())
        values[period+'_falls_not_rejected_08']=int(np.sum(mask&(y==0)&(p>=.8)))
        values[period+'_falls_not_rejected_085']=int(np.sum(mask&(y==0)&(p>=.85)))
        for threshold,name in [(.8,'08'),(.85,'085')]:values[period+'_user_rejected_'+name]=int(np.sum(user&(p<threshold)))
    return values


def validate_user_mix_config(cfg):
    if 'feature_mask_keep_indices' in cfg:
        if cfg['feature_mask_keep_indices']!=[4]+list(range(34,41)) or cfg.get('bce_group_weights')!={'early':.75,'late':.125,'userlate':.125} or not cfg.get('matched_full_prior_training') or not cfg.get('matched_full_prior_manifest_sha256'):raise ValueError('only declared matched speed+query mask allowed')
    checked=copy.deepcopy(cfg);checked['training']['maximum_total_optimizer_updates']=6000
    validate_config(checked)
    if 'bce_group_weights' in cfg and cfg['bce_group_weights']!={'early':.75,'late':.125,'userlate':.125}:raise ValueError('only preregistered prior weight variant allowed')
    if (cfg['models']!=['full_state_mlp'] or cfg['architecture']['activation']!='ReLU' or cfg['training']['full_batch'] is not True or cfg['training']['precision']!='float32' or cfg['training']['maximum_total_optimizer_updates']!=3000 or cfg['training']['selection']!='fixed last1000; no development selection' or
        cfg['ranking']!=dict(arms=['bce_rank'],weight=.5,selectors=['last'],pairing='source_group_id/preparation_s/force_N; positive-negative only') or
        cfg['budget']!=dict(new_physics_steps=0,new_ppo_updates=0,maximum_optimizer_updates=3000,automatic_extensions=False) or
        cfg['user_mix']!=dict(train_user_bank=4,dev_user_bank=6,early_rows=72,oldlate_rows=24,userlate_rows=96,bce_weighting=('group totals early=.75 oldlate=.125 userlate=.125; uniform within group' if 'bce_group_weights' in cfg else 'equal per sample among all192; no class weights'),selection='fixed last1000; no development selection',role='all seen engineering DEV; not independent TEST')):
        raise ValueError('fixed user mix contract mismatch')
    return cfg


def validate_user_mix_datasets(train,dev):
    for data,early_bank,late_bank,user_bank,falls in [(train,3,2,4,15),(dev,0,5,6,8)]:
        if data['full'].shape!=(192,41) or data['y'].shape!=(192,) or len(data['meta'])!=192 or not np.isfinite(data['full']).all() or not np.isin(data['y'],[0,1]).all():raise ValueError('finite complete user mix192 required')
        if len({m['case_id'] for m in data['meta']})!=192:raise ValueError('duplicate user mix case')
        for period,count,bank in [('early',72,early_bank),('late',24,late_bank),('userlate',96,user_bank)]:
            ids=[i for i,m in enumerate(data['meta']) if m['period']==period]
            if len(ids)!=count or any(data['meta'][i]['bank_index']!=bank for i in ids):raise ValueError('user mix source bank/period split differs')
            if period!='early' and not np.all(data['y'][ids]==1):raise ValueError('late labels differ from declared real inventory')
            if period=='early' and np.sum(data['y'][ids]==0)!=falls:raise ValueError('early failures must be retained')
        groups={}
        for m in data['meta']:
            if m['period']=='userlate':groups.setdefault(m['source_group_id'],[]).append(m)
        expected={(a,p,f) for a in (0.,.5,1.) for p in (.2,.6) for f in (-64.,64.)}
        expected_groups={f'bank{user_bank}/{route}/{origin}/task6' for route in ('S_left_90_R1p8','S_right_90_R1p8','S_left_60_R2p2','S_right_60_R2p2') for origin in ('user0_last','userhalf_last')}
        if set(groups)!=expected_groups:raise ValueError('user late original source group identity differs')
        if len(groups)!=8 or any(len(rows)!=12 or {(m['alpha'],m['preparation_s'],m['force_N']) for m in rows}!=expected for rows in groups.values()):raise ValueError('all user late source12 rows required')


def _sealed_prepared_data(prepared):
    prepared=Path(prepared).resolve();seal=json.loads((prepared/'seal.json').read_text())
    for name in ('training.npz','development.npz','training.json','development.json','receipt.json'):
        if name not in seal:raise ValueError('prepared seal incomplete')
    for name,sha in seal.items():
        if digest(prepared/name)!=sha:raise ValueError('prepared dataset changed')
    datasets={}
    for role in ('training','development'):
        with np.load(prepared/(role+'.npz'),allow_pickle=False) as z:datasets[role]=dict(full=z['full'].copy(),y=z['y'].copy(),**json.loads((prepared/(role+'.json')).read_text()))
    return datasets,json.loads((prepared/'receipt.json').read_text())


def prepare_user_mix(config,ancestor_prepared,user_evaluation,q_export,output):
    from torch.utils.tensorboard import SummaryWriter
    from sttw_control.tolerance_coordination import load_export
    config=Path(config).resolve();cfg=validate_user_mix_config(json.loads(config.read_text()));output=Path(output).resolve();started=time.monotonic()
    if output.exists():raise ValueError('fresh user mix preparation required')
    load_export(q_export)
    data,ancestor_receipt=_sealed_prepared_data(ancestor_prepared);validate_temporal_datasets(data['training'],data['development'])
    evaluation=Path(user_evaluation).resolve();seal=json.loads((evaluation/'seal.json').read_text())
    for name,sha in seal.items():
        if digest(evaluation/name)!=sha:raise ValueError('user late evaluation changed')
    for name in ('features_predictions.npz','predictions.json','receipt.json'):
        if name not in seal:raise ValueError('user late evaluation seal incomplete')
    receipt=json.loads((evaluation/'receipt.json').read_text())
    if receipt['source_task_s']!=6. or receipt['resolved_rows']!=192 or receipt['excluded_invalid'] or receipt['new_optimizer_updates']!=0:raise ValueError('real complete user late192 evaluation required')
    inventory_path=evaluation.parent/'inventory.json'
    if receipt.get('inputs_sha256',{}).get(str(inventory_path))!=digest(inventory_path):raise ValueError('user late original inventory changed')
    declared=json.loads(inventory_path.read_text())
    rows=json.loads((evaluation/'predictions.json').read_text())
    if len(declared)!=192 or {m['case_id'] for m in declared}!={m['case_id'] for m in rows}:raise ValueError('user late labels differ from original declared inventory')
    with np.load(evaluation/'features_predictions.npz',allow_pickle=False) as z:features=z['features'].copy();labels=z['labels'].copy()
    if features.shape!=(192,41) or labels.shape!=(192,) or len(rows)!=192 or not np.isfinite(features).all() or not np.all(labels==1):raise ValueError('complete real positive userlate192 required')
    meta=[]
    for row,y in zip(rows,labels):
        if row['label']!=y or row['source_user_alpha'] not in (0.,.5):raise ValueError('user late label/metadata mismatch')
        meta.append({k:row[k] for k in ('case_id','source_group_id','bank_index','alpha','preparation_s','force_N','source_user_alpha')})
    for role,bank in [('training',4),('development',6)]:
        ids=[i for i,m in enumerate(meta) if m['bank_index']==bank];old=data[role]
        old.update(full=np.concatenate([old['full'],features[ids]]),y=np.concatenate([old['y'],labels[ids]]),meta=old['meta']+[dict(meta[i],period='userlate') for i in ids])
    validate_user_mix_datasets(data['training'],data['development'])
    return _write_user_mix_prepared(config,cfg,data,q_export,output,started,dict(ancestor_prepared=str(Path(ancestor_prepared).resolve()),ancestor_seal_sha256=digest(Path(ancestor_prepared)/'seal.json'),user_evaluation=str(evaluation),user_evaluation_seal_sha256=digest(evaluation/'seal.json')))


def _write_user_mix_prepared(config,cfg,data,q_export,output,started,source_identity):
    from torch.utils.tensorboard import SummaryWriter
    output.mkdir(parents=True)
    for role,values in data.items():
        np.savez(output/(role+'.npz'),full=values['full'],y=values['y'])
        write_json(output/(role+'.json'),{k:v for k,v in values.items() if k not in ('full','y')})
    if 'reference_dataset_files_sha256' in source_identity:
        for name,sha in source_identity['reference_dataset_files_sha256'].items():
            if digest(output/name)!=sha:raise ValueError('prior preparation changed sealed corpus bytes')
    tensorboard=output.parent/'tensorboard'
    for seed in (0,1,2):
        with SummaryWriter(str(tensorboard/'bce_rank'/f'seed{seed}')) as writer:writer.add_scalar('train/optimizer_updates',0,0);writer.flush()
    result=dict(config=str(config),config_sha256=digest(config),implementation_sha256=digest(__file__),q_export=str(Path(q_export).resolve()),q_receipt_sha256=digest(Path(q_export)/'receipt.json'),tensorboard_root=str(tensorboard),**source_identity,preparation_s=time.monotonic()-started,new_optimizer_updates=0,new_physics_steps=0,roles=cfg['user_mix'])
    if 'feature_mask_keep_indices' in cfg:result.update(feature_mask_keep_indices=cfg['feature_mask_keep_indices'],matched_full_prior_training=cfg['matched_full_prior_training'],matched_full_prior_manifest_sha256=cfg['matched_full_prior_manifest_sha256'],input_information='speed plus candidate/disturbance query; offline-only, not pure speed')
    write_json(output/'receipt.json',result);write_json(output/'seal.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    write_json(output.parent/'status.json',dict(phase='prepared',stage='prepared',actual_optimizer_updates=0,tensorboard_root=str(tensorboard)))
    return result


def prepare_user_mix_reference(config,prepared_reference,output):
    """Reuse byte-identical sealed192 corpus; only objective configuration differs."""
    from sttw_control.tolerance_coordination import load_export
    config=Path(config).resolve();cfg=validate_user_mix_config(json.loads(config.read_text()))
    if 'bce_group_weights' not in cfg:raise ValueError('prepared reference requires explicit preregistered prior variant')
    output=Path(output).resolve();started=time.monotonic()
    if output.exists():raise ValueError('fresh prior preparation required')
    data,receipt=_sealed_prepared_data(prepared_reference);validate_user_mix_datasets(data['training'],data['development'])
    if 'feature_mask_keep_indices' in cfg:
        validate_matched_full_prior(cfg)
        full_prepared=Path(cfg['matched_full_prior_training']).parent/'prepared'
        for name in ('training.npz','development.npz','training.json','development.json'):
            if digest(Path(prepared_reference)/name)!=digest(full_prepared/name):raise ValueError('ablation corpus differs from matched full prior')
    q_export=receipt['q_export'];load_export(q_export)
    if digest(Path(q_export)/'receipt.json')!=receipt['q_receipt_sha256']:raise ValueError('reference comparator changed')
    return _write_user_mix_prepared(config,cfg,data,q_export,output,started,dict(prepared_reference=str(Path(prepared_reference).resolve()),prepared_reference_seal_sha256=digest(Path(prepared_reference)/'seal.json'),reference_dataset_files_sha256={name:digest(Path(prepared_reference)/name) for name in ('training.npz','development.npz','training.json','development.json')}))


def user_mix_fit(config,prepared,output,*,on_training_complete=None):
    config=Path(config).resolve();validate_user_mix_config(json.loads(config.read_text()));data,receipt=_sealed_prepared_data(prepared)
    if receipt['config_sha256']!=digest(config) or receipt['implementation_sha256']!=digest(__file__):raise ValueError('prepared user mix configuration/source changed')
    if digest(Path(receipt['q_export'])/'receipt.json')!=receipt['q_receipt_sha256']:raise ValueError('old last comparator changed')
    validate_user_mix_datasets(data['training'],data['development'])
    data.update(receipt=receipt,q_export=receipt['q_export'],tensorboard_root=receipt['tensorboard_root'])
    return ranking_fit(config,None,None,output,_prepared=data,on_training_complete=on_training_complete)


def user_mix_selection(y,p,meta):
    values={period+'_brier':float(np.mean((np.asarray(p)[[m['period']==period for m in meta]]-np.asarray(y)[[m['period']==period for m in meta]])**2)) for period in ('early','late','userlate')}
    return float(np.mean((np.asarray(p)-np.asarray(y))**2)),values


def user_mix_references(data,q_export,*,full_prior_training=None):
    import torch
    from sttw_control.tolerance_coordination import load_export
    load_export(q_export);probabilities=[]
    for seed in (0,1,2):
        model=make_model(41);model.load_state_dict(torch.load(Path(q_export)/f'temporal_rank_last_seed{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);probabilities.append(predict(model,data['full']))
    result={'old_frozen_temporal_rank_last/ensemble':np.mean(probabilities,axis=0)}
    if full_prior_training is not None:
        for seed in (0,1,2):
            model=make_model(41);checkpoint=torch.load(Path(full_prior_training)/f'bce_rank/seed{seed}/last.pt',map_location='cpu',weights_only=True)
            if checkpoint.get('feature_mask_keep_indices') is not None:raise ValueError('matched full prior must be unmasked')
            model.load_state_dict(checkpoint['state_dict']);result[f'full_prior_fixed_last/seed{seed}']=predict(model,data['full'])
        result['full_prior_fixed_last/ensemble']=np.mean([result[f'full_prior_fixed_last/seed{s}'] for s in (0,1,2)],axis=0)
    return result


def user_mix_bce_weights(meta,group_totals):
    """Normalized sample weights; no label-dependent weighting or resampling."""
    if group_totals!={'early':.75,'late':.125,'userlate':.125}:raise ValueError('undeclared group weight variant')
    weights=np.zeros(len(meta),np.float32)
    for group,total in group_totals.items():
        ids=[i for i,m in enumerate(meta) if m['period']==group]
        if len(ids)!={'early':72,'late':24,'userlate':96}[group]:raise ValueError('complete group inventory required for weights')
        weights[ids]=total/len(ids)
    if not np.isclose(weights.sum(),1.,atol=1e-7):raise ValueError('group weights not normalized')
    return weights


def bce_objective(logits,labels,sample_weights=None):
    import torch
    raw=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
    if sample_weights is None:return raw,raw
    if sample_weights.shape!=labels.shape:raise ValueError('BCE sample weight shape differs')
    weighted=(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*sample_weights).sum()
    return raw,weighted


def apply_feature_mask(features,keep_indices=None):
    """Copy only for the one declared offline input ablation; source arrays stay intact."""
    x=np.asarray(features)
    if keep_indices is None:return x
    if list(keep_indices)!=[4]+list(range(34,41)) or x.shape[-1]!=41:raise ValueError('declared speed+query mask on full41 required')
    masked=np.zeros_like(x);masked[...,keep_indices]=x[...,keep_indices]
    return masked


def validate_matched_full_prior(cfg):
    root=Path(cfg['matched_full_prior_training']);manifest=json.loads((root/'selection_manifest.json').read_text());seal=json.loads((root/'selection_seal.json').read_text())['artifacts_sha256']
    if digest(root/'selection_manifest.json')!=cfg['matched_full_prior_manifest_sha256'] or manifest['actual_optimizer_updates']!=3000 or digest(manifest['config'])!=manifest['config_sha256']:raise ValueError('matched full prior identity changed')
    for name in ['selection_manifest.json']+[f'initial_seed{s}.pt' for s in (0,1,2)]+[f'bce_rank/seed{s}/last.pt' for s in (0,1,2)]:
        if digest(root/name)!=seal.get(name):raise ValueError('matched full prior checkpoint/init changed')
    prior_cfg=json.loads(Path(manifest['config']).read_text())
    if prior_cfg.get('feature_mask_keep_indices') is not None or prior_cfg.get('bce_group_weights')!={'early':.75,'late':.125,'userlate':.125}:raise ValueError('matched full prior must use original full inputs and same objective')
    return manifest


def five_alpha_partition(protocol):
    """Role is sealed before sampling; bank4 is an additive training-only batch."""
    train={'straight_v2p6','left70_R1p9','right130_R2p7','S_left_90_R1p8'};dev={'right70_R1p9','left130_R2p7','S_right_90_R1p8'}
    split=protocol['fitting_split'];kind=protocol.get('data_partition','mixed')
    if not split['declared_before_new_labels'] or set(split['training_routes'])&set(split['development_routes']):raise ValueError('predeclared disjoint role split required')
    if kind=='training_only':
        expected_train,expected_dev,total,bank=train|dev,set(),560,4
        groups=protocol['source_groups']
        if len(groups)!=28 or {g['source_task_s'] for g in groups}!={1.,2.,4.,6.}:raise ValueError('training-only bank4 requires28states at1/2/4/6')
        if {(g['route'],g['source_task_s']) for g in groups}!={(r,t) for r in train|dev for t in (1.,2.,4.,6.)}:raise ValueError('training-only full route/time product required')
    elif kind=='mixed':expected_train,expected_dev,total,bank=train,dev,280,6
    else:raise ValueError('unknown sealed data partition')
    if set(split['training_routes'])!=expected_train or set(split['development_routes'])!=expected_dev or {g['bank_index'] for g in protocol['source_groups']}!={bank}:raise ValueError('frozen route/bank partition changed')
    ntrain=sum(g['route'] in expected_train for g in protocol['source_groups'])*20;ndev=total-ntrain
    if protocol['expected_cases']!=total or split['expected_training_rows']!=ntrain or split['expected_development_rows']!=ndev:raise ValueError('sealed partition counts differ')
    return dict(kind=kind,training_routes=expected_train,development_routes=expected_dev,cases=total,training_rows=ntrain,development_rows=ndev,bank=bank)


def validate_five_alpha_fit_config(cfg):
    """Separate opt-in budget; historical pilot schemas remain unchanged."""
    t=cfg['training'];b=cfg['budget'];f=cfg['features'];d=cfg['dataset']
    steps=t['updates_per_model_seed']
    if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=1000:
        raise ValueError('fixed per-seed updates must be an integer in1..1000')
    analyses=d.get('analyses')
    batches=len(analyses) if analyses is not None else 1
    if batches not in (1,2,3) or (analyses is not None and (not isinstance(analyses,list) or len(set(analyses))!=batches or 'analysis' in d)):
        raise ValueError('one to three distinct analysis batches required')
    if (cfg['mode']!='five_alpha_fit' or cfg['architecture']['hidden']!=[16,16] or
        t['seeds']!=[0,1,2] or t['updates_per_model_seed']!=steps or
        t['maximum_total_optimizer_updates']!=3*steps or t['learning_rate']!=.001 or
        t['weight_decay']!=.001 or t['optimizer']!='AdamW' or t['device']!='CPU' or
        t['fixed_last'] is not True or t['full_batch'] is not True or t.get('ranking_weight',.5)!=.5 or
        cfg['ranking']!=dict(weight=.5,pairing='source_group_id/preparation_s/force_N; positive-negative only') or
        b!=dict(new_physics_steps=0,new_ppo_updates=0,maximum_optimizer_updates=3*steps,automatic_extensions=False) or
        f['full_indices']!=list(range(300,320))+list(range(337,351)) or f['speed_indices']!=[304] or f['full_dim']!=41 or
        (d['expected_cases'],d['expected_training_rows'],d['expected_development_rows']) not in ({1:{(280,160,120)},2:{(560,320,240)},3:{(1120,880,240)}}[batches])):
        raise ValueError('frozen five-alpha fitting contract mismatch')
    return cfg


def _five_alpha_read_json(path,inputs):
    path=Path(path).resolve();inputs[str(path)]=digest(path)
    return json.loads(path.read_text())


def _load_five_alpha_single(analysis,cfg):
    """Authenticate every row and independently rebuild source-only41 features."""
    analysis=Path(analysis).resolve();root=analysis.parent;inputs={}
    read=lambda p:_five_alpha_read_json(p,inputs)
    plan=read(root/'plan.json');seal=read(root/'seal.json');receipt=read(analysis/'receipt.json')
    if plan.get('mode')!='five_alpha_tolerance_labels' or not plan.get('label_parent') or seal['plan_sha256']!=digest(root/'plan.json') or receipt['plan_sha256']!=seal['plan_sha256']:raise ValueError('sealed parent label analysis required')
    if set(plan['models'])!={'H75'} or plan['models']['H75'].get('registry_id')!='STTW_MIDDLE_FULLTAIL_PPO75_U77':raise ValueError('frozen H75 only required')
    for key in ('actor','checkpoint'):
        model=plan['models']['H75']
        if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen H75 checkpoint changed')
        inputs[str(model[key])]=model[key+'_sha256']
    protocol=plan['protocol'];split=protocol['fitting_split']
    partition=five_alpha_partition(protocol);expected_train=partition['training_routes'];expected_dev=partition['development_routes'];total=partition['cases']
    if protocol.get('planning_frequency_hz')!=10. or not plan.get('replanning',{}).get('continuous_replanning') or plan['replanning'].get('period_s')!=.1:raise ValueError('only native10Hz labels accepted; old1Hz data excluded')
    def sealed(p,directory):
        for file,sha in p['input_sha256'].items():
            if digest(file)!=sha:raise ValueError('sealed label input changed '+file)
            inputs[str(file)]=sha
        names=[Path(f).name for f in p['source_sha256']]
        if len(names)!=len(set(names)):raise ValueError('ambiguous archived source basename')
        for file,sha in p['source_sha256'].items():
            snapshot=directory/'source'/Path(file).name
            if digest(snapshot)!=sha:raise ValueError('sealed archived source changed '+str(snapshot))
            inputs[str(snapshot)]=sha
    sealed(plan,root)
    records=read(analysis/'results.json');declared=read(root/'inventory.json')
    if len(records)!=total or len(declared)!=total or len({r['case_id'] for r in records})!=total:raise ValueError('complete unique declared cases required')
    children={};child_records={};physical_sources=[]
    for item in plan['children']:
        child=Path(item);child=child if child.is_absolute() else root/child
        cp=read(child/'plan.json');cs=read(child/'seal.json');cr=read(child/'analysis/receipt.json')
        if cs['plan_sha256']!=digest(child/'plan.json') or cr['plan_sha256']!=cs['plan_sha256'] or cp['models']!=plan['models'] or cp['config']!=plan['config']:raise ValueError('child frozen identity differs')
        sealed(cp,child)
        source_records=read(child/'evaluation/source_records.json')
        if len(source_records)!=7 or any(not s['all_twenty_siblings_exact'] or len(s['case_indices'])!=20 for s in source_records):raise ValueError('complete source sibling proof missing')
        for record in source_records:
            fingerprint=record.get('pre_boundary_fingerprint')
            if not isinstance(fingerprint,str) or len(fingerprint)!=64:raise ValueError('exact complete-carry source fingerprint required')
            physical_sources.append(fingerprint)
        rows=read(child/'analysis/results.json');cases=read(child/'inventory.json')
        if len(rows)!=140 or len(cases)!=140:raise ValueError('child140 required')
        path=child/'evaluation/source_features.npz';inputs[str(path)]=digest(path)
        with np.load(path,allow_pickle=False) as z:sources={k:z[k].copy() for k in z.files}
        for i,(r,c) in enumerate(zip(rows,cases)):
            if any(r.get(k)!=v for k,v in c.items()) or r['case_id'] in children:raise ValueError('child row/inventory identity differs')
            children[r['case_id']]=(child,i,c,sources);child_records[r['case_id']]=r
    feature_path=analysis/'features_labels.npz';inputs[str(feature_path)]=digest(feature_path)
    with np.load(feature_path,allow_pickle=False) as z:features=z['features'].copy();labels=z['labels'].copy();case_ids=z['case_ids'].copy();roles=z['fitting_role'].copy();valid=z['valid'].copy()
    if features.shape!=(total,41) or labels.shape!=(total,) or not np.isfinite(features).all():raise ValueError('finite declared Nx41 features required')
    states={};queries={};excluded=[];meta=[];all_roles=[];source_roles={};actual=set();dev_identity=[]
    for i,(record,case) in enumerate(zip(records,declared)):
        cid=record['case_id']
        if record!=child_records.get(cid) or any(record.get(k)!=v for k,v in case.items()) or str(case_ids[i])!=cid:raise ValueError('parent child row identity differs')
        child,lane,c,sources=children[cid];group=c['source_index'];alpha=c['candidate_alpha']
        condition=(group,alpha,c['preparation_s'],c['force_N'])
        if condition in actual:raise ValueError('duplicate sibling')
        actual.add(condition)
        if c['source_user_alpha']!=.5 or c['alpha']!=.5 or c['user_alpha']!=.5 or c['query_alpha']!=alpha:raise ValueError('source/candidate alpha identity changed')
        role='training' if c['route'] in expected_train else 'development' if c['route'] in expected_dev else None
        if role is None or record['fitting_role']!=role or str(roles[i])!=role:raise ValueError('row split differs')
        if group in source_roles and source_roles[group]!=role:raise ValueError('source sibling split')
        source_roles[group]=role;all_roles.append(role)
        raw_path=Path(record['raw_file'])
        if digest(raw_path)!=record['raw_sha256']:raise ValueError('label raw changed')
        inputs[str(raw_path)]=record['raw_sha256']
        with np.load(raw_path,allow_pickle=False) as z:
            if not z['active_tick'][0] or not z['reference_target_permitted'][0] or abs(float(z['time'][0])-c['source_task_s'])>3e-6:raise ValueError('first source candidate missing')
            query=z['query_observation'][0];latent=z['composed_latent'][0];yaw=float(z['pre_upper_yaw'][0])
            active=np.asarray(z['active_tick'],bool);clock=np.rint(z['time']/.005).astype(int)
            window=active&(clock>=round(c['primary_window_start_s']/.005))&(clock<round(c['primary_window_end_s']/.005))
            faults=z['policy_fault']|z['lower_fault']|z['nonfinite']|z['continuation_geometry_fault']
            fall=bool(np.any(window&(z['contact_failure']|z['physical_failure'])));fault=bool(np.any(window&faults));complete=np.array_equal(clock[window],np.arange(round(c['primary_window_start_s']/.005),round(c['primary_window_end_s']/.005)))
            status='invalid_fault' if fault else 'fall' if fall else 'no_fall' if complete else 'invalid_incomplete'
            if record['primary']['label_status']!=status:raise ValueError('primary label fails independent raw reconstruction')
        g=next(g for g in protocol['source_groups'] if g['source_index']==group)
        if record['lifecycle_role']!=g['lifecycle_role']:raise ValueError('source lifecycle changed')
        for field,value in [('source_task_s',g['source_task_s']),('primary_window_start_s',g['source_task_s']),('onset_s',g['source_task_s']+c['preparation_s']),('primary_window_end_s',g['source_task_s']+c['preparation_s']+.15+3),('duration_s',.15)]:
            if not math.isclose(c[field],value,abs_tol=1e-9):raise ValueError('label timing changed')
        source=sources['native_observation'][lane]
        if abs(np.arctan2(np.sin(yaw-sources['source_world_yaw'][lane]),np.cos(yaw-sources['source_world_yaw'][lane])))>1e-6:raise ValueError('source yaw differs')
        rebuilt,_=feature_row(source,query,latent,yaw,c,plan['config']['residual_interface'],cfg,expected_source_alpha=.5)
        np.testing.assert_array_equal(features[i],rebuilt)
        if group in states:np.testing.assert_array_equal(source,states[group])
        states[group]=source;key=(group,alpha)
        if key in queries:np.testing.assert_array_equal(np.r_[query,latent],queries[key])
        queries[key]=np.r_[query,latent]
        label=resolved_label(record['primary']);expected_valid=label is not None
        if bool(valid[i])!=expected_valid or (expected_valid and labels[i]!=label) or (not expected_valid and not np.isnan(labels[i])):raise ValueError('label cache differs')
        meta.append(dict(case_id=cid,source_group_id=f'source{group}',route=c['route'],alpha=alpha,preparation_s=c['preparation_s'],force_N=c['force_N'],lifecycle_role=record['lifecycle_role']))
        if role=='development':dev_identity.append(dict(case_id=cid,features=features[i].tolist(),label=label,valid=expected_valid,raw_sha256=record['raw_sha256']))
        if not expected_valid:excluded.append(dict(meta[-1],role=role,reason=status))
    expected={(g['source_index'],a,p,f) for g in protocol['source_groups'] for a in [0.,.25,.5,.75,1.] for p in [.2,.6] for f in [-64.,64.]}
    if actual!=expected or all_roles.count('training')!=partition['training_rows'] or all_roles.count('development')!=partition['development_rows']:raise ValueError('full predeclared split/product required')
    data={}
    for role in ('training','development'):
        ids=[i for i,r in enumerate(all_roles) if r==role and valid[i]]
        if not ids and not (role=='development' and partition['kind']=='training_only'):raise ValueError('no valid '+role+' labels')
        data[role]=dict(full=features[ids],y=labels[ids],meta=[meta[i] for i in ids])
    coverage={}
    for role in ('training','development'):
        coverage[role]={}
        for key in ('alpha','preparation_s','lifecycle_role'):
            coverage[role][key]={}
            for value in sorted({m[key] for m in meta}):
                ids=[i for i,m in enumerate(meta) if all_roles[i]==role and m[key]==value]
                coverage[role][key][str(value)]=dict(declared_rows=len(ids),valid_rows=sum(bool(valid[i]) for i in ids),invalid_rows=sum(not bool(valid[i]) for i in ids))
    data['receipt']=dict(inputs=inputs,excluded_invalid=excluded,coverage=coverage,run_namespace=digest(root/'plan.json'),source_semantic_keys=[[protocol['source_plan_sha256'],g['source_task_s'],g['source_case_id']] for g in protocol['source_groups']],declared_training_rows=partition['training_rows'],declared_development_rows=partition['development_rows'],data_partition=partition['kind'],physical_source_fingerprints=physical_sources,development_identity_rows=dev_identity,source_groups=len(states),source_state_audit='Inherited exact native fullcarry/path sibling proof;41features and primary labels independently rebuilt',frozen_model_identity=plan['models'],data_role=f'Related bank{partition["bank"]} native10Hz {partition["kind"]} batch; not independent TEST')
    return data



def load_five_alpha_dataset(analysis,cfg):
    """Combine only authenticated native10Hz batches with no repeated ancestors."""
    configured=cfg['dataset'].get('analyses')
    if analysis is None:analysis=configured or cfg['dataset']['analysis']
    paths=list(analysis) if isinstance(analysis,(list,tuple)) else [analysis]
    if len(paths) not in (1,2,3) or len({str(Path(p).resolve()) for p in paths})!=len(paths):raise ValueError('one to three distinct analyses required')
    d=cfg['dataset']
    if configured is not None and [str(Path(p).resolve()) for p in paths]!=[str(Path(p).resolve()) for p in configured]:raise ValueError('analyses must match frozen config')
    batches=[_load_five_alpha_single(path,cfg) for path in paths]
    totals=(sum(b['receipt']['declared_training_rows']+b['receipt']['declared_development_rows'] for b in batches),sum(b['receipt']['declared_training_rows'] for b in batches),sum(b['receipt']['declared_development_rows'] for b in batches))
    if totals!=(d['expected_cases'],d['expected_training_rows'],d['expected_development_rows']):raise ValueError('declared aggregate row counts differ')
    if len(batches)==1:
        verify_five_alpha_development_freeze(batches[0],cfg);return batches[0]
    if configured is None or [str(Path(p).resolve()) for p in paths]!=[str(Path(p).resolve()) for p in configured]:raise ValueError('multiple analyses must match frozen config')
    seen=set();physical_seen=set();inputs={};excluded=[];coverage={};identities=[];dev_identity=[]
    combined={role:dict(full=[],y=[],meta=[]) for role in ('training','development')}
    for batch in batches:
        receipt=batch['receipt'];namespace=receipt['run_namespace'];keys={tuple(k) for k in receipt['source_semantic_keys']}
        if seen&keys:raise ValueError('duplicate semantic source snapshot/task/case across batches')
        fingerprints=set(receipt['physical_source_fingerprints'])
        if len(fingerprints)!=receipt['source_groups'] or physical_seen&fingerprints:raise ValueError('physical complete-state source collision')
        physical_seen|=fingerprints
        dev_identity.extend(dict(r,case_id=namespace+'/'+r['case_id']) for r in receipt['development_identity_rows'])
        seen|=keys;identities.append(receipt['frozen_model_identity']);inputs.update(receipt['inputs']);coverage[namespace]=receipt['coverage']
        for row in receipt['excluded_invalid']:excluded.append(dict(row,run_namespace=namespace,source_group_id=namespace+'/'+row['source_group_id'],case_id=namespace+'/'+row['case_id']))
        for role in combined:
            source=batch[role];combined[role]['full'].append(source['full']);combined[role]['y'].append(source['y'])
            combined[role]['meta'].extend(dict(row,run_namespace=namespace,source_group_id=namespace+'/'+row['source_group_id'],case_id=namespace+'/'+row['case_id']) for row in source['meta'])
    if any(identity!=identities[0] for identity in identities[1:]):raise ValueError('combined frozen model identities differ')
    for role in combined:
        combined[role]['full']=np.concatenate(combined[role]['full']);combined[role]['y']=np.concatenate(combined[role]['y'])
    combined['receipt']=dict(inputs=inputs,excluded_invalid=excluded,coverage=coverage,analyses=[str(Path(p).resolve()) for p in paths],declared_training_rows=totals[1],declared_development_rows=totals[2],physical_source_fingerprints=sorted(physical_seen),development_identity_rows=dev_identity,source_groups=len(seen),source_semantic_keys=[list(k) for k in sorted(seen)],frozen_model_identity=identities[0],data_role='Related native10Hz batches; existing bank6 route split and additive bank4 training-only sources; no independent TEST')
    verify_five_alpha_development_freeze(combined,cfg)
    return combined


def write_five_alpha_development_freeze(data,path):
    """Explicit snapshot writer; caller chooses when to freeze existing DEV."""
    rows=data['receipt']['development_identity_rows']
    if len(rows)!=240 or len({r['case_id'] for r in rows})!=240:raise ValueError('exact240 declared development rows required')
    write_json(path,dict(schema='five_alpha_development_freeze_v1',rows=rows))
    return dict(file=str(Path(path).resolve()),sha256=digest(path))


def verify_five_alpha_development_freeze(data,cfg):
    frozen=cfg['dataset'].get('development_freeze')
    if frozen is None:return
    file=Path(frozen['file']).resolve()
    if digest(file)!=frozen['sha256']:raise ValueError('development freeze hash changed')
    payload=json.loads(file.read_text());rows=payload['rows'];current=data['receipt']['development_identity_rows']
    if len(rows)!=240 or len(current)!=240:raise ValueError('exact frozen240 DEV rows required')
    if payload['schema']=='sttw_development_freeze_v1':
        for path,sha in payload['inputs'].items():
            if data['receipt']['inputs'].get(path)!=sha:raise ValueError('frozen development authenticated inputs changed')
        old={r['case_id']:{k:r[k] for k in ('case_id','features','label')} for r in rows}
        new={r['case_id']:{k:r[k] for k in ('case_id','features','label')} for r in current}
        if any(not r['valid'] for r in current):raise ValueError('frozen development valid population changed')
    elif payload['schema']=='five_alpha_development_freeze_v1':
        old={r['case_id']:r for r in rows};new={r['case_id']:r for r in current}
    else:raise ValueError('unknown development freeze schema')
    if len(old)!=240 or len(new)!=240 or old!=new:raise ValueError('frozen development IDs/features/labels/raw hashes changed')
    data['receipt']['inputs'][str(file)]=frozen['sha256'];data['receipt']['development_freeze_verified']=True


def five_alpha_grouped_metrics(y,p,meta):
    out=dict(overall=metrics(y,p,meta),groups={})
    for key in ('alpha','preparation_s','lifecycle_role'):
        out['groups'][key]={}
        for value in sorted({m[key] for m in meta}):
            ids=[i for i,m in enumerate(meta) if m[key]==value]
            out['groups'][key][str(value)]=metrics(y[ids],p[ids],[meta[i] for i in ids])
    return out


def five_alpha_fit(config,analysis,output,*,on_training_complete=None):
    """Explicit budgeted three-seed CPU fixed-last BCE+.5ranking; never launches physics."""
    import time
    import torch
    from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(1);started=time.monotonic();config=Path(config).resolve();cfg=validate_five_alpha_fit_config(json.loads(config.read_text()));output=Path(output).resolve()
    configured=cfg['dataset'].get('analyses') or [cfg['dataset']['analysis']]
    requested=configured if analysis is None else list(analysis) if isinstance(analysis,(list,tuple)) else [analysis]
    if [str(Path(p).resolve()) for p in configured]!=[str(Path(p).resolve()) for p in requested]:raise ValueError('analysis differs from frozen fitting config')
    if output.exists():raise ValueError('fresh five-alpha fit output required')
    data=load_five_alpha_dataset(requested,cfg);train=data['training'];dev=data['development']
    if len(np.unique(train['y']))!=2:raise ValueError('training needs both resolved classes; no automatic resampling')
    output.mkdir(parents=True);write_json(output/'data_receipt.json',data['receipt']);writer=SummaryWriter(str(output/'tensorboard'));updates=0;per_seed={};selected=[];inflight=False;timings=dict(initialization_s=time.monotonic()-started,optimization_s=0.,validation_s=0.)
    steps=cfg['training']['updates_per_model_seed']
    x=torch.as_tensor(train['full']);y=torch.as_tensor(train['y']);pairs=discordant_pairs(train['y'],train['meta']);models=[]
    try:
        writer.add_scalar('train/optimizer_updates',0,0);writer.flush()
        for seed in (0,1,2):
            torch.manual_seed(seed);model=make_model(41);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            for step in range(1,steps+1):
                tick=time.monotonic();model.train();optimizer.zero_grad();logit=model(x).squeeze(-1);bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,y);rank=ranking_loss(logit,pairs);loss=bce+.5*rank
                if not torch.isfinite(loss):raise ValueError('nonfinite five-alpha loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite five-alpha gradient')
                inflight=True;optimizer.step();updates+=1;inflight=False;per_seed[str(seed)]=step;timings['optimization_s']+=time.monotonic()-tick
                if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite postupdate parameters')
                row=dict(seed=seed,step=step,bce=float(bce.detach()),rank_loss=float(rank.detach()),total_loss=float(loss.detach()))
                for key in ('bce','rank_loss','total_loss'):writer.add_scalar(f'seed{seed}/train/{key}',row[key],step)
                writer.add_scalar('train/optimizer_updates',updates,updates)
                if step%100==0 or step==steps:
                    tick=time.monotonic();score=metrics(dev['y'],predict(model,dev['full']),dev['meta']);row['development_brier']=score['brier'];writer.add_scalar(f'seed{seed}/development/brier',score['brier'],step);timings['validation_s']+=time.monotonic()-tick;writer.flush()
                with (output/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                write_json(output/'status.json',dict(phase='fitting',actual_optimizer_updates=updates,per_seed=per_seed,timings=timings))
            file=output/f'full_state_mlp_seed{seed}_last.pt';torch.save(dict(state_dict=model.state_dict(),dim=41,seed=seed,step=steps,arm='bce_rank',selector='last'),file)
            selected.append(dict(model='full_state_mlp',key='full',dim=41,seed=seed,step=steps,arm='bce_rank',selector='last',file=str(file),sha256=digest(file)));models.append(model)
        manifest=dict(kind='five_alpha_fixed_last',config=str(config),config_sha256=digest(config),implementation_sha256=digest(__file__),selected=selected,actual_optimizer_updates=updates,data_receipt_sha256=digest(output/'data_receipt.json'),new_physics_steps=0,new_ppo_updates=0,candidate_alpha_grid=[0.,.25,.5,.75,1.],feature_dimension=41,selection='All three fixed last, no development selection',limits='Related single-bank DEV; no calibration, safety or online gain claim')
        write_json(output/'selection_manifest.json',manifest);write_json(output/'selection_seal.json',dict(selection_manifest_sha256=digest(output/'selection_manifest.json'),selected_sha256={r['file']:r['sha256'] for r in selected}))
        training_receipt=dict(actual_optimizer_updates=updates,per_seed=per_seed,timings=timings,stage_notified=False)
        write_json(output/'training_complete.json',training_receipt)
        if on_training_complete:
            on_training_complete(training_receipt);training_receipt['stage_notified']=True;write_json(output/'training_complete.json',training_receipt)
        for role in ('training','development'):
            d=data[role];probs=[predict(model,d['full']) for model in models];report={f'seed{s}':five_alpha_grouped_metrics(d['y'],p,d['meta']) for s,p in enumerate(probs)};ensemble=np.mean(probs,axis=0);report['ensemble']=five_alpha_grouped_metrics(d['y'],ensemble,d['meta']);write_json(output/(role+'_metrics.json'),report)
            write_json(output/(role+'_predictions.json'),[dict(m,label=int(label),probabilities=[float(p[i]) for p in probs],ensemble=float(ensemble[i])) for i,(m,label) in enumerate(zip(d['meta'],d['y']))])
        params={}
        for i,layer in enumerate((0,2,4)):
            params[f'w{i}']=np.stack([model[layer].weight.detach().numpy().T for model in models])[None]
            params[f'b{i}']=np.stack([model[layer].bias.detach().numpy() for model in models])[None]
        np.savez_compressed(output/'parameters.npz',**params)
        write_json(output/'export_manifest.json',dict(parameters_sha256=digest(output/'parameters.npz'),selection_manifest_sha256=digest(output/'selection_manifest.json'),arms=['bce_rank_last'],shape='arm1,seed3;w0(41,16),w1(16,16),w2(16,1); ReLU/sigmoid',candidate_alpha_grid=[0.,.25,.5,.75,1.],source_user_alpha=.5,online_coordinator_compatibility='Weight layout compatible; use explicit arm0, five-alpha grid and0.15s hypothetical duration. Online rollout integration remains unvalidated by this export'))
        write_json(output/'status.json',dict(phase='completed',actual_optimizer_updates=updates,per_seed=per_seed,timings=timings))
    except BaseException as exc:
        write_json(output/'status.json',dict(phase='error',actual_optimizer_updates=updates,per_seed=per_seed,optimizer_step_may_be_partial=inflight,error=repr(exc)));raise
    finally:writer.close()
    return manifest
