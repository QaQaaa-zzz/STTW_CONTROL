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


def feature_row(source,query,latent,yaw,case,interface,cfg):
    """Source-only state plus explicit hypothetical intervention; no outcome."""
    source=np.asarray(source,np.float32);query=np.asarray(query,np.float32)
    if source.shape!=(351,) or query.shape!=(351,) or not np.isfinite(source).all() or not np.isfinite(query).all():
        raise ValueError('finite native351 source/query required')
    np.testing.assert_allclose(np.delete(source,336),np.delete(query,336),atol=1e-6,rtol=1e-6)
    if abs(float(source[336])-1)>1e-6 or abs(float(query[336])-case['candidate_alpha'])>1e-6:raise ValueError('source/query alpha mismatch')
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

def load_dataset(analysis,role,cfg,*,released=False,audit_fn=source_audit):
    """Only a development-only analysis is permitted during fitting."""
    analysis=Path(analysis).resolve()
    if role=='evaluation' and not released:raise ValueError('held-out labels require sealed selection release')
    if role=='development' and analysis.name!='analysis_development':raise ValueError('development-only artifact required; do not open combined labels')
    root=analysis.parent;plan=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('dataset plan identity changed')
    reviewed=json.loads((analysis/'initial_carry_review.json').read_text())
    if reviewed.get('initial_carry_sha256')!=digest(root/'evaluation/initial_carry.pkl') or not reviewed.get('all_complete_source_copy_leaves_exact'):
        raise ValueError('saved source-state provenance does not match analysis')
    records=json.loads((analysis/'results.json').read_text())
    bank={'training':3,'development':0,'evaluation':1}[role]
    if role=='development' and any(r.get('bank_index')!=0 for r in records):raise ValueError('held-out contamination in development artifact')
    records=[r for r in records if r.get('bank_index',3)==bank]
    if len(records)!=72 or len({r['case_id'] for r in records})!=72:raise ValueError('exact bank72 inventory required')
    declared=json.loads((root/'inventory.json').read_text());byid={r['case_id']:i for i,r in enumerate(declared)}
    audit=audit_fn(root)
    with np.load(root/'evaluation/source_features.npz',allow_pickle=False) as data:sources=data['native_observation_user1']
    full=[];speed=[];labels=[];meta=[];inputs={str(analysis/'initial_carry_review.json'):digest(analysis/'initial_carry_review.json'),str(root/'plan.json'):digest(root/'plan.json'),str(analysis/'results.json'):digest(analysis/'results.json'),str(root/'evaluation/initial_carry.pkl'):digest(root/'evaluation/initial_carry.pkl'),str(root/'evaluation/source_features.npz'):digest(root/'evaluation/source_features.npz')}
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
        x,s=feature_row(sources[lane],query,latent,yaw,case,plan['config']['residual_interface'],cfg)
        group=case.get('source_group_id',f"bank3/source{case['source_index']}")
        if group in feature_groups:np.testing.assert_array_equal(sources[lane],feature_groups[group])
        feature_groups[group]=sources[lane]
        pair=(group,case['candidate_alpha'])
        if pair in candidate_queries:np.testing.assert_allclose(latent,candidate_queries[pair],atol=1e-6,rtol=1e-6)
        candidate_queries[pair]=latent
        full.append(x);speed.append(s);labels.append(label);meta.append(dict(case_id=case['case_id'],source_group_id=group,bank_index=bank,alpha=case['candidate_alpha'],preparation_s=case['preparation_s'],force_N=case['force_N']))
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


def predict(model,x):
    import torch
    model.eval()
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
