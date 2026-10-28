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


def ranking_fit(config,train_analysis,dev_analysis,output):
    """Fixed scratch CPU comparison; never reads or releases bank1 labels."""
    import torch
    from torch.utils.tensorboard import SummaryWriter
    config=Path(config).resolve();cfg=validate_ranking_config(json.loads(config.read_text()))
    # Enforce the development boundary before even opening the training dataset.
    if Path(dev_analysis).name!='analysis_development':raise ValueError('development-only artifact required')
    output=Path(output).resolve()
    if output.exists():raise ValueError('new ranking output only; no implicit retry')
    torch.set_num_threads(1)
    train=load_dataset(train_analysis,'training',cfg);dev=load_dataset(dev_analysis,'development',cfg)
    if len(train['y'])!=72 or len(dev['y'])!=72 or len(np.unique(train['y']))!=2:raise ValueError('exact resolved train72/dev72 required')
    output.mkdir(parents=True);updates=0;inflight=False;selected=[];progress={}
    write_json(output/'data_receipt.json',dict(train={k:v for k,v in train.items() if k not in ('full','speed','y')},development={k:v for k,v in dev.items() if k not in ('full','speed','y')}))
    write_json(output/'resolved_config.json',cfg)
    snapshot=output/'source';snapshot.mkdir();(snapshot/'tolerance_predictor_pilot.py').write_bytes(Path(__file__).read_bytes())
    pairs=discordant_pairs(train['y'],train['meta']);x=torch.as_tensor(train['full']);y=torch.as_tensor(train['y'])
    write_json(output/'status.json',dict(phase='initialized',actual_optimizer_updates=0,per_seed=progress))
    try:
        for seed in (0,1,2):
            torch.manual_seed(seed);initial=copy.deepcopy(make_model(41).state_dict())
            init_file=output/f'initial_seed{seed}.pt';torch.save(initial,init_file)
            for arm in ('bce','bce_rank'):
                model=make_model(41);model.load_state_dict(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                folder=output/arm/f'seed{seed}';folder.mkdir(parents=True);writer=SummaryWriter(str(folder/'tensorboard'));best={};rows=[]
                writer.add_scalar('train/optimizer_updates',0,0);writer.flush()
                try:
                    for step in range(1,1001):
                        if updates>=6000:raise ValueError('ranking optimizer budget exhausted')
                        model.train();optimizer.zero_grad();logits=model(x).squeeze(-1)
                        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,y);rank=ranking_loss(logits,pairs)
                        loss=bce+(.5 if arm=='bce_rank' else 0.)*rank
                        if not torch.isfinite(loss):raise ValueError('nonfinite ranking loss')
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite ranking gradient')
                        inflight=True;optimizer.step();updates+=1;inflight=False
                        row=dict(step=step,bce=float(bce.detach()),rank_loss=float(rank.detach()),total_loss=float(loss.detach()),development_brier=None,development_rank=None)
                        for key in ('bce','rank_loss','total_loss'):writer.add_scalar('train/'+key,row[key],step)
                        if step%100==0:
                            score=metrics(dev['y'],predict(model,dev['full']),dev['meta']);db=score['brier'];dr=score['discordant_pair_ranking']['mean']
                            writer.add_scalar('train/optimizer_updates',step,step)
                            row.update(development_brier=db,development_rank=dr);writer.add_scalar('development/brier',db,step)
                            if dr is not None:writer.add_scalar('development/rank',dr,step)
                            for selector in ('brier','rank'):
                                key=selector_key(selector,db,dr,step)
                                if selector not in best or key<best[selector]['key']:
                                    best[selector]=dict(key=key,step=step,development_brier=db,development_rank=dr,state_dict=copy.deepcopy(model.state_dict()))
                            progress[f'{arm}/seed{seed}']=step
                            write_json(output/'status.json',dict(phase='fitting',arm=arm,seed=seed,step=step,actual_optimizer_updates=updates,per_seed=progress));writer.flush()
                        rows.append(row)
                        with (folder/'history.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
                    write_json(folder/'history.json',rows)
                    for selector,record in best.items():
                        state=record.pop('state_dict');record.pop('key');path=folder/f'{selector}_selected.pt'
                        torch.save(dict(state_dict=state,arm=arm,seed=seed,selector=selector,dim=41,**record),path)
                        selected.append(dict(arm=arm,seed=seed,selector=selector,dim=41,file=str(path),sha256=digest(path),initial_sha256=digest(init_file),**record))
                finally:writer.close()
        manifest=dict(experiment_id=cfg['experiment_id'],implementation_sha256=digest(__file__),source_snapshot_sha256=digest(snapshot/'tolerance_predictor_pilot.py'),config=str(config),config_sha256=digest(config),resolved_config_sha256=digest(output/'resolved_config.json'),data_receipt_sha256=digest(output/'data_receipt.json'),selected=selected,actual_optimizer_updates=updates,new_physics_steps=0,new_ppo_updates=0,heldout_release=False)
        write_json(output/'selection_manifest.json',manifest)
        for role,data in [('training',train),('development',dev)]:
            scores={};predictions={}
            for r in selected:
                checkpoint=torch.load(r['file'],map_location='cpu',weights_only=True);model=make_model(41);model.load_state_dict(checkpoint['state_dict'])
                name=f"{r['arm']}/{r['selector']}/seed{r['seed']}";p=predict(model,data['full']);predictions[name]=p;scores[name]=metrics(data['y'],p,data['meta'])
            for arm in ('bce','bce_rank'):
                for selector in ('brier','rank'):
                    name=f'{arm}/{selector}/ensemble';p=np.mean([predictions[f'{arm}/{selector}/seed{s}'] for s in (0,1,2)],axis=0);predictions[name]=p;scores[name]=metrics(data['y'],p,data['meta'])
            write_json(output/f'{role}_metrics.json',scores)
            write_json(output/f'{role}_predictions.json',[dict(m,label=int(v),probabilities={k:float(p[i]) for k,p in predictions.items()}) for i,(m,v) in enumerate(zip(data['meta'],data['y']))])
        artifacts=[p for p in output.rglob('*') if p.is_file() and p.name!='status.json']
        write_json(output/'selection_seal.json',dict(artifacts_sha256={str(p.relative_to(output)):digest(p) for p in artifacts}))
        write_json(output/'status.json',dict(phase='completed',actual_optimizer_updates=updates,per_seed=progress,selection_sealed=True,heldout_release=False))
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
