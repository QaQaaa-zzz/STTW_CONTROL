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
        dev=partition.get('all_development',False) or (row['force_N']==partition.get('development_force_N') and row['preparation_s']==partition.get('development_preparation_s'))
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


def predict_outcomes(export,features):
    """CPU NumPy inference from the exact frozen multi-head ensemble export."""
    x=np.asarray(features,np.float32)
    if x.ndim not in (1,2) or x.shape[-1]!=42 or not np.isfinite(x).all():raise ValueError('finite conditional42 feature vector/matrix required')
    x=np.clip((x-export['mean'])/export['scale'],-5.,5.)
    h=np.einsum('...i,sij->s...j',x,export['w0'])+export['b0'][:,None,:] if x.ndim==2 else np.einsum('i,sij->sj',x,export['w0'])+export['b0']
    h=np.where(h>=0,h,np.expm1(np.minimum(h,0.)))
    h=np.einsum('s...i,sij->s...j',h,export['w1'])+(export['b1'][:,None,:] if x.ndim==2 else export['b1'])
    h=np.where(h>=0,h,np.expm1(np.minimum(h,0.)))
    logits=np.einsum('s...i,sij->s...j',h,export['w2'])+(export['b2'][:,None,:] if x.ndim==2 else export['b2'])
    probabilities=1./(1.+np.exp(-np.clip(logits[...,:2],-60.,60.)))
    continuous=np.expm1(np.clip(logits[...,2:],0.,10.))
    return np.concatenate([probabilities,continuous],axis=-1).mean(axis=0)


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
    root=Path(meta['source_run']);parent=json.loads((root/'plan.json').read_text());rows=json.loads((root/'analysis/results.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(training/'parameters.npz')!=fitted['parameters_sha256'] or sha(parent['models']['H75']['actor'])!=parent['models']['H75']['actor_sha256']:
        raise ValueError('frozen outcome model or middle actor changed')
    with np.load(training/'parameters.npz') as z:export=dict(z)
    with Path(parent['models']['H75']['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
    actor=PathCommandActor(architecture=parent['config']['network']['architecture'])
    source={};checks=[]
    for route in {c['route'] for c in cases}:
        row=next(r for r in rows if r['route']==route and r['force_N']==0. and r['alpha']==0.)
        with np.load(row['raw_file']) as z:raw=dict(z)
        obs=raw['query_observation'][0];latent=np.asarray(actor.apply(params,j.asarray(obs)))
        np.testing.assert_allclose(latent,raw['composed_latent'][0],atol=3e-5,rtol=3e-5)
        source[route]=(obs,float(raw['pre_upper_yaw'][0]));checks.append(dict(route=route,maximum_actor_latent_error=float(np.max(abs(latent-raw['composed_latent'][0])))))
    queries=[]
    for case in cases:
        obs=source[case['route']][0].copy();obs[336]=case['alpha'];queries.append(obs)
    latents=np.asarray(actor.apply(params,j.asarray(np.stack(queries))))
    features=np.stack([outcome_features(obs,z,source[c['route']][1],c,parent['config']['residual_interface']) for obs,z,c in zip(queries,latents,cases)])
    prediction=predict_outcomes(export,features)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_suffix('.npz').exists():raise FileExistsError('frozen prospective prediction already exists')
    np.savez_compressed(output.with_suffix('.npz'),features=features,prediction=prediction,case_ids=np.asarray([c['case_id'] for c in cases]))
    import datetime
    result=dict(schema=SCHEMA,recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=str(training),training_manifest_sha256=sha(training/'manifest.json'),parameters_sha256=fitted['parameters_sha256'],query_data=str(output.with_suffix('.npz')),query_data_sha256=sha(output.with_suffix('.npz')),cases=cases,actor_export_checks=checks,scope='Prospective predictions frozen before new physical alpha interpolation labels. Known four source states/challenges, no new-scenario generalization claim.',interval_examples=dict(no_fall_thresholds=[.7,.8,.9],task_threshold=.5,path_rmse_limits_m=[.3,1.,2.,4.]))
    output.write_text(json.dumps(result,indent=2)+'\n');return result


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
    if (cfg['architecture']['input_dim']!=42 or cfg['architecture']['hidden']!=[16,16] or cfg['architecture']['outputs']!=4
        or cfg['training']['device']!='cpu' or cfg['training']['selection']!='fixed_last'
        or cfg['budget']['physics_ticks']!=0
        or cfg['budget']['optimizer_updates']!=len(cfg['training']['seeds'])*cfg['training']['updates_per_seed']):
        raise ValueError('declared small multi-output CPU fit contract differs')
    meta=json.loads((dataset/'manifest.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if meta['schema']!=SCHEMA or meta['dataset_sha256']!=sha(dataset/'dataset.npz'):raise ValueError('prepared dataset identity changed')
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
        torch.manual_seed(seed);model=nn.Sequential(nn.Linear(42,16),nn.ELU(),nn.Linear(16,16),nn.ELU(),nn.Linear(16,4))
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
        torch.save(dict(state_dict=model.state_dict(),seed=seed,schema=SCHEMA,mean=mean,scale=scale,dataset_sha256=meta['dataset_sha256']),output/'checkpoints'/f'seed{seed}.pt')
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
    result=dict(schema=SCHEMA,dataset=str(dataset),dataset_manifest_sha256=sha(dataset/'manifest.json'),config=cfg,source_model=meta['source_model'],metrics=metrics,selection='fixed last per seed; DEV curves diagnostic, not checkpoint selection',history=history,parameters_sha256=sha(output/'parameters.npz'),adoption='pending',limits='Related four-state pilot; uncalibrated probabilities, no independent TEST or continuous alpha guarantee. Path/speed regression trained only on full-primary-window survival.')
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
