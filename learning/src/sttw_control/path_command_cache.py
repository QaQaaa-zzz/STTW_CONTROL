"""Validate and reuse already-paid fixed zero-upper trajectories."""
from pathlib import Path
import json,shutil
import numpy as np
from .preference_best import config_hash,digest,atomic_json
from .path_command_selection import cases

def reuse_zero_upper(parent,root,config,source,expected_paths):
    parent=Path(parent);root=Path(root);manifest=json.loads((parent/'manifest.json').read_text())
    if config_hash(manifest['config'])!=config_hash(config) or config_hash(manifest['source'])!=config_hash(source):
        raise ValueError('baseline config/model/bank/timing identity mismatch')
    out=root/'zero_upper';out.mkdir(parents=True,exist_ok=True);receipt={}
    count=round(config['training_future_phase_C']['episode_s']/config['timing']['lower_dt_s'])
    prior_receipt=parent/'zero_upper_reuse.json'
    prior=json.loads(prior_receipt.read_text()).get('cases',{}) if prior_receipt.exists() else {}
    for name,route,speed in cases(config):
        file=parent/'zero_upper'/f'{name}.npz';pathfile=file.with_name(name+'_path.npz')
        if not file.exists() or not pathfile.exists():continue
        if name in prior and (digest(file)!=prior[name]['trace_sha256'] or digest(pathfile)!=prior[name]['path_sha256']):raise ValueError('baseline cache SHA changed '+name)
        d=dict(np.load(file));p=dict(np.load(pathfile));expected=expected_paths[route]
        for key in ['s','xy','heading','curvature','goal','turn_end']:
            np.testing.assert_allclose(p[key],np.asarray(getattr(expected,key)),rtol=0,atol=2e-6,err_msg='cached path changed '+name)
        if len(d['time'])!=count:raise ValueError('incomplete baseline '+name)
        np.testing.assert_allclose(d['time'],np.arange(count)*config['timing']['lower_dt_s'],atol=2e-6,rtol=0)
        np.testing.assert_allclose(d['v_user'],speed,atol=1e-6,rtol=0)
        for key in ['offsets','target_offset','filtered_offset']:np.testing.assert_array_equal(d[key],np.zeros_like(d[key]))
        if any(np.any(d[k]) for k in ['physical_failure','domain_exit','policy_fault','lower_fault']):raise ValueError('baseline requires terminal-contract audit '+name)
        if any(not np.isfinite(a).all() for a in d.values() if np.issubdtype(a.dtype,np.number)):raise ValueError('nonfinite cached baseline')
        for src in [file,pathfile]:shutil.copyfile(src,out/src.name)
        receipt[name]={'trace_sha256':digest(file),'path_sha256':digest(pathfile),'source':str(file.resolve()),'steps':count}
    atomic_json(root/'zero_upper_reuse.json',dict(parent=str(parent.resolve()),config_sha256=config_hash(config),source_sha256=config_hash(source),cases=receipt))
    return set(receipt)


def reuse_final_panel(parent,out,alpha,config,source,best,expected_paths):
    """Reuse only complete traces attested at the deliberate saved boundary."""
    parent=Path(parent);out=Path(out)
    receipt_file=parent/'optimization_boundary_interrupt.json'
    if not receipt_file.exists():return set()
    receipt=json.loads(receipt_file.read_text())
    old=receipt['final_panel_source']
    if old['alpha']!=alpha:return set()
    manifest=json.loads((parent/'manifest.json').read_text())
    for key,value in [('config',config),('source',source)]:
        if config_hash(manifest[key])!=config_hash(value):raise ValueError('final cache identity mismatch')
    for key in ['actor_sha256','checkpoint_sha256','source_update','schema','alpha']:
        if old[key]!=best[key]:raise ValueError('final cache selected model mismatch '+key)
    for key in ['actor','checkpoint']:
        if digest(best[key])!=best[key+'_sha256']:raise ValueError('selected weight changed')
    attested={Path(p).name:sha for p,sha in receipt['complete_final_files'].items()}
    count=round(config['training_future_phase_C']['episode_s']/config['timing']['lower_dt_s'])
    reused={}
    for name,route,speed in cases(config):
        files=[parent/f'alpha{alpha}'/'final_best'/f'{name}{suffix}.npz' for suffix in ['', '_path']]
        if not all(p.name in attested for p in files):continue
        if any(digest(p)!=attested[p.name] for p in files):raise ValueError('attested final trace changed')
        with np.load(files[0]) as archive:d={k:archive[k] for k in archive.files}
        with np.load(files[1]) as archive:p={k:archive[k] for k in archive.files}
        for key in ['s','xy','heading','curvature','goal','turn_end']:
            np.testing.assert_allclose(p[key],np.asarray(getattr(expected_paths[route],key)),rtol=0,atol=2e-6)
        size=len(d['time'])
        if not 0<size<=count:raise ValueError('invalid cached length')
        if size<count and not (d['physical_failure'][-1] or d['domain_exit'][-1]):raise ValueError('unfinished final case')
        if any(np.any(d[k]) for k in ['policy_fault','lower_fault']):raise ValueError('engineering failure in cache')
        if any(not np.isfinite(a).all() for a in d.values() if np.issubdtype(a.dtype,np.number)):raise ValueError('nonfinite cache')
        np.testing.assert_allclose(d['time'],np.arange(size)*config['timing']['lower_dt_s'],rtol=0,atol=2e-6)
        np.testing.assert_allclose(d['v_user'],speed,rtol=0,atol=1e-6)
        target=out/'final_best';target.mkdir(exist_ok=True)
        for file in files:shutil.copyfile(file,target/file.name)
        reused[name]={p.name:attested[p.name] for p in files}
    atomic_json(out/'final_panel_reuse.json',dict(parent=str(parent.resolve()),source=old,attestation=str(receipt_file.resolve()),cases=reused))
    return set(reused)


def inherit_numerical_history(parent,out,alpha):
    """Keep parent DEV and statistics available without any physical replay."""
    previous=Path(parent)/f'alpha{alpha}';out=Path(out);copied=[]
    for pattern in ['validation/update_*/*metrics.json','statistics/update_*.json']:
        for src in previous.glob(pattern):
            dst=out/'inherited_history'/src.relative_to(previous)
            if dst.exists():continue
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            copied.append(str(src.resolve()))
    atomic_json(out/'inherited_numerical_history.json',dict(sources=copied))


def _read_complete_panel(folder,alpha,config):
    """Read all six declared cases before any output mutation; retain failures."""
    from .path_command_selection import summarize,summarize_task
    folder=Path(folder);declared=cases(config)
    if len(declared)!=6:raise ValueError('cache requires the complete fixed six-case DEV')
    traces={};files={}
    for name,route,speed in declared:
        trace=folder/f'{name}.npz';path=folder/f'{name}_path.npz'
        if not trace.is_file() or not path.is_file():raise ValueError('missing complete cached case '+name)
        with np.load(trace,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with np.load(path,allow_pickle=False) as z:geometry={k:z[k] for k in z.files}
        for arrays in (data,geometry):
            if any(not np.isfinite(a).all() for a in arrays.values() if np.issubdtype(a.dtype,np.number)):
                raise ValueError('nonfinite cached panel '+name)
        required={'s','xy','heading','curvature','goal','turn_end'}
        if not required.issubset(geometry):raise ValueError('incomplete cached path '+name)
        s=geometry['s'];n=len(s)
        if n<2 or geometry['xy'].shape!=(n,2) or any(geometry[k].shape!=(n,) for k in ('heading','curvature')) or np.any(np.diff(s)<=0):
            raise ValueError('invalid cached path shape '+name)
        np.testing.assert_allclose(data['v_user'],speed,rtol=0,atol=1e-6,err_msg='cached speed changed '+name)
        np.testing.assert_allclose(data['goal_progress'],geometry['goal'],rtol=0,atol=2e-6,err_msg='cached goal changed '+name)
        traces[name]=data
        files[name]={trace.name:digest(trace),path.name:digest(path)}
    # These validate timestep ordering, horizon or genuine terminal endpoint,
    # absence of engineering faults, finite tracking metrics and every case.
    summary=summarize(traces,alpha,config);task=summarize_task(traces,alpha,config)
    return traces,files,summary,task


def _copy_panel(folder,target,files):
    folder=Path(folder);target=Path(target)
    # Check every destination before starting the copy, never overwrite evidence.
    for pair in files.values():
        for name,sha in pair.items():
            dest=target/name
            if dest.exists() and digest(dest)!=sha:raise ValueError('immutable cached destination differs '+str(dest))
    target.mkdir(parents=True,exist_ok=True)
    for pair in files.values():
        for name,sha in pair.items():
            src=folder/name;dest=target/name
            if digest(src)!=sha:raise ValueError('cached source changed while copying '+str(src))
            if not dest.exists():shutil.copyfile(src,dest)
            if digest(dest)!=sha:raise ValueError('cached copy SHA mismatch '+str(dest))


def reuse_fork_parent(receipt,out,alpha,cfg,identity,initial_checkpoint,register=True):
    """Seed a new fork's update-zero candidate from its immutable parent DEV.

    The parent is never modified. On resume, register=False only reads and
    validates parent evidence, leaving the new run's selected best untouched.
    """
    from .path_command_selection import persist_best,persist_task_best
    from .path_command_fork import _audit_saved_parent
    import pickle
    parent=Path(receipt['parent_run']);out=Path(out)
    record=receipt['identities'][f'alpha{alpha}'];folder=parent/f'alpha{alpha}'/'final_best'
    if digest(parent/'manifest.json')!=receipt['parent_manifest_sha256']:
        raise ValueError('parent manifest changed since fork audit')
    frozen=json.loads(json.dumps(cfg));frozen.pop('training_sampler',None)
    original=receipt['parent_manifest']['config']
    frozen['training_future_phase_C']['validation_updates']=original['training_future_phase_C']['validation_updates']
    if config_hash(frozen)!=config_hash(original):raise ValueError('fork cached DEV frozen config mismatch')
    source_file=folder/'source.json';source=json.loads(source_file.read_text())
    if config_hash(source)!=config_hash(record):raise ValueError('parent final panel source differs from fork identity')
    if record['alpha']!=alpha:raise ValueError('parent panel alpha mismatch')
    for field in ('checkpoint','actor'):
        if digest(record[field])!=record[field+'_sha256']:raise ValueError('parent panel '+field+' SHA mismatch')
    traces,files,summary,task=_read_complete_panel(folder,alpha,cfg)
    if not register:return traces
    checkpoint=Path(initial_checkpoint);actor=checkpoint.parent/'actor_0000.pkl'
    if identity.get('alpha')!=alpha or identity.get('config_sha256')!=config_hash(cfg):
        raise ValueError('fork initial candidate identity mismatch')
    candidate=dict(identity,source_update=0,checkpoint=str(checkpoint.resolve()),
                   checkpoint_sha256=digest(checkpoint),actor=str(actor.resolve()),actor_sha256=digest(actor))
    # This also verifies exported actor arrays against the complete saved learner,
    # its identity, update zero, optimizer contents and terminal contract.
    _audit_saved_parent(candidate,identity)
    with Path(record['actor']).open('rb') as stream:old=pickle.load(stream)['params']
    with actor.open('rb') as stream:new=pickle.load(stream)['params']
    if set(old)!=set(new):raise ValueError('fork initial actor layer mismatch')
    for layer in old:
        if set(old[layer])!=set(new[layer]):raise ValueError('fork initial actor parameter mismatch')
        for key in old[layer]:
            if not np.array_equal(np.asarray(old[layer][key]),np.asarray(new[layer][key])):
                raise ValueError('fork initial actor arrays differ from parent')
    target=out/'validation'/'update_0000'
    if (target/'source.json').exists() and config_hash(json.loads((target/'source.json').read_text()))!=config_hash(candidate):
        raise ValueError('immutable update-zero source differs')
    _copy_panel(folder,target,files)
    atomic_json(target/'metrics.json',summary);atomic_json(target/'task_metrics.json',task)
    atomic_json(target/'source.json',candidate)
    atomic_json(target/'reuse.json',dict(parent=str(parent.resolve()),parent_source=source,
                parent_source_sha256=digest(source_file),cases=files,new_source_update=0,
                semantics='same deterministic actor and frozen DEV; no physical rollout'))
    persist_best(out,checkpoint,actor,summary,dict(identity,source_update=0))
    persist_task_best(out,checkpoint,actor,task,dict(identity,source_update=0))
    return traces


def reuse_selected_validation(out,best,cfg):
    """Copy the selected candidate's complete existing DEV, never rerun physics."""
    from .path_command_fork import IDENTITY_KEYS
    out=Path(out);update=best['source_update'];folder=out/'validation'/f'update_{update:04d}'
    source_file=folder/'source.json';source=json.loads(source_file.read_text())
    source_keys=set(source)-{'checkpoint','actor'}
    for key in set(IDENTITY_KEYS)|source_keys|{'source_update','checkpoint_sha256','actor_sha256'}:
        if source.get(key)!=best.get(key):raise ValueError('selected validation source mismatch '+key)
    if source['config_sha256']!=config_hash(cfg):raise ValueError('selected validation config mismatch')
    for record in (source,best):
        for field in ('checkpoint','actor'):
            if digest(record[field])!=record[field+'_sha256']:raise ValueError('selected validation '+field+' SHA mismatch')
    traces,files,summary,task=_read_complete_panel(folder,best['alpha'],cfg)
    target=out/'final_best'
    if (target/'source.json').exists() and config_hash(json.loads((target/'source.json').read_text()))!=config_hash(best):
        raise ValueError('immutable final panel source differs')
    _copy_panel(folder,target,files)
    atomic_json(target/'metrics.json',task);atomic_json(target/'task_metrics.json',task)
    atomic_json(target/'source.json',best)
    atomic_json(out/'selected_validation_reuse.json',dict(source=str(folder.resolve()),
                source_sha256=digest(source_file),source_update=update,cases=files,
                semantics='selected complete saved DEV; no physical rollout'))
    return set(traces)


def _rescore_trace(data,alpha,cfg):
    """Derived reward identity changes; raw physics and original reference remain."""
    from .path_command_reporting import reconstruct_costs,reward_series
    result={k:np.array(v,copy=True) for k,v in data.items()}
    rebuilt=reconstruct_costs(result,alpha,cfg);reward,components=reward_series(result,alpha,cfg)
    for group in ('raw','effective','capped'):
        for key,value in rebuilt[group].items():result[f'{dict(raw="raw_cost",effective="effective_cost",capped="component_capped")[group]}_{key}']=value
    for key,value in rebuilt['diagnostic_subcosts'].items():result[f'diagnostic_subcosts_{key}']=value
    result['offset_rate']=rebuilt['offset_rate'];result['scored_tick_reward']=reward
    result['rho']=np.maximum(np.abs(result['phi']),np.abs(result['phi']+cfg['path_reward'].get('risk_prediction_s',.10)*result['phi_dot']))
    result['tick_reward']=-cfg['path_reward']['scale']*cfg['timing']['lower_dt_s']*sum(rebuilt['effective'].values())
    return result


def reuse_reward_parent(receipt,out,alpha,cfg,identity,initial_checkpoint,expected_paths):
    from .path_command_fork import physical_config_identity
    from .path_command_selection import summarize,summarize_task,persist_best,persist_task_best
    import torch
    out=Path(out);parent=Path(receipt['parent_run']);record=receipt['identities'][f'alpha{alpha}'];folder=parent/f'alpha{alpha}'/'final_best'
    if digest(parent/'manifest.json')!=receipt['parent_manifest_sha256']:raise ValueError('reward parent manifest changed')
    if physical_config_identity(cfg)!=physical_config_identity(receipt['parent_manifest']['config']):raise ValueError('reward cache physical identity mismatch')
    if config_hash(json.loads((folder/'source.json').read_text()))!=config_hash(record):raise ValueError('reward cache selected parent differs')
    saved=torch.load(initial_checkpoint,map_location='cpu',weights_only=False)
    original=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    for key,value in saved['policy'].items():
        if key.startswith('actor.') or key=='log_std':
            if not torch.equal(value,original['policy'][key]):raise ValueError('reward cache initial weights differ from parent')
    if saved['optimizer']['state'] or saved['update']!=0 or saved['consecutive_rejections']!=0:raise ValueError('reward fork initial learner is not fresh')
    traces,files,_,_=_read_complete_panel(folder,alpha,receipt['parent_manifest']['config'])
    for name,route,_ in cases(cfg):
        with np.load(folder/f'{name}_path.npz') as archive:
            for key in ('s','xy','heading','curvature','goal','turn_end'):
                np.testing.assert_allclose(archive[key],np.asarray(getattr(expected_paths[route],key)),rtol=0,atol=2e-6,err_msg='reward parent original path mismatch '+name)
    target=out/'validation'/'update_0000';target.mkdir(parents=True)
    for name,d in traces.items():
        traces[name]=_rescore_trace(d,alpha,cfg)
        np.savez_compressed(target/f'{name}.npz',**traces[name])
        shutil.copyfile(folder/f'{name}_path.npz',target/f'{name}_path.npz')
    actor=Path(initial_checkpoint).with_name('actor_0000.pkl')
    source=dict(identity,source_update=0,checkpoint=str(Path(initial_checkpoint).resolve()),checkpoint_sha256=digest(initial_checkpoint),actor=str(actor.resolve()),actor_sha256=digest(actor))
    atomic_json(target/'source.json',source)
    summary=summarize(traces,alpha,cfg);task=summarize_task(traces,alpha,cfg)
    atomic_json(target/'metrics.json',summary);atomic_json(target/'task_metrics.json',task)
    atomic_json(out/'reward_parent_reuse.json',dict(physical_config_identity=physical_config_identity(cfg),reward_config_sha256=config_hash(cfg['path_reward']),original_parent=record,source_files=files,physics_executed=0,scoring='independent reconstruction from raw physical quantities'))
    persist_best(out,initial_checkpoint,actor,summary,dict(identity,source_update=0))
    persist_task_best(out,initial_checkpoint,actor,task,dict(identity,source_update=0))
    return traces


def reuse_reward_baseline(parent,root,cfg,source,expected_paths):
    from .path_command_fork import physical_config_identity
    parent=Path(parent);manifest=json.loads((parent/'manifest.json').read_text())
    if physical_config_identity(cfg)!=physical_config_identity(manifest['config']):raise ValueError('baseline physical identity mismatch')
    names=reuse_zero_upper(parent,root,manifest['config'],source,expected_paths)
    # A copied baseline is alpha-neutral physics. Derived scoring is performed
    # independently for each alpha by report_final_panel; no common score implied.
    atomic_json(Path(root)/'baseline_reward_identity.json',dict(physical_identity=physical_config_identity(cfg),reward_identity=config_hash(cfg['path_reward']),scoring='per-alpha independent reconstruction',physics_executed=0))
    return names


def inherit_reward_validation(parent,out,alpha,cfg,identity):
    """Audit and preserve saved DEV0/50 for exact reward-pilot resumes."""
    from .path_command_fork import IDENTITY_KEYS
    parent=Path(parent);out=Path(out);count=0
    for folder in sorted((parent/f'alpha{alpha}'/'validation').glob('update_*')):
        record=json.loads((folder/'source.json').read_text())
        if any(record.get(k)!=v for k,v in identity.items()):raise ValueError('resumed validation identity mismatch')
        for key in ('actor','checkpoint'):
            if digest(record[key])!=record[key+'_sha256']:raise ValueError('resumed validation weights changed')
        traces,files,summary,task=_read_complete_panel(folder,alpha,cfg)
        target=out/'validation'/folder.name
        _copy_panel(folder,target,files)
        atomic_json(target/'source.json',record);atomic_json(target/'metrics.json',summary);atomic_json(target/'task_metrics.json',task)
        count+=1
    if not count:raise ValueError('reward resume requires saved complete DEV candidates')
    atomic_json(out/'inherited_validation_receipt.json',dict(parent=str(parent.resolve()),complete_candidates=count,physics_executed=0))
    return count


def diagnostic_cache_identity(context,transform):
    """Dedicated diagnostic identity; never represents the unchanged panel cache."""
    if not context.get('actor_sha256') or not context.get('case_ids'):
        raise ValueError('diagnostic cache requires Actor SHA and explicit case IDs')
    return config_hash(dict(schema='path_fixed_actor_diagnostic_v1',context=context,transform=transform))
