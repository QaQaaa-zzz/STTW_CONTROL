"""Explicit foreground recovery from an authenticated retained boundary.

Reuse complete cached collections or restart only an unfinished collection from
its authenticated optimizer boundary. Abandoned work is charged separately.
"""
import copy
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from .finite_value_online import FiniteValueLearner,ROLES,FiniteCollectionCritic
from .finite_value_run import validate_contract,_tensorboard,_verify_reward,collection_value_diagnostic
from .preference_best import atomic_json,digest


def validate_boundary(state,boundary):
    expected=[max(0,boundary-2)*4,boundary*4,boundary*4]
    if (not 1<=boundary<64 or state.get('schema')!='sttw_finite_value_boundary_v1'
            or state.get('exact_boundary_resume') is not True or state.get('next_collection')!=boundary
            or state.get('optimizer_steps')!=expected or state.get('actor_steps')!=expected[0]
            or state.get('critic_steps')!=expected[1] or state.get('settings',{}).get('epochs')!=4):
        raise ValueError('retained complete boundary and original Adam counters required')


def remaining_collections(schedule,boundary):
    if (not 1<=boundary<64 or len(schedule)!=64 or
            any(not cases or any(c['collection_index']!=i for c in cases) for i,cases in enumerate(schedule))):
        raise ValueError('original complete64 collection clock required')
    return [(i,cases) for i,cases in enumerate(schedule) if i>=boundary]


def restore_models(config,plan):
    import torch
    result={}
    if set(plan['resume_models'])!=set(ROLES):raise ValueError('all three original arms required')
    for name,record in plan['resume_models'].items():
        for key in ['actor','checkpoint']:
            if digest(record[key])!=record[key+'_sha256']:raise ValueError('boundary artifact changed: '+name)
        state=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        validate_boundary(state,plan['resume_boundary'])
        obj=FiniteValueLearner.restore(config,state)
        if obj.fingerprint()!=record['fingerprint'] or obj.value_contract()!=record['value_contract']:
            raise ValueError('restored boundary identity differs: '+name)
        result[name]=obj
    return result


def validate_storage_source(state,boundary):
    if (not 1<=boundary<64 or state.get('phase')!='error' or state.get('failure_kind') not in ['disk_full','host_reboot']
            or state.get('declared_updates')!=64 or state.get('completed_updates')!=boundary):
        raise ValueError('latest complete boundary of an explicitly interrupted original64 required')


def authenticate_previous_runtime(source,plan,*,workdir=None):
    workdir=Path(workdir or Path.cwd()).resolve();changed=[]
    coordinator=workdir/'learning/src/sttw_control/finite_value_storage_recovery.py'
    for file,sha in plan['source_sha256'].items():
        file=Path(file)
        snapshot=Path(source)/'source_at_launch'/file.relative_to(workdir)
        if digest(snapshot)!=sha:raise ValueError('historical source snapshot changed: '+str(file))
        current=digest(file)
        if current!=sha:
            if file!=coordinator:raise ValueError('sealed algorithm/runtime changed: '+str(file))
            changed.append(dict(file=str(file),previous_sha256=sha,current_sha256=current,
                                scope='recovery wiring/cost reporting only; original snapshot retained'))
    return changed


def collection_folder(source,index):
    seen=set();source=Path(source)
    while True:
        resolved=source.resolve()
        if resolved in seen:raise ValueError('cyclic recovery lineage')
        seen.add(resolved);plan=json.loads((source/'plan.json').read_text())
        if 'resume_source_run' not in plan or index>plan['resume_boundary']:
            return source/'execution'/f'update_{index:04d}'
        parent=Path(plan['resume_source_run'])
        if digest(parent/'plan.json')!=plan['resume_source_plan_sha256']:raise ValueError('ancestor plan changed')
        source=parent


def collection_allocated_slots(folder):
    chunks=list(Path(folder).glob('chunk_*.npz'));cache=Path(folder)/'cached_physics_receipt.json'
    if cache.exists():
        if chunks:raise ValueError('cached collection cannot also charge copied chunks')
        return len(json.loads(cache.read_text())['chunk_files'])*36000
    return len(chunks)*36000


def reported_costs(state):
    result=dict(state)
    result['recorded_total_active_lower_ticks']=state['active_lower_ticks']+state.get('interrupted_active_lower_ticks',0)
    result['recorded_total_allocated_lower_slots']=state['allocated_lower_slots']+state.get('interrupted_allocated_lower_slots',0)
    result['counters_scope']='active/allocated counters: retained completed collections plus current attempt; recorded_total also includes abandoned partial attempts; unrecorded reboot work UNKNOWN'
    return result


def validate_cached_accounting(pair,state,cache):
    if (cache.get('phase')!='completed' or cache.get('authenticated_boundary')!=45
        or cache.get('sampling_identity_reproduced') is not True
        or cache.get('all_180_outcomes_known') is not True
        or cache.get('physics_steps')!=0 or cache.get('optimizer_updates')!=0
        or cache.get('cached_allocated_lower_slots')!=360000
        or any(pair[k]+cache[c]!=state[k] for k,c in [
          ('active_lower_ticks','cached_actual_lower_ticks'),
          ('allocated_lower_slots','cached_allocated_lower_slots')])):
        raise ValueError('cached46 accounting must match recorded physical collection exactly')


def prepare(root,*,source,boundary):
    from prepare_short_recovery_campaign import register
    root=Path(root).resolve();source=Path(source).resolve()
    source_state=json.loads((source/'status.json').read_text())
    validate_storage_source(source_state,boundary)
    old=json.loads((source/'plan.json').read_text())
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('original plan changed')
    runtime_changes=authenticate_previous_runtime(source,old)
    for file,sha in old['files_sha256'].items():
        if digest(source/file)!=sha:raise ValueError('original training input changed')
    pair_file=source/f'checkpoint_pair_{boundary:04d}.json';pair=json.loads(pair_file.read_text())
    if pair['completed_updates']!=boundary or pair['plan_sha256']!=digest(source/'plan.json'):
        raise ValueError('explicit matched checkpoint boundary required')
    if digest(Path(old['resume_source_run'])/'plan.json')!=old['resume_source_plan_sha256']:
        raise ValueError('original retained prefix plan changed')
    cache_file=source/'cached46_audit.json'
    cache=None;abandoned_actual=abandoned_slots=0;partial_audit=None
    if cache_file.exists() and boundary==45:
        cache=json.loads(cache_file.read_text());validate_cached_accounting(pair,source_state,cache)
        for record in cache['chunk_files']:
            if digest(record['file'])!=record['sha256']:raise ValueError('cached chunk changed')
    elif source_state['failure_kind']=='host_reboot':
        partial_file=source/f'reboot{boundary}_boundary_audit.json'
        partial_audit=json.loads(partial_file.read_text())
        if (partial_audit.get('phase')!='completed' or partial_audit.get('latest_authenticated_boundary')!=boundary
            or partial_audit.get('next_sampling_identity_reproduced') is not True
            or partial_audit.get('physics_steps')!=0 or partial_audit.get('optimizer_updates')!=0
            or partial_audit.get('intermediate_complete_carry_persisted') is not False):
            raise ValueError('authenticated incomplete collection without saved intermediate carry required')
        for record in partial_audit[f'partial{boundary+1}_chunks']:
            if digest(record['file'])!=record['sha256']:raise ValueError('abandoned partial chunk changed')
        abandoned_actual=partial_audit[f'partial{boundary+1}_recorded_actual_lower_ticks']
        abandoned_slots=partial_audit[f'partial{boundary+1}_recorded_allocated_lower_slots']
        if (pair['active_lower_ticks']+abandoned_actual!=source_state['active_lower_ticks']
            or pair['allocated_lower_slots']+abandoned_slots!=source_state['allocated_lower_slots']):
            raise ValueError('abandoned partial cost does not match last recorded status')
    else:raise ValueError('complete cache or explicit reboot partial audit required')
    cfg=json.loads((source/'effective_config.json').read_text())
    models={record['arm']:record for record in pair['models']}
    learners=restore_models(cfg,dict(resume_models=models,resume_boundary=boundary))
    pending=json.loads((source/'execution'/f'update_{boundary+1:04d}'/'sampling_identity.json').read_text())
    for name,obj in learners.items():
        if obj.freeze_collection()!=pending['models'][name]:raise ValueError('next collection sampling identity differs')
    remaining_collections(json.loads((source/'training_cases.json').read_text()),boundary)
    actual=allocated=legal=actor_samples=0
    for index in range(1,boundary+1):
        folder=collection_folder(source,index)
        receipt=json.loads((folder/'collection_complete.json').read_text())
        if not receipt['all_known']:raise ValueError('retained prefix has unknown outcome')
        for key,file in [('raw_decisions_sha256','decisions.npz'),('raw_physical_sha256','physical.npz')]:
            if digest(folder/file)!=receipt[key]:raise ValueError('retained raw prefix changed')
        if set(receipt['case_summaries'])!=set(ROLES):raise ValueError('retained prefix arms incomplete')
        for name,rows in receipt['case_summaries'].items():
            if len(rows)!=60:raise ValueError('retained prefix episodes incomplete')
            actual+=sum(c['physical_ticks'] for c in rows)
            result=json.loads((folder/(name+'_learner_result.json')).read_text())
            legal+=result['samples']
            if index>2:actor_samples+=result['samples']
        allocated+=collection_allocated_slots(folder)
    if actual!=pair['active_lower_ticks'] or allocated!=pair['allocated_lower_slots']:
        raise ValueError('retained boundary cost mismatch')
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 中断后三臂原边界恢复准备',root/'prepare_status.json')
    try:
        for name in old['files_sha256']:shutil.copyfile(source/name,root/name)
        prefix=dict(source=str(source),source_plan_sha256=digest(source/'plan.json'),
            checkpoint_pair=str(pair_file),checkpoint_pair_sha256=digest(pair_file),boundary=boundary,
            retained_active_lower_ticks=actual,retained_allocated_lower_slots=allocated,
            retained_legal_decisions=legal,retained_actor_samples=actor_samples,
            prefix_data='Authenticated recursive completed lineage; no repeated optimization',
            runtime_changes=runtime_changes,partial_audit=partial_audit,
            interrupted_active_lower_ticks=abandoned_actual+old.get('resume_prefix',{}).get('interrupted_active_lower_ticks',0),
            interrupted_allocated_lower_slots=abandoned_slots+old.get('resume_prefix',{}).get('interrupted_allocated_lower_slots',0),
            cached_collection=str(source/'execution'/f'update_{boundary+1:04d}') if cache else None,
            cached_audit=str(cache_file) if cache else None,cached_audit_sha256=digest(cache_file) if cache else None,
            cached_carry_sha256={name:digest(source/'execution'/f'update_{boundary+1:04d}'/name) for name in ['initial_carry.pkl','final_carry.pkl']} if cache else {},
            cached_actual_lower_ticks=cache['cached_actual_lower_ticks'] if cache else 0,
            cached_allocated_lower_slots=cache['cached_allocated_lower_slots'] if cache else 0,
            unrecorded_pre_reboot_physical_and_optimizer_work='UNKNOWN; no full-cost claim across reboot',
            next_sampling_identity_reproduced=True,additional_optimizer_updates=0)
        atomic_json(root/'resume_contract.json',prefix)
        p=copy.deepcopy(old)
        p.update(experiment='finite_value_online_v1_recovery_boundary'+str(boundary),
            resume_source_run=str(source),resume_source_plan_sha256=digest(source/'plan.json'),
            resume_boundary=boundary,resume_models=models,resume_prefix=prefix,
            initialization=f'Exact{boundary} Actor/Critic/logstd/Adam; original remaining global sampling indices; complete cache only if authenticated',
            remaining_updates=64-boundary,
            maximum_recorded_allocated_lower_slots=old['maximum_allocated_lower_slots']+prefix['interrupted_allocated_lower_slots'],
            engineering_reexecution_collection=boundary+1 if partial_audit else None,
            cost_contract='Original64 logical collections plus explicit abandoned partial-attempt overhead; unrecorded reboot work UNKNOWN',
            code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            stop='Original64 endpoint only; no retained prefix replay; any new error preserves and stops without automatic retry')
        files=list(Path('learning/src/sttw_control').glob('*.py'))+[Path('learning/cli/resume_finite_value_storage.py'),
            Path('docs/evidence/tolerance_progress_20261022/finite_value_online/RECOVERY_BOUNDARY.md')]
        sources={}
        for file in files:
            sources[str(file.resolve())]=digest(file);dest=root/'source_at_launch'/file
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        p['source_sha256']=sources
        p['files_sha256']={name:digest(root/name) for name in [*old['files_sha256'],'resume_contract.json']}
        atomic_json(root/'plan.json',p);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',completed_updates=boundary,declared_updates=64,
            active_lower_ticks=actual,allocated_lower_slots=allocated,legal_decisions_collected=legal,
            actor_training_samples=actor_samples,tensorboard_reward_verified=False,
            optimizer_steps_by_arm={name:[max(0,boundary-2)*4,boundary*4,boundary*4] for name in ROLES},
            resume_boundary=boundary,retained_prefix_source=str(source),
            interrupted_active_lower_ticks=prefix['interrupted_active_lower_ticks'],
            interrupted_allocated_lower_slots=prefix['interrupted_allocated_lower_slots'],
            maximum_recorded_allocated_lower_slots=p['maximum_allocated_lower_slots']+prefix['interrupted_allocated_lower_slots']))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 原三臂64轮从认证边界续训',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 三臂恢复180路零积分shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),checked_unix=time.time()))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc),checked_unix=time.time()));raise


def execute(root,*,shape_only=False):
    root=root.resolve();previous=json.loads((root/'status.json').read_text())
    if previous['phase']!='prepared':raise ValueError('fresh prepared execution only; no implicit restart')
    directory=root/'shape' if shape_only else root/'execution';directory.mkdir(exist_ok=False)
    state=dict(previous,phase='running',stage='authenticating',pid=os.getpid());writers={}
    status_file=directory/'status.json' if shape_only else root/'status.json'
    def update(**kw):
        state.update(kw,checked_unix=time.time());state.update(reported_costs(state));atomic_json(status_file,state)
    update()
    try:
        import torch,jax
        import jax.numpy as j
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor,make_path_critic,export_torch_actor
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate
        from .survival_live_collection import LiveSurvivalCollectionState,make_live_survival_collection_step
        from .survival_live_batch import assemble_live_arm
        from .anticipatory_run import flatten_ticks
        p=json.loads((root/'plan.json').read_text())
        assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        assert digest(Path(p['resume_source_run'])/'plan.json')==p['resume_source_plan_sha256']
        assert digest(p['resume_prefix']['checkpoint_pair'])==p['resume_prefix']['checkpoint_pair_sha256']
        for f,sha in p['source_sha256'].items():assert digest(f)==sha,f
        for f,sha in p['files_sha256'].items():assert digest(root/f)==sha,f
        assert digest(p['source_plan'])==p['source_plan_sha256']
        for k in ['actor','checkpoint']:assert digest(p['model'][k])==p['model'][k+'_sha256']
        cfg=json.loads((root/'effective_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        validate_contract(cfg,p)
        torch.set_num_threads(4)
        learners=restore_models(cfg,p);start=p['resume_boundary']
        for name,obj in learners.items():
            assert obj.next_collection==start
            with Path(p['resume_models'][name]['actor']).open('rb') as stream:old_actor=pickle.load(stream)
            new_actor=obj.sampling_parameters()['actor']
            assert jax.tree.structure(new_actor)==jax.tree.structure(old_actor)
            for a,b in zip(jax.tree.leaves(new_actor),jax.tree.leaves(old_actor)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        atomic_json(directory/'initialization_receipt.json',dict(exact_retained_boundary=True,
            resume_boundary=start,resume_models=p['resume_models'],fresh_adam=False,
            optimizer_steps={name:obj.optimizer_steps for name,obj in learners.items()},
            actual_settings={name:obj.settings for name,obj in learners.items()},
            original_global_sampling_clock=True,physics_replay_bitwise_claim=False))
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2');paths={};origins={}
        if not np.isclose(env.cc.dt,p['lower_dt_s']):raise ValueError('actual lower clock mismatch')
        for name,c in p['contexts'].items():
            assert digest(c['initial'])==c['initial_sha256'] and digest(c['path'])==c['path_sha256']
            with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
            s=jax.tree.map(lambda x:j.asarray(x[0]),bank)
            assert int(s.tick)==0 and int(s.physical.physical_tick)==700 and not bool(s.physical.failed|s.fault|s.domain_exit)
            origins[name]=s
            with np.load(c['path'],allow_pickle=False) as z:paths[name]=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        schedule=json.loads((root/'training_cases.json').read_text());cases=schedule[0]
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        states=[origins[c['context']].replace(alpha=j.float32(c['alpha'])) for c in cases]
        batch_path=stack([paths[c['context']] for c in cases]);speeds=j.asarray([c['speed'] for c in cases],j.float32)
        profiles=j.asarray([c['profile'] for c in cases],j.int32)
        env.set_log_template(states[0],paths[cases[0]['context']],speeds[0])
        one=make_live_survival_collection_step(env,make_path_actor(cfg),horizon_s=10.,task_critic=FiniteCollectionCritic(make_path_critic(cfg)))
        @jax.jit
        def chunk(carry,params,logstd,pulses):
            return jax.lax.scan(lambda s,_:jax.vmap(one)(s,params,logstd,batch_path,speeds,pulses,profiles),carry,None,length=50)
        def collection_inputs(index):
            cases=schedule[index]
            if [(c['context'],c['alpha'],c['speed'],c['arm'],c['profile']) for c in cases] != [(c['context'],c['alpha'],c['speed'],c['arm'],c['profile']) for c in schedule[0]]:raise ValueError('lane ordering changed')
            key=jax.random.fold_in(jax.random.PRNGKey(p['seed']),index)
            carry=stack([LiveSurvivalCollectionState(s,jax.random.fold_in(key,c['rng_fold_in']),initial_gate(),j.int32(0)) for s,c in zip(states,cases)])
            params={n:o.sampling_parameters() for n,o in learners.items()}
            params=stack([params[c['arm']] for c in cases])
            logstd=j.stack([j.asarray(learners[c['arm']].log_std.detach().numpy()) for c in cases])
            pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+.1,
                *[c['force_N']*d for d in c['world_direction']]] for c in cases],j.float32)
            return carry,params,logstd,pulses
        if shape_only:
            shaped=jax.eval_shape(chunk,*collection_inputs(0))
            assert shaped[1][0]['valid'].shape==(50,180) and shaped[1][1]['active_tick'].shape==(50,180,4)
            atomic_json(directory/'receipt.json',dict(decision_shape=[50,180],lower_shape=[50,180,4],physics_steps=0,optimizer_updates=0))
            update(phase='completed',stage='shape_complete');return
        assert json.loads((root/'shape/status.json').read_text())['phase']=='completed'
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        url,writers=_tensorboard(root);update(tensorboard_url=url,stage='starting_first_collection')
        def save_pair(index):
            records=[]
            for name,obj in learners.items():
                assert obj.next_collection==index
                d=root/name/'checkpoints';d.mkdir(parents=True,exist_ok=True)
                pt=d/f'learner_{index:04d}.pt';actor=d/f'actor_{index:04d}.pkl'
                torch.save(obj.checkpoint(),pt)
                with actor.open('wb') as f:pickle.dump(jax.device_get(export_torch_actor(obj.actor,cfg)),f)
                records.append(dict(arm=name,value_contract=obj.value_contract(),profile=p['arms'][name],checkpoint=str(pt),checkpoint_sha256=digest(pt),actor=str(actor),actor_sha256=digest(actor),fingerprint=obj.fingerprint()))
            atomic_json(root/f'checkpoint_pair_{index:04d}.json',dict(completed_updates=index,models=records,plan_sha256=digest(root/'plan.json'),
                active_lower_ticks=state['active_lower_ticks'],allocated_lower_slots=state['allocated_lower_slots']))
        save_pair(start);started=time.monotonic()
        for iteration,cases in remaining_collections(schedule,start):
            droot=directory/f'update_{iteration+1:04d}';droot.mkdir(exist_ok=False)
            frozen={n:o.freeze_collection() for n,o in learners.items()}
            atomic_json(droot/'sampling_identity.json',dict(models=frozen,cases=cases,plan_sha256=digest(root/'plan.json')))
            decisions=[];physical=[];warmup=iteration<p['critic_only_updates']
            if iteration==start and p['resume_prefix'].get('cached_collection'):
                cache=Path(p['resume_prefix']['cached_collection'])
                audit_file=Path(p['resume_prefix']['cached_audit'])
                if digest(audit_file)!=p['resume_prefix']['cached_audit_sha256']:raise ValueError('cached audit changed')
                audit=json.loads(audit_file.read_text())
                identity=json.loads((cache/'sampling_identity.json').read_text())
                if identity['models']!=frozen or identity['cases']!=cases:raise ValueError('cached46 frozen sampling mismatch')
                for record in audit['chunk_files']:
                    file=Path(record['file'])
                    if digest(file)!=record['sha256']:raise ValueError('cached46 chunk changed')
                    with np.load(file,allow_pickle=False) as z:
                        decisions.append({k[9:]:z[k] for k in z.files if k.startswith('decision_')})
                        physical.append({k[6:]:z[k] for k in z.files if k.startswith('lower_')})
                if len(decisions)!=10:raise ValueError('all original46 chunks required')
                for name,sha in p['resume_prefix']['cached_carry_sha256'].items():
                    if digest(cache/name)!=sha:raise ValueError('cached carry changed')
                    shutil.copyfile(cache/name,droot/name)
                state['active_lower_ticks']+=audit['cached_actual_lower_ticks']
                state['allocated_lower_slots']+=audit['cached_allocated_lower_slots']
                assert state['active_lower_ticks']<=p['maximum_active_lower_ticks'] and state['allocated_lower_slots']<=p['maximum_allocated_lower_slots']
                atomic_json(droot/'cached_physics_receipt.json',dict(source=str(cache),
                    chunk_files=audit['chunk_files'],new_physics_steps=0,new_optimizer_before_assembly=0))
                update(stage='assembling_cached46',current_update=iteration+1,training_phase='ppo',completed_blocks_current_update=10)
            else:
                carry,params,logstd,pulses=collection_inputs(iteration)
                with (droot/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
                decisions=[];physical=[];warmup=iteration<p['critic_only_updates']
                update(stage='collecting',current_update=iteration+1,training_phase='critic_warmup' if warmup else 'ppo',completed_blocks_current_update=0)
                for block in range(10):
                    carry,(d,logs)=chunk(carry,params,logstd,pulses);jax.block_until_ready(carry)
                    host,d,logs=jax.device_get((carry,d,logs));logs=flatten_ticks(logs)
                    decisions.append(d);physical.append(logs)
                    np.savez_compressed(droot/f'chunk_{block:02d}.npz',**{'decision_'+k:v for k,v in d.items()},**{'lower_'+k:v for k,v in logs.items()})
                    state['active_lower_ticks']+=int(logs['active_tick'].sum());state['allocated_lower_slots']+=36000
                    assert state['active_lower_ticks']<=p['maximum_active_lower_ticks'] and state['allocated_lower_slots']<=p['maximum_allocated_lower_slots']
                    update(completed_blocks_current_update=block+1,elapsed_training_s=time.monotonic()-started,
                        completed_episodes_current_update=int(np.sum(host.state.physical.failed|(host.state.tick>=2000))))
                    if np.any(host.state.fault):raise RuntimeError('engineering fault; raw preserved, no automatic retry')
                    if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
                with (droot/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
            decision={k:np.concatenate([x[k] for x in decisions]) for k in decisions[0]};phys={k:np.concatenate([x[k] for x in physical]) for k in physical[0]}
            np.savez_compressed(droot/'decisions.npz',**decision);np.savez_compressed(droot/'physical.npz',**phys)
            batches={};rows={};value_diagnostics={}
            for name in learners:
                ids=[i for i,c in enumerate(cases) if c['arm']==name]
                batches[name],rows[name]=assemble_live_arm(decision,phys,cases,ids,collection_id=frozen[name]['collection_id'],dt_s=.005,horizon_s=10.,gamma=p['gamma'],gae_lambda=learners[name].gae_lambda)
                assert len(rows[name])==60
                value_diagnostics[name]=collection_value_diagnostic(decision,phys,ids,gamma=p['gamma'])
            atomic_json(droot/'value_mc_diagnostics.json',dict(semantics='realized same-collection MC residual; not independent calibration or causal evidence',arms=value_diagnostics))
            atomic_json(droot/'collection_complete.json',dict(case_summaries=rows,raw_decisions_sha256=digest(droot/'decisions.npz'),raw_physical_sha256=digest(droot/'physical.npz'),all_known=True))
            expected={};update(stage='updating')
            for name,obj in learners.items():
                result=obj.update(batches[name],warmup=warmup);atomic_json(droot/(name+'_learner_result.json'),result)
                torch.save(obj.checkpoint(),droot/(name+'_post_update.pt'))
                records=rows[name];writer=writers[name];expected[name]=float(np.mean([r['task_scored_return'] for r in records]))
                for window,diagnostic in value_diagnostics[name].items():
                    for metric,value in diagnostic.items():
                        if value is not None:writer.add_scalar(f'value_mc/{window}/{metric}',value,iteration+1)
                for k,v in dict(reward=expected[name],episode_survival=np.mean([r['outcome']=='survived' for r in records]),legal_actor_decisions=result['samples'],
                    fallback_s=sum(r['fallback_ticks'] for r in records)*.005,mean_full_pulse_exposure=np.mean([r['full_pulse_exposure'] for r in records]),actual_lower_ticks=sum(r['physical_ticks'] for r in records)).items():writer.add_scalar('rollout/'+k,v,iteration+1)
                for alpha in [0.,.25,.5,.75,1.]:
                    chosen=[r for r in records if r['alpha']==alpha];tag=f'alpha_{alpha:g}'
                    for k,v in dict(reward=np.mean([r['task_scored_return'] for r in chosen]),survival=np.mean([r['outcome']=='survived' for r in chosen]),
                        legal_actor_decisions=sum(r['actor_decisions'] for r in chosen),max_path_error=np.mean([r['max_task_path_error'] for r in chosen])).items():writer.add_scalar(tag+'/'+k,v,iteration+1)
                for context in ['left','right']:
                    for force in [0.,-32.,32.]:
                        chosen=[r for r in records if cases[r['case_index']]['context']==context and r['force_N']==force]
                        writer.add_scalar(f'exposure/{context}_F{force:g}_legal_decisions',sum(r['actor_decisions'] for r in chosen),iteration+1)
                        writer.add_scalar(f'exposure/{context}_F{force:g}_actual_ticks',sum(r['physical_ticks'] for r in chosen),iteration+1)
                for k in ['exact_gaussian_KL_mean','likelihood_ratio_clip_fraction','task_advantage_scale','actor_steps','critic_steps']:writer.add_scalar('train/'+k,result[k],iteration+1)
                writer.add_scalar('train/critic_only',int(warmup),iteration+1)
                for k,v in result['history'][-1].items():writer.add_scalar('train/'+k,v,iteration+1)
                writer.flush();state['legal_decisions_collected']+=result['samples']
                if not warmup:state['actor_training_samples']+=result['samples']
            save_pair(iteration+1)
            if iteration in [start,63]:_verify_reward(root,url,iteration+1,expected);state['tensorboard_reward_verified']=True
            update(stage='collection_boundary',completed_updates=iteration+1,optimizer_steps_by_arm={n:o.optimizer_steps for n,o in learners.items()},elapsed_training_s=time.monotonic()-started)
        atomic_json(root/'results.json',dict(completed_updates=64,episodes_per_arm=3840,active_lower_ticks=state['active_lower_ticks'],allocated_lower_slots=state['allocated_lower_slots'],
            evaluation_status='not_started',adoption='none',fixed_endpoint=str(root/'checkpoint_pair_0064.json'),
            resume_boundary=start,retained_prefix=p['resume_prefix'],
            new_active_lower_ticks=state['active_lower_ticks']-p['resume_prefix']['retained_active_lower_ticks']-p['resume_prefix']['cached_actual_lower_ticks'],
            new_allocated_lower_slots=state['allocated_lower_slots']-p['resume_prefix']['retained_allocated_lower_slots']-p['resume_prefix']['cached_allocated_lower_slots'],
            cached_allocated_lower_slots=p['resume_prefix']['cached_allocated_lower_slots'],
            interrupted_active_lower_ticks=p['resume_prefix']['interrupted_active_lower_ticks'],
            interrupted_allocated_lower_slots=p['resume_prefix']['interrupted_allocated_lower_slots'],
            recorded_total_active_lower_ticks=reported_costs(state)['recorded_total_active_lower_ticks'],
            recorded_total_allocated_lower_slots=reported_costs(state)['recorded_total_allocated_lower_slots'],
            unrecorded_pre_reboot_work='UNKNOWN; excluded from retained-prefix plus resumed counters'))
        for k in ['actor','checkpoint']:assert digest(p['model'][k])==p['model'][k+'_sha256']
        update(phase='completed',stage='training_complete_evaluation_separate')
    except BaseException as exc:
        original_traceback=traceback.format_exc()
        if 'learners' in locals():
            state['optimizer_steps_by_arm']={n:o.optimizer_steps for n,o in learners.items()}
            state['learner_journal']={n:dict(next_collection=o.next_collection,pending_collection=o._pending) for n,o in learners.items()}
            try:
                # May contain a partially applied epoch: diagnostic only, never
                # label this a resumable paired boundary or replay its update.
                torch.save({n:dict(value_contract=o.value_contract(),actor=o.actor.state_dict(),task_critic=o.task_critic.state_dict(),
                    physical_critic=o.physical_critic.state_dict(),log_std=o.log_std.detach(),
                    optimizers=[x.state_dict() for x in o.optimizers],optimizer_steps=o.optimizer_steps,
                    next_collection=o.next_collection,pending_collection=o._pending,exact_boundary_resume=False)
                    for n,o in learners.items()},directory/'error_diagnostic_state.pt')
            except BaseException as diagnostic_error:state['diagnostic_write_error']=repr(diagnostic_error)
        (directory/'traceback.txt').write_text(original_traceback)
        update(phase='error',error=repr(exc));raise
    finally:
        for writer in writers.values():writer.close()
