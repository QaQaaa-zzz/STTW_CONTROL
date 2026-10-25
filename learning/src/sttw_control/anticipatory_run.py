"""Fixed, staged DEV evaluation of legal progress-gated reference candidates."""
import json
import os
import pickle
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json, digest


def flatten_ticks(data):
    return {k:np.swapaxes(v,1,2).reshape((-1,v.shape[1])+v.shape[3:])
            for k,v in data.items()}


def physical_outcome(active,contact,fault,limit):
    active=np.asarray(active,bool)
    if np.any(active & np.asarray(fault,bool)):return 'unknown'
    if np.any(active & np.asarray(contact,bool)):return 'contact'
    return 'survived' if int(active.sum())==limit else 'unknown'


def stage_indices(cases,stage):
    if stage not in ('nominal','disturbed','endpoints'):raise ValueError('explicit stage required')
    return [i for i,c in enumerate(cases) if c['stage']==stage]


def architecture_groups(indices,cases,models,common_config,*,batch_size=16):
    """Keep incompatible parameter trees in separate compiled Actor groups."""
    from .path_command_policy import architecture_identity
    families={}
    for i in indices:
        cfg=models[cases[i]['actor_role']].get('actor_config',common_config)
        family=architecture_identity(cfg)
        families.setdefault(family,dict(actor_config=cfg,case_indices=[]))['case_indices'].append(i)
    groups=[]
    for family,item in families.items():
        ids=item['case_indices']
        for start in range(0,len(ids),batch_size):
            groups.append(dict(architecture=family,actor_config=item['actor_config'],case_indices=ids[start:start+batch_size]))
    return groups


def claim_stage(directory):
    """Allow a notification-only preregistration; never overwrite run evidence."""
    if directory.exists():
        if any(p.name!='ACTIVE_RUN.json' for p in directory.iterdir()):
            raise FileExistsError('stage already contains execution evidence: '+str(directory))
    else:directory.mkdir()


def prepare(inventory,root):
    from .survival_endpoint_models import authenticate_endpoint
    specification=json.loads(inventory.read_text());source=Path(specification['source_run'])
    source_plan=json.loads((source/'plan.json').read_text())
    auth=authenticate_endpoint(source_plan['training_run'])
    for path,key in [(Path(specification['source_initial_state']),'source_initial_state_sha256'),
                     (source/'cases.json','source_cases_sha256'),(source/'groups.json','source_groups_sha256')]:
        if digest(path)!=specification[key]:raise ValueError('source inventory drift '+str(path))
    root.mkdir(parents=True,exist_ok=False)
    for name in ['common_environment_config.json','source_declaration.json']:
        (root/name).write_bytes((source/name).read_bytes())
    atomic_json(root/'inventory.json',specification)
    atomic_json(root/'authenticated_models.json',auth)
    module=Path(__file__).parent
    files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_anticipatory_reference.py').resolve()]
    archive=root/'source_snapshot';archive.mkdir()
    for n,p in enumerate(sorted(files)):(archive/f'{n:03d}_{p.name}').write_bytes(p.read_bytes())
    plan=dict(project='STTW_CONTROL',purpose=specification['question'],data_role='DEV',
        maximum_active_lower_ticks=160000,optimizer_updates=0,batch_size=16,
        steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,
        stage_counts=dict(nominal=16,disturbed=64),
        sealed_files={str((root/name).resolve()):digest(root/name) for name in
            ['inventory.json','common_environment_config.json','source_declaration.json','authenticated_models.json']},
        source_sha256={str(p.resolve()):digest(p) for p in files},
        source_initial_state=specification['source_initial_state'],
        source_initial_state_sha256=specification['source_initial_state_sha256'],
        path_file=str((source/'reference_paths/right90_R1p1.npz').resolve()),
        path_sha256=digest(source/'reference_paths/right90_R1p1.npz'),
        contract='50Hz sample-held reference, four 5ms ticks; cumulative domain exit switches to live zero next upper boundary; real contact/fault terminates; original alpha/path/speed unchanged.',
        stop_rule='No automatic restart, expansion, extra training or horizon extension. Record faults as unknown. Nominal stage precedes disturbed stage.',
        inference='Opportunity screen in one related context; no novelty, causal selection or generalization claim from best eventwise candidate.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=80,completed_cases=0,
        active_lower_ticks=0,optimizer_updates=0,checked_unix=time.time()))


def execute(root,stage,*,shape_only=False):
    from prepare_short_recovery_campaign import register
    directory=root/(stage+'_shape' if shape_only else stage)
    claim_stage(directory)
    target=directory/'status.json'
    state=dict(phase='running',stage='authenticate',pid=os.getpid(),completed_cases=0,
        active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,checked_unix=time.time())
    def update(**kw):
        state.update(kw,checked_unix=time.time());atomic_json(target,state)
    update();register(directory/'ACTIVE_RUN.json',
        'STTW_CONTROL 提前参考'+stage+('零积分接口' if shape_only else '固定条件评测'),target)
    try:
        import jax
        import jax.numpy as j
        from .contact_only_environment import load_environment_contact_only
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate,validate_causal_contract
        from .anticipatory_reference import ReferenceEvaluationState,make_reference_step
        from .path_command_policy import make_path_actor
        from .runtime import configure_compilation_cache
        plan=json.loads((root/'plan.json').read_text())
        for name,sha in {**plan['sealed_files'],**plan['source_sha256'],
                         plan['source_initial_state']:plan['source_initial_state_sha256'],
                         plan['path_file']:plan['path_sha256']}.items():
            if digest(name)!=sha:raise ValueError('changed sealed source '+name)
        if stage=='disturbed' and not shape_only:
            nominal=json.loads((root/'nominal/status.json').read_text())
            if nominal['phase']!='completed' or nominal['completed_cases']!=16:
                raise ValueError('complete nominal stage required first')
        inventory=json.loads((root/'inventory.json').read_text());cases=inventory['cases']
        indices=stage_indices(cases,stage);update(declared_cases=len(indices))
        cfg=json.loads((root/'common_environment_config.json').read_text())
        auth=json.loads((root/'authenticated_models.json').read_text())
        declaration=json.loads((root/'source_declaration.json').read_text())
        env,_=load_environment_contact_only(cfg,declaration['source'],'terminal_contract_v2')
        validate_causal_contract(cfg,env.plant_spec['network']['frame_fields'])
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        with Path(plan['source_initial_state']).open('rb') as f:source=pickle.load(f).state
        with np.load(plan['path_file']) as z:
            path=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        parameters={}
        for role in {cases[i]['actor_role'] for i in indices}:
            model=auth['models'][role]
            for kind in ['actor','checkpoint']:
                if model.get(kind) and digest(model[kind])!=model[kind+'_sha256']:
                    raise ValueError('changed frozen model '+role+'/'+kind)
            with Path(model['actor']).open('rb') as f:parameters[role]=jax.tree.map(j.asarray,pickle.load(f))
        first=jax.tree.map(lambda a:j.asarray(a[cases[indices[0]]['source_batch_slot']]),source)
        env.set_log_template(first,path,j.float32(1.5))
        def build_chunk(step):
            @jax.jit
            def chunk(c,p,paths,speeds,pulses,profiles,floors,zero):
                return jax.lax.scan(lambda st,_:jax.vmap(step)(st,p,paths,speeds,pulses,profiles,floors,zero),c,None,length=50)
            return chunk
        stack=lambda values:jax.tree.map(lambda *x:j.stack(x),*values)
        results=[];index=[];traces=[];began=time.monotonic();compiled={}
        (directory/'cases').mkdir()
        groups=architecture_groups(indices,cases,auth['models'],cfg,batch_size=plan['batch_size'])
        for group,piece in enumerate(groups):
            ids=piece['case_indices'];selected=[cases[i] for i in ids];n=len(ids)
            key=(piece['architecture'],n)
            if key not in compiled:
                compiled[key]=build_chunk(make_reference_step(env,make_path_actor(piece['actor_config']),horizon_s=10.))
            chunk=compiled[key]
            states=[jax.tree.map(lambda a:j.asarray(a[c['source_batch_slot']]),source) for c in selected]
            if any(int(s.tick)!=0 or bool(s.physical.failed|s.fault|s.domain_exit) for s in states):
                raise ValueError('invalid complete task0 initial state')
            if plan.get('allow_task0_alpha_context_override',False):
                states=[s.replace(alpha=j.float32(c['state_alpha'])) for s,c in zip(states,selected)]
            if any(abs(float(s.alpha)-c['state_alpha'])>1e-7 for s,c in zip(states,selected)):
                raise ValueError('source user-alpha mismatch')
            carry=ReferenceEvaluationState(stack(states),stack([initial_gate() for _ in states]))
            p=stack([parameters[c['actor_role']] for c in selected]);paths=stack([path]*n)
            speeds=j.asarray([c['requested_speed_m_s'] for c in selected],j.float32)
            start=np.asarray([float(s.physical.physical_tick)*.005 for s in states])
            pulses=j.asarray([[t+c['pulse_onset_task_tick']*.005,t+c['pulse_onset_task_tick']*.005+c['pulse_duration_s'],
                *(c['force_N']*np.asarray(c['world_direction']))] for t,c in zip(start,selected)],j.float32)
            profiles=j.asarray([c['profile'] for c in selected],j.int32)
            floors=j.asarray([c['reference_speed_floor_m_s'] for c in selected],j.float32)
            zero=j.asarray([c['zero_mode'] for c in selected])
            args=(p,paths,speeds,pulses,profiles,floors,zero)
            if shape_only:
                shaped=jax.eval_shape(chunk,carry,*args)
                assert shaped[1]['active_tick'].shape==(50,n,4)
                traces.append(dict(group=group,batch=n,architecture=piece['architecture'],active_shape=list(shaped[1]['active_tick'].shape)))
                continue
            if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
            gd=directory/f'group_{group:03d}';gd.mkdir()
            with (gd/'initial.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            chunks=[]
            for block in range(10):
                update(stage='collecting',group=group,block=block)
                carry,logs=chunk(carry,*args);jax.block_until_ready(carry)
                host,data=jax.device_get((carry,logs));flat=flatten_ticks(data);chunks.append(flat)
                np.savez_compressed(gd/f'chunk_{block:02d}.npz',**flat)
                with (gd/'latest_state.pkl').open('wb') as f:pickle.dump(host,f)
                update(active_lower_ticks=state['active_lower_ticks']+int(flat['active_tick'].sum()),
                    allocated_lower_slots=state['allocated_lower_slots']+200*n,elapsed_s=time.monotonic()-began)
                if state['active_lower_ticks']>2000*len(indices):raise ValueError('stage physical budget exceeded')
                if np.any(host.state.fault):raise ValueError('engineering fault; raw evidence saved')
                if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
            else:
                if not np.all(host.state.physical.failed|(host.state.tick>=2000)):
                    raise ValueError('unfinished stage horizon')
            for lane,(case_id,c) in enumerate(zip(ids,selected)):
                data={k:np.concatenate([a[k][:,lane] for a in chunks]) for k in chunks[0]}
                file=directory/'cases'/f'case_{case_id:03d}.npz';np.savez_compressed(file,**data)
                fault=data['nonfinite']|data['lower_fault']|data['policy_fault']
                outcome=physical_outcome(data['active_tick'],data['contact_failure'],fault,2000)
                results.append(dict(c,physical_outcome=outcome,physical_ticks=int(data['active_tick'].sum())))
                pose=np.asarray(env.physics.helpers.pose(states[lane].physical.data))
                index.append(dict(case_id=c['case_id'],file=str(file.relative_to(root)),sha256=digest(file),
                    initial_xy=pose[:2].tolist(),initial_yaw=float(pose[2])))
            atomic_json(directory/'results.json',results);atomic_json(directory/'case_index.json',index)
            update(completed_cases=len(results))
        if shape_only:atomic_json(directory/'receipt.json',dict(groups=traces,physics_steps=0,optimizer_updates=0))
        elif len(results)!=len(indices) or any(r['physical_outcome']=='unknown' for r in results):
            raise ValueError('incomplete stage matrix')
        update(phase='completed',stage='interface_traced' if shape_only else 'collection_complete_analysis_pending')
        if not shape_only:
            stages=[json.loads((root/s/'status.json').read_text()) for s in plan['stage_counts'] if (root/s/'status.json').exists()]
            atomic_json(root/'status.json',dict(phase='completed' if len(stages)==len(plan['stage_counts']) and all(s['phase']=='completed' for s in stages) else 'partial',
                completed_cases=sum(s['completed_cases'] for s in stages),declared_cases=sum(plan['stage_counts'].values()),
                active_lower_ticks=sum(s['active_lower_ticks'] for s in stages),optimizer_updates=0,checked_unix=time.time()))
    except BaseException as exc:
        (directory/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
