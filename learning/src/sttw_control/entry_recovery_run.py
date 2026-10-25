"""Sealed32-new/8-cache offline entry/recovery factorial, no optimizer."""
import json,os,pickle,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .entry_recovery_factorial import factorial_inventory



def validate_prior_records(status,declared,results,index,*,expected_count=100):
    """Associate runtime rows only with complete, unique sealed conditions."""
    if status.get('phase')!='completed' or status.get('completed_cases')!=expected_count:
        raise ValueError('completed prior route required')
    maps=[]
    for records in (declared,results,index):
        keyed={r['case_id']:r for r in records}
        if len(records)!=expected_count or len(keyed)!=expected_count:
            raise ValueError('unique complete prior records required')
        maps.append(keyed)
    if not maps[0].keys()==maps[1].keys()==maps[2].keys():
        raise ValueError('prior case identities differ')
    for case_id,condition in maps[0].items():
        for key,value in condition.items():
            if maps[1][case_id].get(key)!=value:
                raise ValueError(f'prior condition drift: {case_id}/{key}')
    return maps[2]


def prepare(root):
    from prepare_short_recovery_campaign import register
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 原新Actor进入恢复机制准备',root/'prepare_status.json')
    try:
        source=Path('runs/matched_live_ppo_v1_fixed64_evaluation').resolve()
        assert json.loads((source/'status.json').read_text())['phase']=='completed'
        campaign=json.loads((source/'plan.json').read_text());local=Path(campaign['routes']['right']['path'])
        assert digest(local/'plan.json')==campaign['routes']['right']['plan_sha256']
        prior=json.loads((local/'plan.json').read_text());auth=json.loads((local/'authenticated_models.json').read_text())
        for f,sha in {**prior['sealed_files'],**prior['source_sha256']}.items():assert digest(f)==sha,f
        for k,sha in [('source_initial_state','source_initial_state_sha256'),('path_file','path_sha256')]:assert digest(prior[k])==prior[sha]
        models={name:auth['models'][role] for name,role in [('A','parent_plain'),('B','trained_plain')]}
        cfg=json.loads((local/'common_environment_config.json').read_text())
        for m in models.values():
            assert m['runtime_kind']=='reentry' and m['actor_config']==cfg
            for k in ['actor','checkpoint']:assert digest(m[k])==m[k+'_sha256']
        results=json.loads((local/'reentry/results.json').read_text())
        index=validate_prior_records(json.loads((local/'reentry/status.json').read_text()),
            json.loads((local/'inventory.json').read_text())['cases'],results,
            json.loads((local/'reentry/case_index.json').read_text()))
        cases=factorial_inventory(results[0]['world_direction']);cache=[]
        for i,c in enumerate(cases):
            if not c['cache_candidate']:continue
            role='parent_plain' if c['mode']=='AA' else 'trained_plain'
            matches=[r for r in results if r['actor_role']==role and r['alpha']==c['alpha'] and r['force_N']==c['force_N']
                and (not c['force_N'] or r['pulse_onset_task_tick']==c['pulse_onset_task_tick'])]
            assert len(matches)==1,(c['case_id'],len(matches))
            old=matches[0]
            for key in ['profile','source_batch_slot','requested_speed_m_s','pulse_duration_s','world_direction']:
                assert old[key]==c[key],(c['case_id'],key)
            assert old['state_alpha']==c['alpha'] and old['reference_speed_floor_m_s']==cfg['residual_interface']['speed_reference_bounds_m_s'][0]
            item=index[old['case_id']];file=local/item['file'];assert digest(file)==item['sha256']
            from .anticipatory_run import physical_outcome
            with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            active=raw['active_tick'].astype(bool);ticks=int(active.sum())
            assert np.array_equal(raw['physical_tick'][active],np.arange(ticks))
            outcome=physical_outcome(active,raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            assert outcome==old['physical_outcome'] and outcome!='unknown' and ticks==old['physical_ticks']
            cache.append(dict(case_index=i,case_id=c['case_id'],source_case=old,source_file=str(file),
                source_file_sha256=item['sha256'],initial_xy=item['initial_xy'],initial_yaw=item['initial_yaw'],
                note='Exact diagonal controller/interface; zero-force onset metadata physically irrelevant, retained in source_case.'))
        assert len(cache)==8
        new_indices=[i for i,c in enumerate(cases) if not c['cache_candidate']];assert len(new_indices)==32
        for name in ['common_environment_config.json','source_declaration.json']:shutil.copyfile(local/name,root/name)
        atomic_json(root/'inventory.json',dict(cases=cases,new_case_indices=new_indices,cached=cache))
        atomic_json(root/'authenticated_models.json',dict(models=models))
        files=list(Path('learning/src/sttw_control').glob('*.py'))+[Path('learning/cli/evaluate_entry_recovery.py'),Path('docs/superpowers/plans/2026-10-25-entry-recovery-factorial.md')]
        sources={}
        for f in files:
            sources[str(f.resolve())]=digest(f);dest=root/'source_at_launch'/f
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
        plan=dict(project='STTW_CONTROL',experiment='entry_recovery_factorial_v1',data_role='posthoc_selected_DEV_mechanism',
            source_campaign=str(source),source_campaign_plan_sha256=digest(source/'plan.json'),source_route=str(local),
            source_route_plan_sha256=digest(local/'plan.json'),models=models,declared_cases=40,new_cases=32,cached_cases=8,
            maximum_new_active_lower_ticks=64000,maximum_new_allocated_lower_slots=64000,optimizer_updates=0,
            source_initial_state=prior['source_initial_state'],source_initial_state_sha256=prior['source_initial_state_sha256'],
            path_file=prior['path_file'],path_sha256=prior['path_sha256'],horizon_s=10.,lower_dt_s=.005,
            sealed_files={str(root/n):digest(root/n) for n in ['inventory.json','authenticated_models.json','common_environment_config.json','source_declaration.json']},
            source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            privilege='Known event-time handoff, including no-force metadata time; offline mechanism only, not causal deployment.',
            contract='Same full task0/physical700 slot0, only originalalpha context assigned; profile0, no state/history/ESO/filter reset; legal16clean handback.',
            stop='Exactly32 new plus8 authenticated cache; known contact is outcome, fault/unknown stops. No retry, threshold/time/force sweep or training.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=40,new_cases=32,cached_cases=8,completed_new_cases=0,
            completed_cases=8,new_active_lower_ticks=0,new_allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 原新策略进入恢复32新物理诊断',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 进入恢复32路零积分形状检查',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),checked_unix=time.time()))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc),checked_unix=time.time()));raise


def execute(root,*,shape_only=False):
    root=root.resolve();old=json.loads((root/'status.json').read_text())
    if old['phase']!='prepared':raise ValueError('fresh prepared run required; no implicit restart')
    out=root/('shape' if shape_only else 'physical');out.mkdir(exist_ok=False)
    file=out/'status.json' if shape_only else root/'status.json';state=dict(old,phase='running',stage='authenticate',pid=os.getpid())
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(file,state)
    update()
    try:
        import jax
        import jax.numpy as j
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate
        from .reference_reentry import ReentryEvaluationState
        from .entry_recovery_factorial import make_factorial_step
        from .anticipatory_run import flatten_ticks,physical_outcome
        p=json.loads((root/'plan.json').read_text());assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        for f,sha in {**p['sealed_files'],**p['source_sha256'],p['source_initial_state']:p['source_initial_state_sha256'],p['path_file']:p['path_sha256']}.items():assert digest(f)==sha,f
        inv=json.loads((root/'inventory.json').read_text());indices=inv['new_case_indices'];cases=[inv['cases'][i] for i in indices]
        for cached in inv['cached']:assert digest(cached['source_file'])==cached['source_file_sha256']
        cfg=json.loads((root/'common_environment_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        params={}
        for name,m in p['models'].items():
            for k in ['actor','checkpoint']:assert digest(m[k])==m[k+'_sha256']
            with Path(m['actor']).open('rb') as f:params[name]=jax.tree.map(j.asarray,pickle.load(f))
        assert jax.tree.structure(params['A'])==jax.tree.structure(params['B'])
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2')
        with Path(p['source_initial_state']).open('rb') as f:bank=pickle.load(f).state
        original=jax.tree.map(lambda x:j.asarray(x[0]),bank)
        assert int(original.tick)==0 and int(original.physical.physical_tick)==700 and not bool(original.physical.failed|original.fault|original.domain_exit)
        with np.load(p['path_file'],allow_pickle=False) as z:path=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        states=[original.replace(alpha=j.float32(c['alpha'])) for c in cases]
        carry=ReentryEvaluationState(stack(states),stack([initial_gate() for _ in states]),j.zeros(32,j.int32))
        paths=stack([path]*32);parameters=stack([params]*32);speeds=j.full(32,1.5,j.float32)
        modes=j.asarray([c['mode_index'] for c in cases],j.int32);handoffs=j.asarray([c['handoff_tick'] for c in cases],j.int32)
        pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+.1,
            *[c['force_N']*d for d in c['world_direction']]] for c in cases],j.float32)
        env.set_log_template(states[0],path,j.float32(1.5))
        one=make_factorial_step(env,make_path_actor(cfg),horizon_s=10.)
        @jax.jit
        def chunk(carry,parameters,paths,speeds,pulses,modes,handoffs):
            return jax.lax.scan(lambda st,_:jax.vmap(one)(st,parameters,paths,speeds,pulses,modes,handoffs),carry,None,length=50)
        if shape_only:
            shaped=jax.eval_shape(chunk,carry,parameters,paths,speeds,pulses,modes,handoffs);assert shaped[1]['active_tick'].shape==(50,32,4)
            atomic_json(out/'receipt.json',dict(lower_shape=[50,32,4],physics_steps=0,optimizer_updates=0))
            update(phase='completed',stage='shape_complete');return
        assert json.loads((root/'shape/status.json').read_text())['phase']=='completed'
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        with (out/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        logs=[];start=time.monotonic()
        for block in range(10):
            update(stage='collecting',block=block)
            carry,data=chunk(carry,parameters,paths,speeds,pulses,modes,handoffs);jax.block_until_ready(carry);host,data=jax.device_get((carry,data));data=flatten_ticks(data);logs.append(data)
            np.savez_compressed(out/f'chunk_{block:02d}.npz',**data)
            with (out/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
            state['new_active_lower_ticks']+=int(data['active_tick'].sum());state['new_allocated_lower_slots']+=6400
            assert state['new_active_lower_ticks']<=64000 and state['new_allocated_lower_slots']<=64000
            update(elapsed_s=time.monotonic()-start)
            if np.any(host.state.fault):raise RuntimeError('engineering fault; raw retained, no retry')
            if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
        data={k:np.concatenate([x[k] for x in logs]) for k in logs[0]};(out/'cases').mkdir()
        results=[];index=[]
        for lane,(i,c) in enumerate(zip(indices,cases)):
            raw={k:v[:,lane] for k,v in data.items()};target=out/'cases'/f'case_{i:03d}.npz';np.savez_compressed(target,**raw)
            outcome=physical_outcome(raw['active_tick'],raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            pose=np.asarray(env.physics.helpers.pose(states[lane].physical.data))
            results.append(dict(c,case_index=i,physical_outcome=outcome,physical_ticks=int(raw['active_tick'].sum())))
            index.append(dict(case_id=c['case_id'],file=str(target.relative_to(root)),sha256=digest(target),initial_xy=pose[:2].tolist(),initial_yaw=float(pose[2])))
        atomic_json(out/'results.json',results);atomic_json(out/'case_index.json',index)
        if len(results)!=32 or any(r['physical_outcome']=='unknown' for r in results):raise ValueError('complete32 known physical outcomes required')
        update(phase='completed',stage='physical_complete_analysis_pending',completed_new_cases=32,completed_cases=40)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
