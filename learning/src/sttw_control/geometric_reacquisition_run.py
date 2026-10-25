"""Bounded complete-state geometry suffixes; no task0 reset or repeated pulse."""
import json,os,pickle,time,traceback,copy
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest


def extension_cases(states):
    if len(states)!=2:raise ValueError('two declared complete states required')
    return [dict(s,source_state_index=i,case_id=f'snapshot_{i:02d}/{role}',role=role,mode=mode,
        event_id=f'snapshot_{i:02d}',stage='extension',alpha=s['user_state_alpha'],
        state_alpha=s['user_state_alpha'],source_case_alpha=s['original_case_alpha'],
        horizon_s=22.,suffix_s=12.,suffix_force_N=0.,force_N=0.,pulse_onset_task_tick=0,pulse_duration_s=0.,
        route='right90_R1p1',case_unit='new12s suffix plus cached10s prefix')
        for i,s in enumerate(states) for mode,role in enumerate(['live_zero','geo_fixed','geo_adaptive'])]


def prepare(inventory,root):
    from .tolerance_shaping import tolerance_enabled
    inventory=inventory.resolve();root=root.resolve();spec=json.loads(inventory.read_text())
    cfg=json.loads(Path(spec['common_config']).read_text());declaration=json.loads(Path(spec['source_declaration']).read_text())
    if digest(spec['common_config'])!=spec['config_sha256'] or digest(spec['source_declaration'])!=spec['source_declaration_sha256'] or digest(spec['path_file'])!=spec['path_sha256']:
        raise ValueError('source config/path/declaration drift')
    if tolerance_enabled(cfg):raise ValueError('horizon-dependent shaping requires separate prefix contract')
    if cfg['training_future_phase_C']['episode_s']!=10.:raise ValueError('original10s prefix required')
    files={str(inventory):digest(inventory),spec['common_config']:spec['config_sha256'],spec['source_declaration']:spec['source_declaration_sha256']}
    prefixes={}
    for i,s in enumerate(spec['states']):
        for path,key in [(s['extracted_file'],'extracted_sha256'),(s['raw_case'],'raw_sha256')]:
            if digest(path)!=s[key]:raise ValueError('source complete-state/raw drift')
            files[path]=s[key]
        stage=Path(s['raw_case']).parents[1]
        index={r['case_id']:r for r in json.loads((stage/'case_index.json').read_text())}
        old={r['case_id']:r for r in json.loads((stage/'results.json').read_text())}
        if old[s['source_case_id']]['physical_outcome']!='survived':raise ValueError('healthy full10s only')
        prefixes[i]=dict(index=index[s['source_case_id']],case=old[s['source_case_id']])
        with Path(s['extracted_file']).open('rb') as f:c=pickle.load(f)
        if int(c.state.tick)!=2000 or int(c.state.physical.physical_tick)!=2700 or bool(c.state.physical.failed|c.state.fault):
            raise ValueError('complete source clock/health mismatch')
    lower=declaration['source']['candidate']
    for kind,key in [('actor','actor_sha256'),('checkpoint','checkpoint_sha256'),('config','config_file_sha256')]:
        if digest(lower[kind])!=lower[key]:raise ValueError('frozen local300 identity drift')
        files[lower[kind]]=lower[key]
    cases=extension_cases(spec['states'])
    for c in cases:
        old=prefixes[c['source_state_index']]['case']
        c.update(force_N=old['force_N'],pulse_onset_task_tick=old['pulse_onset_task_tick'],pulse_duration_s=old['pulse_duration_s'],
            requested_speed_m_s=old['requested_speed_m_s'],prefix_initial_xy=prefixes[c['source_state_index']]['index']['initial_xy'])
    root.mkdir(parents=True,exist_ok=False)
    (root/'source_environment_config.json').write_bytes(Path(spec['common_config']).read_bytes())
    runtime=copy.deepcopy(cfg);runtime['training_future_phase_C']['episode_s']=22.
    atomic_json(root/'common_environment_config.json',runtime)
    (root/'source_declaration.json').write_bytes(Path(spec['source_declaration']).read_bytes())
    atomic_json(root/'inventory.json',spec);atomic_json(root/'cases.json',cases)
    sources=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/evaluate_geometry_reacquisition.py').resolve()]
    archive=root/'source_snapshot';archive.mkdir()
    for i,p in enumerate(sorted(sources)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    plan=dict(project='STTW_CONTROL',data_role='DEV',purpose='Is there legal geometric reacquisition opportunity after demonstrated physical survival, beyond same-horizon livezero/fixed6m?',
        stage_counts=dict(extension=6),logical_cases=6,inherited_cases=0,new_case_unit='12s physical suffix; full22s rows include cached10s prefix',
        inherited_unique_prefixes=2,inherited_unique_prefix_ticks=4000,inherited_logical_prefix_copies=6,
        maximum_active_lower_ticks=14400,optimizer_updates=0,start_task_tick=2000,start_physical_tick=2700,end_task_tick=4400,horizon_s=22.,suffix_s=12.,
        batch_size=6,steps_per_chunk=50,max_chunks=12,comparison_baseline_role='live_zero',
        path_file=spec['path_file'],path_sha256=spec['path_sha256'],
        sealed_files={**files,**{str((root/n).resolve()):digest(root/n) for n in ['source_environment_config.json','common_environment_config.json','source_declaration.json','inventory.json','cases.json']}},
        source_sha256={str(p.resolve()):digest(p) for p in sources},
        information='Current pose/speed/progress and original path only. New force pulse explicitly zeros; no reset or future event information.',
        contract='Original prefix10s unchanged/failed; new same22s mission for all3 suffix rules. Floors/filters/history/lower unchanged. No old Actor query; original PP publisher remains active.',
        stop_rule='Six suffixes only; no extra force, horizon/candidate expansion, optimizer, automatic retry or parameter tuning.',
        inference='Two selected related DEV end states, not independent new starts. Forward geometry is not a physical safety certificate; fixed rule coverage rejects method novelty claim.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=6,completed_cases=0,new_suffix_lower_ticks=0,active_lower_ticks=0,optimizer_updates=0,checked_unix=time.time()))


def execute(root,*,shape_only=False):
    from prepare_short_recovery_campaign import register
    from .anticipatory_run import claim_stage,flatten_ticks,physical_outcome
    directory=root/('extension_shape' if shape_only else 'extension');claim_stage(directory)
    target=directory/'status.json';state=dict(phase='running',stage='authenticate',pid=os.getpid(),declared_cases=6,completed_cases=0,
        active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,checked_unix=time.time())
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(target,state)
    update();register(directory/'ACTIVE_RUN.json','STTW_CONTROL 几何再捕获六条后缀'+('零积分接口' if shape_only else '真实物理对照'),target)
    try:
        import jax
        import jax.numpy as j
        from .contact_only_environment import load_environment_contact_only
        from .geometric_path import Path as GeometryPath
        from .geometric_reacquisition import make_geometry_step,join_prefix_suffix
        from .runtime import configure_compilation_cache
        began=time.monotonic();plan=json.loads((root/'plan.json').read_text())
        for path,sha in {**plan['sealed_files'],**plan['source_sha256'],plan['path_file']:plan['path_sha256']}.items():
            if digest(path)!=sha:raise ValueError('changed sealed source '+path)
        cfg=json.loads((root/'common_environment_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2')
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        cases=json.loads((root/'cases.json').read_text());carries=[]
        for c in cases:
            with Path(c['extracted_file']).open('rb') as f:carry=jax.tree.map(j.asarray,pickle.load(f))
            s=carry.state
            if int(s.tick)!=2000 or int(s.physical.physical_tick)!=2700 or bool(s.physical.failed|s.fault):raise ValueError('no reset/no resurrection source contract')
            carries.append(carry)
        with np.load(plan['path_file']) as z:path=GeometryPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        env.set_log_template(carries[0].state,path,j.float32(1.5))
        stack=lambda xs:jax.tree.map(lambda *v:j.stack(v),*xs)
        carry=stack(carries);paths=stack([path]*6);modes=j.asarray([c['mode'] for c in cases],j.int32)
        speeds=j.asarray([c['requested_speed_m_s'] for c in cases],j.float32);floors=j.asarray([c['reference_speed_floor_m_s'] for c in cases],j.float32)
        pulses=j.zeros((6,5),j.float32) # Explicitly no new pulse; do not add old onset to currentphysical time.
        one=make_geometry_step(env,horizon_s=22.)
        @jax.jit
        def chunk(c):
            return jax.lax.scan(lambda s,_:jax.vmap(one)(s,modes,paths,speeds,pulses,floors),c,None,length=50)
        if shape_only:
            shaped=jax.eval_shape(chunk,carry);assert shaped[1]['active_tick'].shape==(50,6,4)
            atomic_json(directory/'receipt.json',dict(active_shape=[50,6,4],start_task_ticks=[int(c.state.tick) for c in carries],start_physical_ticks=[int(c.state.physical.physical_tick) for c in carries],physics_steps=0,optimizer_updates=0,explicit_pulse=pulses.tolist()))
            update(phase='completed',stage='interface_traced');return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        gd=directory/'batch';gd.mkdir();(directory/'suffix_cases').mkdir();(directory/'cases').mkdir()
        with (gd/'initial.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        update(stage='collecting',initialization_s=time.monotonic()-began);chunks=[]
        for block in range(12):
            update(block=block);carry,logs=chunk(carry);jax.block_until_ready(carry)
            host,raw=jax.device_get((carry,logs));flat=flatten_ticks(raw);chunks.append(flat)
            np.savez_compressed(gd/f'chunk_{block:02d}.npz',**flat)
            with (gd/'latest_state.pkl').open('wb') as f:pickle.dump(host,f)
            count=int(flat['active_tick'].sum());update(active_lower_ticks=state['active_lower_ticks']+count,allocated_lower_slots=state['allocated_lower_slots']+1200,elapsed_s=time.monotonic()-began)
            if state['active_lower_ticks']>14400:raise ValueError('suffix budget exceeded')
            if np.any(flat['applied_force_impulse'][flat['active_tick']]):raise ValueError('unexpected new force')
            if np.any(host.state.fault):raise ValueError('engineering fault; preserve actual prefix/chunk')
            if np.all(host.state.physical.failed|(host.state.tick>=4400)):break
        if not np.all(host.state.physical.failed|(host.state.tick>=4400)):raise ValueError('unfinished fixed22s horizon')
        results=[];indices=[];suffix_indices=[]
        for lane,c in enumerate(cases):
            suffix={k:np.concatenate([x[k][:,lane] for x in chunks]) for k in chunks[0]}
            sf=directory/'suffix_cases'/f'case_{lane:03d}.npz';np.savez_compressed(sf,**suffix)
            with np.load(c['raw_case']) as z:prefix={k:z[k] for k in z.files}
            data=join_prefix_suffix(prefix,suffix);file=directory/'cases'/f'case_{lane:03d}.npz';np.savez_compressed(file,**data)
            fault=data['nonfinite']|data['lower_fault']|data['policy_fault']
            outcome=physical_outcome(data['active_tick'],data['contact_failure'],fault,4400)
            results.append(dict(c,physical_outcome=outcome,physical_ticks=int(data['active_tick'].sum()),new_suffix_ticks=int(suffix['active_tick'].sum()),inherited_prefix_ticks=2000,handover_s=10.,handover_xy=c['last_xy']))
            indices.append(dict(case_id=c['case_id'],file=str(file.relative_to(root)),sha256=digest(file),initial_xy=c['prefix_initial_xy']))
            suffix_indices.append(dict(case_id=c['case_id'],file=str(sf.relative_to(root)),sha256=digest(sf),new_ticks=int(suffix['active_tick'].sum()),source_prefix=c['raw_case'],source_prefix_sha256=c['raw_sha256']))
        atomic_json(directory/'results.json',results);atomic_json(directory/'case_index.json',indices);atomic_json(directory/'suffix_index.json',suffix_indices)
        if len(results)!=6 or any(r['physical_outcome']=='unknown' for r in results):raise ValueError('incomplete six suffixes')
        update(phase='completed',stage='collection_complete_analysis_pending',completed_cases=6,elapsed_s=time.monotonic()-began)
        atomic_json(root/'status.json',dict(phase='completed',completed_cases=6,declared_cases=6,active_lower_ticks=state['active_lower_ticks'],allocated_lower_slots=state['allocated_lower_slots'],inherited_unique_prefix_ticks=4000,optimizer_updates=0,checked_unix=time.time()))
    except BaseException as exc:
        (directory/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc))
        if not shape_only:atomic_json(root/'status.json',dict(state,execution_status='extension/status.json'))
        raise
