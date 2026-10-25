"""Finite measured-sensor relay evaluation; no optimizer or event oracle."""
import json,os,pickle,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .causal_recovery_relay import relay_inventory


def prepare(root):
    from prepare_short_recovery_campaign import register
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 传感器触发短时恢复准备',root/'prepare_status.json')
    try:
        source=Path('runs/entry_recovery_factorial_v1').resolve()
        prior=json.loads((source/'plan.json').read_text())
        assert digest(source/'plan.json')==json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']
        status=json.loads((source/'status.json').read_text());analysis=json.loads((source/'analysis/status.json').read_text())
        assert status['phase']==analysis['phase']=='completed' and status['completed_cases']==analysis['completed_cases']==40
        for f,sha in {**prior['sealed_files'],**prior['source_sha256']}.items():assert digest(f)==sha,f
        for k,sha in [('source_initial_state','source_initial_state_sha256'),('path_file','path_sha256')]:assert digest(prior[k])==prior[sha]
        old=json.loads((source/'inventory.json').read_text())
        results=json.loads((source/'analysis/results.json').read_text())
        assert len(results)==40 and len({c['case_id'] for c in results})==40
        declared={c['case_id']:c for c in old['cases']};assert set(declared)=={c['case_id'] for c in results}
        for c in results:
            for k,v in declared[c['case_id']].items():assert c[k]==v
        cases=relay_inventory(results[0]['world_direction']);cache=[]
        from .anticipatory_run import physical_outcome
        for c in results:
            if c['mode'] not in ['AA','BB']:continue
            file=Path(c['raw_file']);assert digest(file)==c['raw_sha256']
            with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            active=raw['active_tick'].astype(bool);ticks=int(active.sum())
            assert np.array_equal(raw['physical_tick'][active],np.arange(ticks))
            outcome=physical_outcome(active,raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            assert outcome==c['physical_outcome'] and outcome!='unknown'
            # Reuse actual original pose coordinates; geometry analysis supplies initial pose only.
            cached=next((x for x in old['cached'] if x['case_id']==c['case_id']),None)
            if cached is not None:initial=cached['initial_xy'];yaw=cached['initial_yaw']
            else:
                ix=next(x for x in json.loads((source/'physical/case_index.json').read_text()) if x['case_id']==c['case_id'])
                assert source/ix['file']==file and ix['sha256']==c['raw_sha256']
                initial=ix['initial_xy'];yaw=ix['initial_yaw']
            cache.append(dict(case_index=len(cases),case_id=c['case_id'],source_case=c,source_file=str(file),source_file_sha256=c['raw_sha256'],initial_xy=initial,initial_yaw=yaw))
            cases.append(declared[c['case_id']])
        assert len(cache)==20 and len(cases)==50
        models=prior['models']
        for m in models.values():
            for k in ['actor','checkpoint']:assert digest(m[k])==m[k+'_sha256']
        for name in ['common_environment_config.json','source_declaration.json']:shutil.copyfile(source/name,root/name)
        atomic_json(root/'inventory.json',dict(cases=cases,new_case_indices=list(range(30)),cached=cache))
        atomic_json(root/'authenticated_models.json',dict(models=models))
        files=list(Path('learning/src/sttw_control').glob('*.py'))+[Path('learning/cli/evaluate_causal_relay.py'),Path('docs/superpowers/plans/2026-10-25-causal-recovery-relay.md')]
        sources={}
        for f in files:
            sources[str(f.resolve())]=digest(f);dest=root/'source_at_launch'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
        plan=dict(project='STTW_CONTROL',experiment='causal_recovery_relay_v1',data_role='selected_DEV_causal_control_opportunity',
            source_campaign=str(source),source_campaign_plan_sha256=digest(source/'plan.json'),models=models,
            declared_cases=50,new_cases=30,cached_cases=20,maximum_new_active_lower_ticks=60000,maximum_new_allocated_lower_slots=60000,optimizer_updates=0,
            source_initial_state=prior['source_initial_state'],source_initial_state_sha256=prior['source_initial_state_sha256'],
            path_file=prior['path_file'],path_sha256=prior['path_sha256'],horizon_s=10.,lower_dt_s=.005,
            sealed_files={str(root/n):digest(root/n) for n in ['inventory.json','authenticated_models.json','common_environment_config.json','source_declaration.json']},
            source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            information='Current measured roll rate sampled before advance; existing .8/.4rad/s,.6s gate. No force/onset/direction input to selection.',
            contract='Same complete task0/physical700 slot0 and originalalpha; actual ESO/filter/history/state preserved; common legal16clean handback.',
            stop='Exactly30 new plus20 authenticated diagonals; no timing/threshold/force/reward tuning, optimizer or implicit retry.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=50,new_cases=30,cached_cases=20,completed_new_cases=0,completed_cases=20,new_active_lower_ticks=0,new_allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 因果接管30新物理对照',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 因果接管30路零积分shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),checked_unix=time.time()))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc());atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc),checked_unix=time.time()));raise

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
        from .causal_recovery_relay import CausalRelayState,initial_gate as initial_relay
        from .causal_recovery_relay import make_relay_step
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
        carry=CausalRelayState(stack(states),stack([initial_gate() for _ in states]),j.zeros(30,j.int32),stack([initial_relay() for _ in states]))
        paths=stack([path]*30);parameters=stack([params]*30);speeds=j.full(30,1.5,j.float32)
        modes=j.asarray([c['mode_index'] for c in cases],j.int32)
        pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+.1,
            *[c['force_N']*d for d in c['world_direction']]] for c in cases],j.float32)
        env.set_log_template(states[0],path,j.float32(1.5))
        one=make_relay_step(env,make_path_actor(cfg),horizon_s=10.)
        @jax.jit
        def chunk(carry,parameters,paths,speeds,pulses,modes):
            return jax.lax.scan(lambda st,_:jax.vmap(one)(st,parameters,paths,speeds,pulses,modes),carry,None,length=50)
        if shape_only:
            shaped=jax.eval_shape(chunk,carry,parameters,paths,speeds,pulses,modes);assert shaped[1]['active_tick'].shape==(50,30,4)
            atomic_json(out/'receipt.json',dict(lower_shape=[50,30,4],physics_steps=0,optimizer_updates=0))
            update(phase='completed',stage='shape_complete');return
        assert json.loads((root/'shape/status.json').read_text())['phase']=='completed'
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        with (out/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        logs=[];start=time.monotonic()
        for block in range(10):
            update(stage='collecting',block=block)
            carry,data=chunk(carry,parameters,paths,speeds,pulses,modes);jax.block_until_ready(carry);host,data=jax.device_get((carry,data));data=flatten_ticks(data);logs.append(data)
            np.savez_compressed(out/f'chunk_{block:02d}.npz',**data)
            with (out/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
            state['new_active_lower_ticks']+=int(data['active_tick'].sum());state['new_allocated_lower_slots']+=6000
            assert state['new_active_lower_ticks']<=60000 and state['new_allocated_lower_slots']<=60000
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
        if len(results)!=30 or any(r['physical_outcome']=='unknown' for r in results):raise ValueError('complete30 known physical outcomes required')
        update(phase='completed',stage='physical_complete_analysis_pending',completed_new_cases=30,completed_cases=50)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
