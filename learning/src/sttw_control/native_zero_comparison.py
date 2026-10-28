"""Native zero/static-reference controls; imports use sealed old source.

No old snapshot is patched. This runner calls its original reset, observation,
and policy_step with an explicit zero two-channel neural target. The nominal
follower, lower network, observer, filters, limits and native stops remain.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)


def prepare(source, root):
    source=Path(source).resolve();root=Path(root).resolve()
    status=json.loads((source/'evaluation/status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=8:
        raise ValueError('completed native current8 required, no baseline rerun')
    p=json.loads((source/'plan.json').read_text());seal=json.loads((source/'prelaunch_receipt.json').read_text())
    for f,k in [('plan.json','plan_sha256'),('declaration.json','declaration_sha256'),('cache_receipt.json','cache_sha256')]:
        if digest(source/f)!=seal[k]:raise ValueError('native declaration changed '+f)
    for f,sha in p['native_runtime_files'].items():
        if digest(f)!=sha:raise ValueError('archived native source changed '+f)
    review=json.loads((source/'analysis/review.json').read_text())
    if len(review['records'])!=16:raise ValueError('all native16 cached records required')
    for r in review['records']:
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('cached native raw changed')
    old=json.loads((source/'declaration.json').read_text())
    if len(old['routes'])!=2 or old['speeds']!=[2.6,2.8] or old['alphas']!=[0.,1.] or old['forces_N']!=[0.]:
        raise ValueError('original fixed native four-context panel required')
    if any(p[k] for k in ('feedback','short_support','goal_preview')):
        raise ValueError('no modern support or geometric feedback allowed')
    # Native10s only: this isolates neural preference from the previous20s
    # replanning comparison and never relabels its physics as native evidence.
    assert old['config']['training_future_phase_C']['episode_s']==10.
    cases=[dict(route=r['name'],builder=r['builder'],speed=v,alpha=a,policy='zero',force_N=0.,onset_s=1.)
        for r,v,a in itertools.product(old['routes'],old['speeds'],old['alphas'])]
    files=[source/n for n in ('plan.json','declaration.json','prelaunch_receipt.json','cache_receipt.json','analysis/review.json','evaluation/manifest.json')]
    cache=json.loads((source/'cache_receipt.json').read_text())
    files.append(Path(cache['source_run'])/'manifest.json')
    own=[Path(__file__).resolve(),Path('learning/cli/run_native_zero_comparison.py').resolve()]
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in own:(root/'source'/f.name).write_bytes(f.read_bytes())
    save(root/'cached_records.json',review['records'])
    files.append(root/'cached_records.json')
    plan=dict(project='STTW_CONTROL',experiment_id='native_zero_preference_v1',source=str(source),
        native_runtime_root=str(source/'native_runtime/learning/src'),native_tracked_commit=p['native_tracked_code_commit'],
        native_source_sha256=p['native_runtime_files'],runner_sha256={str(f):digest(f) for f in own},
        input_sha256={str(f):digest(f) for f in files},cases=cases,config=old['config'],initial_source=old['source'],
        cached_cases=16,new_cases=8,maximum_new_lower_ticks=16000,maximum_allocated_lower_slots=16000,
        horizon_s=10.,optimizer_updates=0,initialization='Original native prepared-bank snapshot and native env.reset; explicit zero neural target from tick0, no synthetic trained checkpoint',
        question='Does learned preference provide native task tradeoffs beyond the SAME ordinary follower and frozen lower?',
        data_role='Four already-seen native DEV contexts, shared ancestor; not independent TEST or training repeats.',
        control='Native follower/filter/slew/bounds/ECBC/ESO/local300; no modern feedback, short speed support, goal-preview or replan.',
        scoring='Original user0/user1 V3 costs reconstructed on original fixed route; 1m goal arrival before native physical/domain stop separately from whole-window survival and observed-window errors.',
        stop='Exactly8 nominal trajectories, native physical/domain/fault/10s stop; error preserves partial evidence, no implicit retry or tuning; no automatic adoption.')
    save(root/'plan.json',plan);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def capability_cases(protocol):
    import math
    import re
    contexts=protocol['contexts']
    if (protocol.get('horizon_s')!=10. or protocol.get('alphas')!=[0.,.5,1.] or len(contexts)!=4 or
        protocol.get('expected_cases')!=16 or protocol.get('maximum_lower_ticks')!=32000):
        raise ValueError('capability requires four contexts, H10, alpha0/.5/1 and explicit16/32000 budget')
    if len({c['route'] for c in contexts})!=4 or any(not re.fullmatch(r'[A-Za-z0-9_-]+',c['route']) for c in contexts):
        raise ValueError('capability unique safe route labels required')
    pairs={}
    for c in contexts:
        if any(not math.isfinite(c[k]) or c[k]<=0 for k in ('angle_deg','radius_m','speed')) or c['direction'] not in (-1.,1.):
            raise ValueError('capability finite positive geometry required')
        if not 1.5<=c['speed']<=3.5:raise ValueError('capability speed outside native reference bounds')
        key=tuple(c[k] for k in ('angle_deg','radius_m','speed'))
        pairs.setdefault(key,[]).append(c['direction'])
    if len(pairs)!=2 or any(sorted(v)!=[-1.,1.] for v in pairs.values()):
        raise ValueError('capability requires two left/right geometry pairs')
    return [dict(c,alpha=a,policy=role,force_N=0.,onset_s=1.) for c in contexts
            for role,a in [('H75',0.),('H75',.5),('H75',1.),('zero',0.)]]


def prepare_capability(source,root,protocol):
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve()
    old=json.loads((source/'plan.json').read_text())
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('native source seal changed')
    files={**old['files_sha256'],**old['native_source_sha256']}
    for f,sha in files.items():
        if digest(f)!=sha:raise ValueError('native dependency changed '+f)
    cfg=old['config'];initial=old['initial_source']
    if (old['h75']['actor_sha256']!='d10642082f74e730ce1e834608575eab04b04b16d7373784646499ad1cd2b0df' or
        old['h75']['checkpoint_sha256']!='86cebb390ccf4a17d13e0b0a0db02f6273ab87857162359a5650a7e50ec64cf2'):
        raise ValueError('frozen H75 identity differs')
    if (cfg['follower']['steer_reference_max_rad']!=.3 or
        cfg['residual_interface']['steer_reference_bound_rad']!=.35 or
        cfg['residual_interface']['speed_reference_bounds_m_s']!=[1.5,3.5] or
        cfg['timing']['lower_dt_s']!=.005 or cfg['timing']['upper_dt_s']!=.02 or
        cfg['timing']['physics_dt_s_expected']!=.0002 or cfg['timing']['physics_substeps_per_control_expected']!=25 or
        old['horizon_s']!=10.):raise ValueError('native bounds or timing differ')
    identity=[(initial['prepared_bank'],initial['prepared_bank_sha256'])]
    for owner in (old['h75'],initial['candidate']):
        identity.extend((owner[k],owner[k+'_sha256']) for k in ('actor','checkpoint'))
    identity.append((initial['candidate']['config'],initial['candidate']['config_file_sha256']))
    identity.extend((str(Path(initial['upper_run'])/('alpha'+a)/'frozen_config_stage52.json'),sha)
                    for a,sha in initial['upper_config_file_sha256'].items())
    for f,sha in identity:
        if digest(f)!=sha:raise ValueError('native model/initial source identity changed '+f)
    proposal=json.loads(protocol.read_text());cases=capability_cases(proposal)
    own=[Path(__file__).resolve(),Path('learning/cli/run_native_zero_comparison.py').resolve()]
    inputs=[source/'plan.json',source/'seal.json',protocol]+[Path(f) for f,_ in identity]
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in own:(root/'source'/f.name).write_bytes(f.read_bytes())
    plan=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode='native_capability_v1',
        source=str(source),native_runtime_root=old['native_runtime_root'],native_source_sha256=old['native_source_sha256'],
        runner_sha256={str(f):digest(f) for f in own},input_sha256={str(f):digest(f) for f in inputs},
        config=cfg,initial_source=initial,h75=old['h75'],cases=cases,contexts=proposal['contexts'],
        cached_cases=0,new_cases=16,maximum_new_lower_ticks=32000,maximum_allocated_lower_slots=32000,
        horizon_s=10.,optimizer_updates=0,target_policy='capability',
        scoring='Original native fixed full S path; native task/final hold and original XY goal separately; domain stop is censored physical survival, never S1.',
        control='Original native follower/filter/limits/local300, H75 at50Hz or zero. No planner, search, perturbation or training.')
    save(root/'cached_records.json',[]);save(root/'plan.json',plan)
    save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def verify(root):
    root=Path(root).resolve();p=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed plan changed')
    for f,sha in {**p['input_sha256'],**p['native_source_sha256'],**p['runner_sha256']}.items():
        if digest(f)!=sha:raise ValueError('sealed input/source changed '+f)
    for r in json.loads((root/'cached_records.json').read_text()):
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('cached raw changed')
    return p


def run(root, shape_only=False):
    root=Path(root).resolve();p=verify(root)
    if p.get('target_policy','zero') not in ('zero','equilibrium','capability'):raise ValueError('unknown target policy')
    if not shape_only and json.loads((root/'status.json').read_text())['phase']!='prepared':
        raise ValueError('prepared only; no implicit restart')
    native=Path(p['native_runtime_root'])
    # Refuse an accidentally imported modern package rather than silently using
    # an expanded low-speed controller in this native comparison.
    for name,module in list(sys.modules.items()):
        if name=='sttw_control' or name.startswith('sttw_control.'):
            f=getattr(module,'__file__',None)
            if f and not Path(f).resolve().is_relative_to(native):
                raise ValueError('modern sttw_control already imported: '+name)
    sys.path.insert(0,str(native))
    import jax
    import jax.numpy as j
    import numpy as np
    import pickle
    from sttw_control.path_command_training import load_environment
    from sttw_control.geometric_path import build_path
    from sttw_control.path_command_reward import configured_cost_keys
    from sttw_control.recovery_margin_scan import outcome_labels
    from sttw_control.runtime import configure_compilation_cache
    started=time.monotonic();out=root/('shape' if shape_only else 'evaluation');out.mkdir(exist_ok=False)
    status=dict(phase='initializing',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0)
    def update(**kw):
        status.update(kw,elapsed_s=time.monotonic()-started);save(out/'status.json',status)
        if not shape_only:save(root/'status.json',status)
    update()
    try:
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        cfg=p['config'];env,snapshot=load_environment(cfg,p['initial_source'],'terminal_contract_v2')
        cases=p['cases'];n=len(cases);capability=p.get('mode')=='native_capability_v1'
        assert n==(16 if capability else 8)
        pose=np.asarray(env.physics.helpers.pose(snapshot.data))
        if capability:
            from sttw_control.path_command_scenarios import _integrate_route
            bank={c['route']:_integrate_route(pose,cfg['path'],2,c['radius_m'],np.deg2rad(c['angle_deg']),
                  c['speed'],c['direction'],0.,0.).path for c in p['contexts']}
            receipt=[]
            for c in p['contexts']:
                start=cfg['path']['straight_prefix_m'];ramp=cfg['path']['curvature_transition_length_m']
                length=np.deg2rad(c['angle_deg'])*c['radius_m']+ramp
                receipt.append(dict(c,first_start_m=start,first_end_m=start+length,
                    second_start_m=start+length+3.,second_end_m=start+2*length+3.,
                    goal_s_m=float(bank[c['route']].goal)))
            save(out/'geometry_receipt.json',receipt)
        else:bank={c['route']:build_path(c['builder'],pose,cfg) for c in cases}
        paths=jax.tree.map(lambda *v:j.stack(v),*[bank[c['route']] for c in cases])
        speeds=j.asarray([c['speed'] for c in cases]);alphas=j.asarray([c['alpha'] for c in cases])
        states=jax.jit(jax.vmap(lambda path,v,a:env.reset(snapshot,path,v,a)))(paths,speeds,alphas)
        jax.block_until_ready(states)
        env.set_log_template(jax.tree.map(lambda x:x[0],states),bank[cases[0]['route']],speeds[0])
        keys=list(configured_cost_keys(cfg))
        actor=params=None
        neural=j.asarray([c['policy']=='H75' for c in cases])
        neural_indices=np.asarray([i for i,c in enumerate(cases) if c['policy']=='H75'],dtype=np.int32)
        if capability:
            from sttw_control.path_command_policy import make_path_actor
            actor=make_path_actor(cfg)
            with Path(p['h75']['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
        rule=None
        if p.get('target_policy')=='equilibrium':
            import importlib.util
            from sttw_control.direct_command_policy import q_ratio
            spec=importlib.util.spec_from_file_location('native_static_reference_rule',p['rule']['source'])
            rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
        keep=('active_tick','time','actual_xy','reference_xy','actual_forward_speed','wheel_speed_proxy','v_user',
              'path_cross_track','path_heading_error','path_progress','goal_section_signed_distance',
              'phi','phi_dot','actual_delta','peak_roll','physical_failure','domain_exit','policy_fault','lower_fault',
              'governed','nominal','applied_force_impulse','scored_tick_reward','latent_z','offsets','filtered_offset','target_offset','lower_action')
        template=jax.eval_shape(lambda st:env.policy_step(st,j.zeros(2),bank[cases[0]['route']],speeds[0],10.,j.zeros(5)),jax.tree.map(lambda x:x[0],states))[1]
        if capability and any(k not in template for k in ('active_tick','contact_failure','nonfinite')):
            raise ValueError('native capability requires active/contact/nonfinite logs')
        keep=keep+tuple(k for k in ('contact_failure','nonfinite') if k in template)
        def one(st,path,v,z):
            obs,_,fault=env.observation(st,path,v,10.)
            if rule is not None:
                z,_=rule.target(st.physical.raw,st.alpha,cfg,lambda speed:q_ratio(speed,env.cc),p['rule']['budget_rad'],p['rule']['grid_points'])
            end,log=env.policy_step(st.replace(fault=st.fault|fault),z,path,v,10.,j.zeros(5))
            return end,dict({k:log[k] for k in keep},effective_costs=j.stack([log['effective_components'][k] for k in keys],axis=-1))
        def chunk(st):
            def step(s,_):
                targets=j.zeros((n,2),dtype=s.filtered.dtype)
                if capability:
                    neural_states=jax.tree.map(lambda x:x[neural_indices],s)
                    neural_paths=jax.tree.map(lambda x:x[neural_indices],paths)
                    obs,_,_=jax.vmap(lambda state,path,v:env.observation(state,path,v,10.))(neural_states,neural_paths,speeds[neural_indices])
                    targets=targets.at[neural_indices].set(jax.vmap(lambda x:actor.apply(params,x))(obs))
                return jax.vmap(one)(s,paths,speeds,targets)
            return jax.lax.scan(step,st,None,length=50)
        shape=jax.eval_shape(chunk,states)
        if shape_only:
            save(out/'receipt.json',dict(phase='completed',physics_steps=0,optimizer_updates=0,
                native_runtime_root=str(native),active_tick_shape=list(shape[1]['active_tick'].shape),
                source_sha256=p['runner_sha256'],target_policy=p.get('target_policy','zero'),zero_target=rule is None and not capability))
            update(phase='completed');return
        for name,path in bank.items():np.savez_compressed(out/(name+'_path.npz'),**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
        with (out/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(states),f)
        save(out/'manifest.json',dict(config=cfg,source=p['initial_source'],cases=cases,reward_components=keys,
            native_runtime_root=str(native),model_identity=env.physics.bundle.identity,plan_sha256=digest(root/'plan.json'),
            cached_cases=p['cached_cases'],training_steps=0,target_policy=p.get('target_policy','zero'),target_semantics='Declared target before original filters; complete frozen lower remains'))
        before=time.monotonic()
        fn=jax.jit(chunk).lower(states).compile()
        update(compile_s=time.monotonic()-before,sampling_s=0.,inflight_block=None)
        saved=[];active=0
        chunks=out/'chunks';chunks.mkdir()
        for block in range(10):
            update(phase='running',inflight_block=block,allocated_lower_slots=(block+1)*n*200)
            before=time.monotonic()
            states,logs=fn(states);jax.block_until_ready(states)
            elapsed=time.monotonic()-before
            status['sampling_s']+=elapsed
            host=jax.device_get(logs);saved.append(host)
            # Keep the actually returned prefix before checking fault/raising.
            # These raw chunks are evidence, not reusable PPO experience.
            np.savez_compressed(chunks/f'chunk_{block:02d}.npz',**host)
            active+=int(host['active_tick'].sum())
            update(phase='running',completed_blocks=block+1,active_lower_ticks=active,allocated_lower_slots=(block+1)*n*200)
            if np.any(host['policy_fault']|host['lower_fault']) or np.any(np.asarray(states.fault)):
                raise RuntimeError('native engineering fault; no negative control label or automatic retry')
        logs={k:np.concatenate([b[k] for b in saved],axis=0) for k in saved[0]};rows=[]
        for i,c in enumerate(cases):
            raw={k:x[:,i].reshape((-1,)+x.shape[3:]) for k,x in logs.items()}
            raw={k:x[raw['active_tick'].astype(bool)] for k,x in raw.items()}
            fields=('latent_z','offsets','filtered_offset','target_offset','applied_force_impulse') if rule is None and c['policy']=='zero' else ('applied_force_impulse',)
            for f in fields:
                if np.any(raw[f]):raise ValueError('zero native baseline has nonzero '+f)
            name=f"{c['policy']}_{c['route']}_v{c['speed']:g}_a{c['alpha']:g}_t1_f+0"
            file=out/(name+'.npz');np.savez_compressed(file,**raw)
            b=cfg['stage_B_baseline'];labels=outcome_labels(raw,speed=c['speed'],goal_s=float(bank[c['route']].goal),working_roll=b['working_roll_rad'],horizon=10.,hold_s=b['hold_s'])
            rows.append(dict(c,name=name,raw_file=str(file),raw_sha256=digest(file),**labels))
        save(out/'results.json',dict(cases=rows))
        update(phase='completed',completed_cases=n,zero_target_checked_cases=sum(c['policy']=='zero' for c in cases) if rule is None else 0,inflight_block=None)
    except BaseException as e:
        update(phase='error',error=repr(e));raise
