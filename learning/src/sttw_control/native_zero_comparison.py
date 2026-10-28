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


def scheduling_cases(protocol, contexts):
    """Exactly the four completed S routes, with two H75 scheduling arms."""
    import importlib.util
    helper=Path(__file__).with_name('geometric_preference_schedule.py')
    spec=importlib.util.spec_from_file_location('native_schedule_validation',helper)
    rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
    rule.validate_rule(protocol['rule'])
    if (protocol.get('horizon_s')!=10. or protocol.get('original_user_alpha')!=1. or
        protocol.get('expected_cases')!=8 or protocol.get('maximum_lower_ticks')!=16000 or
        len(contexts)!=4):
        raise ValueError('schedule requires fixed user1, same four contexts, H10 and8/16000 budget')
    if 'contexts' in protocol and protocol['contexts']!=contexts:
        raise ValueError('scheduling cannot change completed capability contexts')
    # Reuse the original geometry validation without inventing new contexts.
    capability_cases(dict(contexts=contexts,horizon_s=10.,alphas=[0.,.5,1.],
                          expected_cases=16,maximum_lower_ticks=32000))
    return [dict(c,alpha=1.,original_user_alpha=1.,policy=arm,force_N=0.,onset_s=1.)
            for c in contexts for arm in ('alpha_schedule','curvature_speed')]


def channel_factorial_cases(protocol, contexts):
    """Two channel swaps only; source nominal user alpha remains one."""
    if (protocol.get('horizon_s') != 10. or protocol.get('original_user_alpha') != 1. or
        protocol.get('expected_cases') != 8 or protocol.get('maximum_lower_ticks') != 16000 or
        protocol.get('channel_alpha_pairs') != [[.5, 1.], [1., .5]] or
        any(protocol.get(k) is not False for k in ('training', 'disturbance', 'replanning'))):
        raise ValueError('channel factorial requires fixed user1, two hybrids, H10 and8/16000, no training/force/replan')
    if 'contexts' in protocol and protocol['contexts'] != contexts:
        raise ValueError('channel factorial cannot change cached contexts')
    capability_cases(dict(contexts=contexts,horizon_s=10.,alphas=[0.,.5,1.],
                          expected_cases=16,maximum_lower_ticks=32000))
    return [dict(c,alpha=1.,original_user_alpha=1.,speed_alpha=a,steer_alpha=b,
                 policy=f'speed{a:g}_steer{b:g}',force_N=0.,onset_s=1.)
            for c in contexts for a,b in protocol['channel_alpha_pairs']]


def channel_factorial_decision(state, path, speed, speed_alpha, observe, actor_apply, policy_step):
    """Query one current observation twice, then advance native physics once.

    Alpha slot336 is the native H75 unit-scale alpha; histories and state alpha
    are never rebuilt or replaced. This is an out-of-training action pairing.
    """
    import jax.numpy as j
    obs,_,fault=observe(state,path,speed,10.)
    if obs.shape != (351,):
        raise ValueError('channel factorial requires native351 observation')
    half=actor_apply(obs.at[336].set(.5))
    full=actor_apply(obs.at[336].set(1.))
    z=j.stack([j.where(speed_alpha == .5,half[0],full[0]),
               j.where(speed_alpha == .5,full[1],half[1])])
    end,logs=policy_step(state.replace(fault=state.fault|fault),z,path,speed,10.,j.zeros(5))
    shape=logs['active_tick'].shape
    def ticks(x):return j.broadcast_to(x,shape+j.asarray(x).shape)
    logs.update(queried_latent_alpha_half=ticks(half),queried_latent_alpha_one=ticks(full),
                composed_latent=ticks(z),channel_alpha_pair=ticks(j.array([speed_alpha,1.5-speed_alpha])),
                original_user_alpha=j.ones(shape))
    return end,logs


def prepare_channel_factorial(source, root, protocol):
    return prepare_scheduling(source, root, protocol, channel_factorial=True)


def prepare_scheduling(source, root, protocol, *, channel_factorial=False):
    """Seal a new nominal8 panel while retaining the completed native16."""
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve()
    old=json.loads((source/'plan.json').read_text())
    if (digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256'] or
        old.get('mode')!='native_capability_v1'):
        raise ValueError('sealed native capability source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=16:
        raise ValueError('completed native capability16 required; do not rerun it')
    for f,sha in {**old['input_sha256'],**old['native_source_sha256']}.items():
        if digest(f)!=sha:raise ValueError('capability source identity changed '+f)
    # Current runner now adds an opt-in branch. Verify the numerical runner used
    # for cached evidence against its immutable launch snapshot, not live code.
    archived_runners={}
    for f,sha in old['runner_sha256'].items():
        snapshot=source/'source'/Path(f).name
        if not snapshot.exists() or digest(snapshot)!=sha:
            raise ValueError('capability runner snapshot identity changed '+str(snapshot))
        archived_runners[str(snapshot)]=sha
    review=json.loads((source/'analysis/review.json').read_text())
    records=review['records']
    def key(c):return (c['route'],c['speed'],c['policy'],c['alpha'])
    if (len(records)!=16 or len({key(c) for c in records})!=16 or
            {key(c) for c in records}!={key(c) for c in old['cases']}):
        raise ValueError('complete cached16 capability case inventory required')
    for r in records:
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('cached capability raw changed')
    # Models, initial prepared carry and original route data are scientific inputs.
    identity={**old['input_sha256'],**old['native_source_sha256'],**archived_runners}
    for owner in (old['h75'],old['initial_source']['candidate']):
        for name in ('actor','checkpoint'):
            f=owner[name];sha=owner[name+'_sha256']
            if digest(f)!=sha:raise ValueError('native model identity changed '+f)
            identity[f]=sha
    proposal=json.loads(protocol.read_text());cases=(channel_factorial_cases if channel_factorial else scheduling_cases)(proposal,old['contexts'])
    helper=Path(__file__).with_name('geometric_preference_schedule.py').resolve()
    own=[Path(__file__).resolve(),Path('learning/cli/run_native_zero_comparison.py').resolve(),helper]
    scientific=[source/'plan.json',source/'seal.json',source/'analysis/review.json',
        source/'evaluation/status.json',source/'evaluation/manifest.json',source/'evaluation/results.json',
        source/'evaluation/initial_carry.pkl',protocol]
    scientific += [source/'evaluation'/(c['route']+'_path.npz') for c in old['contexts']]
    identity.update({str(f):digest(f) for f in scientific})
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in own:(root/'source'/f.name).write_bytes(f.read_bytes())
    save(root/'cached_records.json',records)
    identity[str(root/'cached_records.json')]=digest(root/'cached_records.json')
    plan=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode='native_channel_factorial_v1' if channel_factorial else 'native_scheduling_v1',
        source=str(source),cached_source=str(source),native_runtime_root=old['native_runtime_root'],
        native_source_sha256=old['native_source_sha256'],
        runner_sha256={str(f):digest(f) for f in own},input_sha256=identity,
        config=old['config'],initial_source=old['initial_source'],h75=old['h75'],cases=cases,contexts=old['contexts'],
        protocol=proposal,rule=dict(source=str(helper),source_sha256=digest(helper),**proposal.get('rule',{})),
        cached_cases=16,new_cases=8,maximum_new_lower_ticks=16000,maximum_allocated_lower_slots=16000,
        horizon_s=10.,optimizer_updates=0,target_policy='channel_factorial' if channel_factorial else 'scheduling',original_user_alpha=1.,
        cache_identity=dict(plan_sha256=digest(source/'plan.json'),records_sha256=digest(source/'analysis/review.json'),
            case_inventory=[dict(route=r['route'],speed=r['speed'],policy=r['policy'],alpha=r['alpha'],
                raw_file=r['raw_file'],raw_sha256=r['raw_sha256']) for r in records]),
        initialization='Original complete native prepared bank/reset, alpha1 and latchfalse; no state reset at scheduling transitions.',
        scoring='Fixed original user1/original v_user/original full S geometry; executed-alpha own native reward separately reconstructed.',
        control='Frozen H75 at50Hz; schedule alpha before observation OR cap H75 proposed speed via original latent mapping; same physical filters/rates/bounds/ESO/history/local300.',
        stop='Exactly8 nominal H10 trajectories, native stops;0force/0replan/0training; preserve error evidence without implicit retry/tuning.',
        optimizations=['same-shape vmap8','scan50 chunks','shared JAX compilation cache','single compiled function reused'])
    if channel_factorial:
        plan.update(initialization='Exact original native alpha1 full carry; same current observation/history for both queries.',
            scoring='Original user alpha1 and fixed original speed/full S path, native task and goal separately.',
            control='At50Hz query H75 alpha.5/1 on identical current observation except slot336; compose speed/steer latent before original native filters/limits and lower200Hz.',
            interpretation='Hybrid out-of-training action combinations; not a deployable scalar-alpha coordinator. Total closed-loop mapping replacement, not speed-matched causal effect.')
        plan.pop('rule')
    save(root/'plan.json',plan);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
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
    if p.get('target_policy','zero') not in ('zero','equilibrium','capability','scheduling','channel_factorial'):raise ValueError('unknown target policy')
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
        factorial=p.get('mode')=='native_channel_factorial_v1'
        scheduling=p.get('mode')=='native_scheduling_v1'
        cached_native=scheduling or factorial
        assert n==(16 if capability else 8)
        pose=np.asarray(env.physics.helpers.pose(snapshot.data))
        if capability or cached_native:
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
        if cached_native:
            # Same fixed original paths, not references altered by scheduling.
            for name,path in bank.items():
                with np.load(Path(p['source'])/'evaluation'/(name+'_path.npz'),allow_pickle=False) as archived:
                    for field in ('s','xy','heading','curvature','turn_end','goal'):
                        if not np.array_equal(np.asarray(getattr(path,field)),archived[field]):
                            raise ValueError('cached original path differs '+name+'/'+field)
        paths=jax.tree.map(lambda *v:j.stack(v),*[bank[c['route']] for c in cases])
        speeds=j.asarray([c['speed'] for c in cases]);alphas=j.asarray([c['alpha'] for c in cases])
        states=jax.jit(jax.vmap(lambda path,v,a:env.reset(snapshot,path,v,a)))(paths,speeds,alphas)
        jax.block_until_ready(states)
        if cached_native:
            # Compare ALL original closed-loop state leaves to the matching
            # cached alpha1 initialization. No filter/observer/history rebuild.
            with (Path(p['source'])/'evaluation/initial_carry.pkl').open('rb') as f:
                archived=pickle.load(f)
            old_cases=json.loads((Path(p['source'])/'plan.json').read_text())['cases']
            indices=[next(i for i,c in enumerate(old_cases) if c['route']==case['route'] and
                c['speed']==case['speed'] and c['policy']=='H75' and c['alpha']==1.) for case in cases]
            expected=jax.tree.map(lambda x:np.asarray(x)[indices],archived)
            aa,ta=jax.tree.flatten(jax.device_get(states));bb,tb=jax.tree.flatten(expected)
            differences=[]
            for i,(a,b) in enumerate(zip(aa,bb)):
                if a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True):continue
                delta=np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)) if a.shape==b.shape else None
                differences.append(dict(leaf=i,actual_shape=list(a.shape),expected_shape=list(b.shape),
                    dtype=str(a.dtype),maximum_abs_difference=None if delta is None else float(np.nanmax(delta))))
            if ta!=tb or differences:
                save(out/'initial_identity_receipt.json',dict(original_paths_exact=True,
                    all_initial_carry_leaves_exact=False,tree_structure_equal=ta==tb,
                    differences=differences,physics_steps=0))
                raise ValueError('cached native initial carry differs from original alpha1; see initial_identity_receipt.json')
            save(out/'initial_identity_receipt.json',dict(original_paths_exact=True,
                all_initial_carry_leaves_exact=True,leaves=len(aa),cached_alpha=1.,
                initial_state_alpha=1.,initial_schedule_alpha=1. if scheduling else None,
                initial_latch=False if scheduling else None,physics_steps=0))
        env.set_log_template(jax.tree.map(lambda x:x[0],states),bank[cases[0]['route']],speeds[0])
        keys=list(configured_cost_keys(cfg))
        actor=params=None
        neural_indices=np.asarray([i for i,c in enumerate(cases) if c['policy']=='H75' or cached_native],dtype=np.int32)
        if capability or cached_native:
            from sttw_control.path_command_policy import make_path_actor
            actor=make_path_actor(cfg)
            with Path(p['h75']['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
        rule=None;schedule=None
        if scheduling:
            import importlib.util
            spec=importlib.util.spec_from_file_location('native_geometric_preference_schedule',p['rule']['source'])
            schedule=importlib.util.module_from_spec(spec);spec.loader.exec_module(schedule)
            schedule.validate_rule(p['protocol']['rule'])
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
        if (capability or cached_native) and any(k not in template for k in ('active_tick','contact_failure','nonfinite')):
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
        if scheduling:
            is_alpha=j.asarray([c['policy']=='alpha_schedule' for c in cases])
            rule_config=p['protocol']['rule']
            def schedule_chunk(carry):
                def step(carry,_):
                    s,latched=carry
                    def decision(st,path,v,latch):
                        return schedule.advance(st.alpha,latch,v,path.s,path.curvature,
                                                st.projection.progress,rule_config)
                    alpha,latch,diagnostics=jax.vmap(decision)(s,paths,speeds,latched)
                    enabled=~(s.physical.failed|s.fault|s.domain_exit)
                    latch=j.where(enabled,latch,latched)
                    alpha=j.where(enabled&is_alpha,alpha,s.alpha)
                    s=s.replace(alpha=alpha)
                    def act(st,path,v):
                        obs,_,fault=env.observation(st,path,v,10.)
                        return actor.apply(params,obs),fault
                    targets,fault=jax.vmap(act)(s,paths,speeds)
                    def cap(z,st,cap,active,alpha_arm):
                        return schedule.cap_speed_latent(z,st.physical.raw,cap,
                                                         active&~alpha_arm,cfg)
                    targets,speed_audit=jax.vmap(cap)(targets,s,diagnostics['rule_speed_target'],latch,is_alpha)
                    s=s.replace(fault=s.fault|fault)
                    end,logs=jax.vmap(one)(s,paths,speeds,targets)
                    # Persist audit quantities per actual physical tick. Latents
                    # are held over four ticks; physical.raw continues its normal
                    # native publish updates and is never overwritten by the rule.
                    def ticks(x):return j.broadcast_to(x[:,None],logs['active_tick'].shape)
                    logs.update({k:ticks(v) for k,v in {**diagnostics,**speed_audit}.items()})
                    logs['rule_active']=ticks(latch)
                    logs['original_user_alpha']=j.ones_like(logs['time'])
                    logs['executed_alpha']=ticks(alpha)
                    logs['rule_applied_prefilter_target']=logs['nominal']+logs['target_offset']
                    # Native chi is a post-step geometry weight; preserve it for
                    # executed-alpha reward verification and original-user rescore.
                    # Reconstruct chi from the same logged progress and fixed path.
                    from sttw_control.direct_command_policy import q_ratio
                    def chi_of(path,progress,v):
                        k=j.interp(progress,path.s,path.curvature)
                        ff=j.arctan(env.cc.wheelbase*k/j.cos(env.cc.caster))
                        return j.clip((j.abs(ff/q_ratio(v,env.cc))-.18)/.12,0.,1.)
                    chi=jax.vmap(chi_of)(paths,logs['path_progress'],speeds)
                    logs['reward_chi']=chi
                    return (end,latch),logs
                return jax.lax.scan(step,carry,None,length=50)
            chunk=schedule_chunk
            states=(states,j.zeros(n,dtype=j.bool_))
        if factorial:
            context_fields=cfg['network']['context_fields']
            if (len(context_fields)!=15 or context_fields[0] != dict(name='alpha',scale=1.)):
                raise ValueError('native alpha slot336 identity differs')
            speed_alpha=j.asarray([c['speed_alpha'] for c in cases])
            def factorial_chunk(st):
                def step(s,_):
                    end,logs=jax.vmap(lambda state,path,v,a:channel_factorial_decision(
                        state,path,v,a,env.observation,lambda obs:actor.apply(params,obs),env.policy_step))(
                            s,paths,speeds,speed_alpha)
                    costs=j.stack([logs['effective_components'][k] for k in keys],axis=-1)
                    audit=('queried_latent_alpha_half','queried_latent_alpha_one','composed_latent',
                           'channel_alpha_pair','original_user_alpha')
                    return end,dict({k:logs[k] for k in keep+audit},effective_costs=costs)
                return jax.lax.scan(step,st,None,length=50)
            chunk=factorial_chunk
        shape=jax.eval_shape(chunk,states)
        if shape_only:
            save(out/'receipt.json',dict(phase='completed',physics_steps=0,optimizer_updates=0,
                native_runtime_root=str(native),active_tick_shape=list(shape[1]['active_tick'].shape),
                source_sha256=p['runner_sha256'],target_policy=p.get('target_policy','zero'),zero_target=rule is None and not capability and not cached_native))
            update(phase='completed');return
        for name,path in bank.items():np.savez_compressed(out/(name+'_path.npz'),**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
        with (out/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(states[0] if scheduling else states),f)
        manifest=dict(config=cfg,source=p['initial_source'],cases=cases,reward_components=keys,
            native_runtime_root=str(native),model_identity=env.physics.bundle.identity,plan_sha256=digest(root/'plan.json'),
            cached_cases=p['cached_cases'],training_steps=0,target_policy=p.get('target_policy','zero'),target_semantics='Declared target before original filters; complete frozen lower remains')
        if scheduling:manifest.update(original_user_alpha=1.,schedule_rule=p['rule'])
        if factorial:manifest.update(original_user_alpha=1.,channel_alpha_pairs=p['protocol']['channel_alpha_pairs'],
            interpretation=p['interpretation'],observation_alpha_slot=336,observation_alpha_scale=1.)
        save(out/'manifest.json',manifest)
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
            if np.any(host['policy_fault']|host['lower_fault']) or np.any(np.asarray((states[0] if scheduling else states).fault)):
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
