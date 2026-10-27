"""Native finite-candidate comparator and cadence-matched H75, no training.

Loaded by a thin CLI before importing the sealed native sttw_control package.
The optimizer never uses the H75 parameters, observations or action proposals.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def limits(task_lower=2000,hold_lower=40,prediction_lower=200,lanes=8,candidates=26):
    if any(type(v) is not int or v<=0 for v in (task_lower,hold_lower,prediction_lower,lanes,candidates)):
        raise ValueError('positive integral clock and lane units required')
    if any(v%4 for v in (task_lower,hold_lower,prediction_lower)):
        raise ValueError('original four-tick upper boundaries required')
    starts=range(0,task_lower,hold_lower)
    return dict(maximum_planning_rounds=len(starts),
        maximum_prediction_lower_ticks=sum(min(prediction_lower,task_lower-t) for t in starts)*lanes*candidates,
        maximum_prediction_slots=len(starts)*prediction_lower*lanes*candidates,
        maximum_actual_lower_ticks=2*lanes*task_lower,maximum_actual_slots=2*lanes*task_lower)


def prepare(root,engineering):
    engineering=engineering.resolve();ep=json.loads((engineering/'plan.json').read_text())
    if digest(engineering/'plan.json')!=json.loads((engineering/'seal.json').read_text())['plan_sha256']:
        raise ValueError('engineering plan changed')
    if json.loads((engineering/'status.json').read_text())['phase']!='completed':
        raise ValueError('completed engineering receipt required')
    for file,sha in {**ep['files_sha256'],**ep['native_source_sha256']}.items():
        if digest(file)!=sha:raise ValueError('engineering source changed: '+file)
    source=Path(ep['source']);old=json.loads((source/'plan.json').read_text())
    cases=[{k:c[k] for k in ('route','builder','speed','alpha')} for c in old['cases']]
    assert len(cases)==8
    registry_path=Path('learning/configs/frozen_middle_registry.json').resolve()
    registry=json.loads(registry_path.read_text());h75=registry['records']['STTW_MIDDLE_FULLTAIL_PPO75_U77']
    native_evidence_plan=Path(old['source'])/'plan.json'
    history=json.loads(native_evidence_plan.read_text());native_manifest=Path(history['native_manifest'])
    if digest(native_manifest)!=history['source_manifest_sha256']:raise ValueError('native H75 evidence manifest changed')
    original_h75=json.loads(native_manifest.read_text())['declaration']['policies']['ppo75']
    for k in ('actor','checkpoint'):
        if digest(h75[k])!=h75[k+'_sha256'] or original_h75[k+'_sha256']!=h75[k+'_sha256']:
            raise ValueError('frozen/cached H75 identity differs')
    resolved=Path(registry['root'])/'STTW_MIDDLE_FULLTAIL_PPO75_U77_resolved_config.json'
    native_cfg=json.loads(resolved.read_text())
    for key in ('residual_interface','network','timing','path_reward','follower'):
        if native_cfg[key]!=ep['config'][key]:raise ValueError('native H75 contract differs: '+key)
    cached=[r for r in json.loads((source/'analysis/review.json').read_text())['records']
            if r['family'] in ('H75','zero','equilibrium')]
    assert len(cached)==24
    for r in cached:
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('cached control changed')
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    own=[Path(__file__).resolve(),Path('learning/cli/run_native_reference_rollout.py').resolve(),Path(ep['candidate_module'])]
    for p in own:(root/'source'/p.name).write_bytes(p.read_bytes())
    save(root/'cached_records.json',cached)
    inputs=[engineering/'plan.json',engineering/'seal.json',engineering/'engineering/receipt.json',
            engineering/'independent_numerical_review.json',source/'plan.json',source/'seal.json',
            Path(ep['initial_carry']),registry_path,resolved,root/'cached_records.json',native_evidence_plan,native_manifest]
    budget=limits()
    plan=dict(project='STTW_CONTROL',experiment_id='native_reference_rollout_v1',
        question='Does a fixed native dynamic reference-cost comparator cover H75 preference utility after separating decision cadence?',
        engineering=str(engineering),source=str(source),native_runtime_root=ep['native_runtime_root'],
        native_source_sha256=ep['native_source_sha256'],files_sha256={str(p):digest(p) for p in inputs+own},
        config=ep['config'],initial_source=ep['initial_source'],initial_carry=ep['initial_carry'],
        cases=cases,roles=['reference_grid_5hz','H75_5hz'],h75=h75,candidate_module=ep['candidate_module'],
        candidate_count=26,levels=ep['levels'],hold_upper_steps=10,preview_upper_steps=50,horizon_s=10.,
        task_lower_ticks=2000,cached_cases=24,new_cases=16,optimizer_updates=0,**budget,
        initialization='Same original native task0 complete carry for both arms, no H75 prefixes or optimizer state',
        selection='Original native J_alpha argmin, previous index0 wins exact ties; unknown engineering error, known native failures retain original penalty',
        predictor_tail='Hold one latent during1s; truncate at original task10s, no new terminal value; original full remaining-task failure penalty',
        future_force='No pulse query, zero assumed; complete simulator state privileged',
        cadence='Both new arms choose every0.2s, hold target10upper steps; native follower/history/filter/lower still50/200Hz. Cached H75 remains original50Hz.',
        data_role='Same four previously seen native DEV contexts, not independent TEST or training repeats',
        outputs='Every actual40lower chunk and26-way forecast summaries saved; prediction/actual/padding/compiler/wall costs separate',
        stop='Exactly16 nominal native10s/stopping prefixes; no force/time/grid/horizon/frequency tuning, training, automatic retry or extension',
        interpretation='A finite ordinary privileged baseline, not a new algorithm, global MPC optimum, safety certificate or real-time deployment')
    save(root/'plan.json',plan);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',new_prediction_lower_ticks=0,new_actual_lower_ticks=0,completed_cases=0,optimizer_updates=0))


def run(root,shape_only=False):
    plan=json.loads((root/'plan.json').read_text())
    assert digest(root/'plan.json')==json.loads((root/'seal.json').read_text())['plan_sha256']
    for file,sha in {**plan['files_sha256'],**plan['native_source_sha256']}.items():
        if digest(file)!=sha:raise ValueError('sealed input changed: '+file)
    if not shape_only and json.loads((root/'status.json').read_text())['phase']!='prepared':
        raise ValueError('no implicit restart')
    if any(n=='sttw_control' or n.startswith('sttw_control.') for n in sys.modules):
        raise ValueError('native namespace must be clean')
    sys.path.insert(0,plan['native_runtime_root'])
    import jax
    import jax.numpy as j
    import numpy as np
    import pickle
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.path_command_reward import configured_cost_keys
    from sttw_control.geometric_path import build_path
    from sttw_control.runtime import configure_compilation_cache
    output=root/('shape' if shape_only else 'evaluation');output.mkdir(exist_ok=False)
    started=time.monotonic();actual=prediction=allocated_actual=allocated_prediction=0
    phase_hint='initializing'

    def update(phase,**fields):
        nonlocal phase_hint
        phase_hint=phase
        status=dict(phase=phase,pid=os.getpid(),elapsed_s=time.monotonic()-started,
            new_prediction_lower_ticks=prediction,new_actual_lower_ticks=actual,
            allocated_prediction_slots=allocated_prediction,allocated_actual_slots=allocated_actual,
            optimizer_updates=0,completed_cases=0)
        status.update(fields)
        save(output/'status.json',status)
        if not shape_only:save(root/'status.json',status)

    try:
        update('initializing')
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        cfg=plan['config'];env,snapshot=load_environment(cfg,plan['initial_source'],'terminal_contract_v2')
        with Path(plan['initial_carry']).open('rb') as f:bank=pickle.load(f)
        states=jax.tree.map(lambda x:j.concatenate([j.asarray(x),j.asarray(x)],axis=0),bank)
        assert np.array_equal(np.asarray(states.tick),np.zeros(16))
        assert not np.any(np.asarray(states.fault|states.physical.failed|states.domain_exit))
        cases=plan['cases'];alphas=np.asarray([c['alpha'] for c in cases]*2)
        np.testing.assert_array_equal(states.alpha,alphas)
        pose=env.physics.helpers.pose(snapshot.data)
        paths_list=[build_path(c['builder'],pose,cfg) for c in cases]
        paths=jax.tree.map(lambda *x:j.stack(x),*paths_list)
        speeds=j.asarray([c['speed'] for c in cases]);all_speeds=j.concatenate([speeds,speeds])
        all_paths=jax.tree.map(lambda x:j.concatenate([x,x]),paths)
        first=jax.tree.map(lambda x:x[0],states);env.set_log_template(first,paths_list[0],speeds[0])
        governor=load_file('native_rollout_cost_functions',plan['candidate_module'])
        fixed=j.asarray(governor.native_cost_candidates([0.,0.],cfg)['latents'][1:])
        preview=governor.make_native_cost_preview(env,upper_steps=plan['preview_upper_steps'],horizon_s=10.)
        def forecast_fn(st,latents):
            return jax.vmap(lambda s,zs,p,v:jax.vmap(lambda z:preview(s,z,p,v)[1])(zs))(st,latents,paths,speeds)
        forecast=jax.jit(forecast_fn)
        with Path(plan['h75']['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
        actor=make_path_actor(cfg)
        def h75_fn(st):
            obs,_,fault=jax.vmap(lambda s,p,v:env.observation(s,p,v,10.))(st,paths,speeds)
            mu=jax.vmap(lambda x:actor.apply(params,x))(obs)
            return mu,fault
        h75=jax.jit(h75_fn)
        keys=list(configured_cost_keys(cfg))
        keep=('active_tick','time','actual_xy','reference_xy','actual_forward_speed','wheel_speed_proxy','v_user',
              'path_cross_track','path_heading_error','path_progress','goal_section_signed_distance',
              'phi','phi_dot','actual_delta','peak_roll','physical_failure','domain_exit','policy_fault','lower_fault',
              'governed','nominal','applied_force_impulse','scored_tick_reward','latent_z','offsets','filtered_offset','target_offset')
        def one(s,z,p,v):
            _,_,fault=env.observation(s,p,v,10.)
            end,log=env.policy_step(s.replace(fault=s.fault|fault),z,p,v,10.,j.zeros(5))
            return end,dict({k:log[k] for k in keep},effective_costs=j.stack([log['effective_components'][k] for k in keys],axis=-1))
        def advance_fn(st,targets):
            return jax.lax.scan(lambda s,_:jax.vmap(one)(s,targets,all_paths,all_speeds),st,None,length=10)
        advance=jax.jit(advance_fn)
        previous=j.zeros((8,2),j.float32)
        latents=j.concatenate([previous[:,None,:],j.broadcast_to(fixed,(8,25,2))],axis=1)
        mpc_states=jax.tree.map(lambda x:x[:8],states)
        shaped=jax.eval_shape(forecast,mpc_states,latents)
        actual_shape=jax.eval_shape(advance,states,j.zeros((16,2)))
        h75_shape=jax.eval_shape(h75,jax.tree.map(lambda x:x[8:],states))
        assert all(v.shape==(8,26) for v in shaped.values())
        assert actual_shape[1]['active_tick'].shape==(10,16,4) and h75_shape[0].shape==(8,2)
        save(output/'shape.json',dict(forecast={k:list(v.shape) for k,v in shaped.items()},
            actual_active_shape=[10,16,4],h75_target_shape=[8,2],new_prediction_lower_ticks=0,new_actual_lower_ticks=0))
        if shape_only:update('completed');return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU required')
        for c,path in zip(cases,paths_list):
            file=output/(c['route']+'_path.npz')
            if not file.exists():np.savez_compressed(file,s=path.s,xy=path.xy,goal=path.goal)
        save(output/'manifest.json',dict(config=cfg,cases=cases,roles=plan['roles'],source=plan['initial_source'],
             reward_components=keys,plan_sha256=digest(root/'plan.json'),cached_cases=24,new_cases=16))
        update('compiling');clock=time.monotonic()
        forecast_exe=forecast.lower(mpc_states,latents).compile()
        advance_exe=advance.lower(states,j.zeros((16,2))).compile()
        h75_exe=h75.lower(jax.tree.map(lambda x:x[8:],states)).compile()
        compile_s=time.monotonic()-clock
        save(output/'compilation.json',dict(compile_s=compile_s,warmup_integrations=0))
        chunks=output/'chunks';chunks.mkdir();forecasts=output/'forecasts';forecasts.mkdir()
        saved=[];timings=[]
        for block in range(plan['maximum_planning_rounds']):
            mpc_states=jax.tree.map(lambda x:x[:8],states)
            latents=j.concatenate([previous[:,None,:],j.broadcast_to(fixed,(8,25,2))],axis=1)
            update('predicting',completed_planning_rounds=block)
            clock=time.monotonic();summary=forecast_exe(mpc_states,latents);jax.block_until_ready(summary)
            forecast_s=time.monotonic()-clock;host=jax.device_get(summary)
            np.savez_compressed(forecasts/f'round_{block:02d}.npz',**host,candidate_latents=np.asarray(latents))
            prediction+=int(host['observed_lower_ticks'].sum());allocated_prediction+=8*26*200
            update('forecast_returned',completed_planning_rounds=block)
            if not np.all(host['finite']) or not np.isfinite(host['original_user_cost']).all():
                raise RuntimeError('unknown candidate forecast; retained evidence, no implicit deletion/retry')
            selected=np.argmin(host['original_user_cost'],axis=1)
            proposed=j.asarray(np.asarray(latents)[np.arange(8),selected])
            clock=time.monotonic();mu,fault=h75_exe(jax.tree.map(lambda x:x[8:],states));jax.block_until_ready(mu)
            actor_s=time.monotonic()-clock
            if np.any(fault) or not np.isfinite(np.asarray(mu)).all():raise RuntimeError('unknown H75 proposal')
            targets=j.concatenate([proposed,mu]);update('advancing',completed_planning_rounds=block)
            clock=time.monotonic();states,log=advance_exe(states,targets);jax.block_until_ready(states)
            advance_s=time.monotonic()-clock;raw=jax.device_get(log);saved.append(raw)
            np.savez_compressed(chunks/f'round_{block:02d}.npz',**raw)
            actual+=int(raw['active_tick'].sum());allocated_actual+=16*40;previous=proposed
            timings.append(dict(round=block,forecast_s=forecast_s,actor_s=actor_s,advance_s=advance_s,selected=selected.tolist()))
            save(output/'timings.json',timings)
            update('running',completed_planning_rounds=block+1,compile_s=compile_s)
            if np.any(np.asarray(states.fault)) or np.any(raw['policy_fault']|raw['lower_fault']):
                raise RuntimeError('actual native engineering fault; no negative-label substitution')
            if np.all(np.asarray(states.physical.failed|states.domain_exit)):break
        combined={k:np.concatenate([v[k] for v in saved],axis=0) for k in saved[0]}
        rows=[]
        for lane in range(16):
            case=cases[lane%8];role=plan['roles'][lane//8]
            raw={k:v[:,lane].reshape((-1,)+v.shape[3:]) for k,v in combined.items()}
            active=raw['active_tick'].astype(bool);raw={k:v[active] for k,v in raw.items()}
            file=output/f"{role}_{case['route']}_v{case['speed']:g}_a{case['alpha']:g}.npz"
            np.savez_compressed(file,**raw)
            rows.append(dict(role=role,route=case['route'],speed=case['speed'],alpha=case['alpha'],raw_file=str(file),raw_sha256=digest(file)))
        assert prediction<=plan['maximum_prediction_lower_ticks'] and actual<=plan['maximum_actual_lower_ticks']
        save(output/'results.json',dict(cases=rows))
        update('completed',completed_cases=16,compile_s=compile_s)
    except BaseException as exc:
        update('error',error=repr(exc),inflight_prediction_cost_unknown=phase_hint=='predicting',
               inflight_actual_cost_unknown=phase_hint=='advancing')
        raise
