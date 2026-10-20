"""Batched fixed-path DEV, preserving paths and every active lower-step log.

Cases with different path array shapes compile separately rather than padding or
modifying geometry. One host synchronization per 50-policy-step chunk replaces
per-policy-step synchronization. This module never launches training by itself.
"""
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as j
from .preference_best import atomic_json
from .path_command_policy import config_key


def _signature(tree):
    return (str(jax.tree.structure(tree)),tuple((tuple(x.shape),str(x.dtype)) for x in jax.tree.leaves(tree)))


class BatchedPanelEvaluator:
    def __init__(self,env,snapshot,config,check_budget=lambda:None,
                 status=lambda **kw:None,account=lambda ticks:None):
        self.env=env;self.snapshot=snapshot;self.config=config
        self.check_budget=check_budget;self.status=status;self.account=account
        self.compiled={}
        self.paths={}

    def path_for_route(self,route):
        """One immutable geometry per route for this fixed snapshot/config."""
        from .geometric_path import build_path
        if route not in self.paths:
            pose=np.asarray(self.env.physics.helpers.pose(self.snapshot.data))
            self.paths[route]=build_path(route,pose,self.config)
        return self.paths[route]

    def _compiled_group(self,paths,speeds,alpha,params,horizon,*,capture_states=False,diagnostic_transform=None,diagnostic_identity=None,dynamic_modes=None):
        from .path_command_policy import make_path_actor,validate_actor_parameters
        env=self.env
        validate_actor_parameters(params,self.config)
        if (diagnostic_transform is None)!=(diagnostic_identity is None):
            raise ValueError('diagnostic transform requires explicit serializable identity')
        diagnostic_key=None if diagnostic_identity is None else config_key(diagnostic_identity)
        dynamic_key=dynamic_modes is not None
        key=(_signature(alpha),dynamic_key,None if dynamic_modes is None else tuple(np.asarray(dynamic_modes).tolist()),config_key(self.config),env.terminal_contract,_signature(paths),_signature(speeds),_signature(params),float(horizon),bool(capture_states),diagnostic_key)
        if key not in self.compiled:
            def request(v,tick):
                if v.ndim==0:return v
                from .path_command_scenarios import speed_request
                return speed_request(v[:3],v[3:],tick*self.config['timing']['lower_dt_s'])
            def reset(paths,speeds,alpha):
                return jax.vmap(lambda p,v,a:env.reset(self.snapshot,p,request(v,0),a))(paths,speeds,alpha) if alpha.ndim else jax.vmap(lambda p,v:env.reset(self.snapshot,p,request(v,0),alpha))(paths,speeds)
            reset_fn=jax.jit(reset).lower(paths,speeds,alpha).compile()
            states=reset_fn(paths,speeds,alpha);jax.block_until_ready(states)
            example=jax.tree.map(lambda x:x[0],states)
            path=jax.tree.map(lambda x:x[0],paths)
            env.set_log_template(example,path,request(speeds[0],example.tick))
            template=jax.eval_shape(lambda state:env.policy_step(state,j.zeros(2),path,request(speeds[0],state.tick),horizon,evaluation=True),example)[1]
            zero_logs=jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),template)
            if diagnostic_transform is not None:
                _,metadata=diagnostic_transform(j.zeros(2),example,path,request(speeds[0],example.tick),env)
                lower_ticks=round(self.config['timing']['upper_dt_s']/self.config['timing']['lower_dt_s'])
                zero_logs=dict(zero_logs,**{'diagnostic_'+k:j.zeros((lower_ticks,)+v.shape,v.dtype) for k,v in metadata.items()})
            if dynamic_key:zero_logs=dict(zero_logs,alpha_request=j.zeros(4),alpha_applied=j.zeros(4))
            horizon_ticks=round(horizon/self.config['timing']['lower_dt_s'])
            actor=make_path_actor(self.config)
            def chunk(states,paths,speeds,params):
                def one(state,path,v,mode):
                    active=(state.tick<horizon_ticks)&~state.physical.failed&~state.fault&~state.domain_exit
                    def advance(state):
                        current_request=request(v,state.tick)
                        if dynamic_key:
                            from .path_command_dynamic import apply_dynamic_alpha
                            state,alpha_request=apply_dynamic_alpha(state,mode,self.config)
                        obs,_,fault=env.observation(state,path,current_request,horizon)
                        state=state.replace(fault=state.fault|fault)
                        z=actor.apply(params,obs)
                        if diagnostic_transform is None:
                            nxt,logs=env.policy_step(state,z,path,current_request,horizon,evaluation=True)
                            if dynamic_key:logs=dict(logs,alpha_request=j.full(4,alpha_request),alpha_applied=j.full(4,state.alpha))
                            return nxt,logs
                        applied,metadata=diagnostic_transform(z,state,path,current_request,env)
                        # A transform cannot sanitize an original Actor fault.
                        state=state.replace(fault=state.fault|~j.all(j.isfinite(z)))
                        nxt,logs=env.policy_step(state,applied,path,current_request,horizon,evaluation=True)
                        logs=dict(logs,**{'diagnostic_'+k:j.broadcast_to(value,(logs['active_tick'].shape[0],)+value.shape) for k,value in metadata.items()})
                        return nxt,logs
                    return jax.lax.cond(active,advance,lambda state:(state,zero_logs),state)
                def step(states,_):
                    nxt,logs=jax.vmap(one)(states,paths,speeds,dynamic_modes if dynamic_key else j.zeros((speeds.shape[0],),j.int32))
                    return nxt,(logs,nxt) if capture_states else logs
                return jax.lax.scan(step,states,None,length=50)
            chunk_fn=jax.jit(chunk).lower(states,paths,speeds,params).compile()
            self.compiled[key]=(reset_fn,chunk_fn)
        return self.compiled[key]

    def evaluate(self,params,alpha,output,cached_names=None,*,case_names=None,diagnostic_transform=None,diagnostic_identity=None,dynamic_rows=None,alpha_grid=None):
        from .path_command_selection import cases
        from .direct_command_training import flatten_logs
        started=time.perf_counter();out=Path(output)
        if out.exists() and cached_names is None:raise FileExistsError('validation output immutable')
        out.mkdir(parents=True,exist_ok=cached_names is not None)
        declared=cases(self.config)
        if dynamic_rows is not None:
            if cached_names is not None or diagnostic_transform is not None:raise ValueError('dynamic panel cannot reuse fixed traces or transforms')
            declared=[row[:3] for row in dynamic_rows]
            modes={row[0]:row[3] for row in dynamic_rows}
        names={name for name,_,_ in declared}
        if case_names is not None:
            requested=list(case_names)
            if not requested or len(set(requested))!=len(requested) or not set(requested)<=names:raise ValueError('invalid diagnostic case subset')
            declared=[row for row in declared if row[0] in requested];names=set(requested)
        if diagnostic_transform is not None:
            if cached_names is not None:raise ValueError('diagnostics cannot use unchanged panel cache')
            if diagnostic_identity is None:raise ValueError('diagnostic identity required')
        cached=set(cached_names or ())
        if not cached<=names:raise ValueError('unknown cached validation case')
        lane_alphas={}
        if alpha_grid is not None:
            from .path_command_selection import alpha_key
            if cached or dynamic_rows is not None or diagnostic_transform is not None:raise ValueError('parallel alpha grid cannot mix cache or diagnostic modes')
            original=declared
            declared=[(alpha_key(a)+'/'+name,route,speed) for a in alpha_grid for name,route,speed in original]
            lane_alphas={alpha_key(a)+'/'+name:float(a) for a in alpha_grid for name,_,_ in original}
        conflict=self.config.get('training_sampler',{}).get('version')=='path_conflict_exposure_v1'
        definitions={}
        if conflict:
            from .path_command_scenarios import explicit_case_declarations
            definitions={row['name']:row for row in explicit_case_declarations(self.config)}
        def definition(name):
            key=name.rsplit('/',1)[-1]
            if key not in definitions and dynamic_rows is not None:key=key.rsplit('_',1)[0]
            return definitions[key]
        horizon=self.config['training_future_phase_C']['episode_s']
        upper_dt=self.config['timing']['upper_dt_s'];intervals=round(horizon/upper_dt)
        if abs(upper_dt-.02)>1e-12 or intervals%50:
            raise ValueError('batched evaluator requires existing 50Hz whole-second horizon')
        traces={};groups={}
        timing=dict(schema='path_batched_evaluation_v1',compile_seconds=0.,physics_seconds=0.,
            transfer_seconds=0.,file_seconds=0.,reset_seconds=0.,cached_cases=sorted(cached),
            chunk_policy_steps=50,lower_dt_s=self.config['timing']['lower_dt_s'],groups=[],completed=False,diagnostic_identity=diagnostic_identity)
        def write_timing():
            timing['total_seconds']=time.perf_counter()-started
            atomic_json(out/'evaluation_timings.json',timing)
        def save_group(rows,paths,chunks):
            before=time.perf_counter()
            if not chunks:return
            flat=flatten_logs(jax.tree.map(lambda *xs:np.concatenate(xs,axis=0),*chunks))
            for i,(name,_,_) in enumerate(rows):
                # scan-time, case, lower-tick, remaining value axes.
                data={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()}
                active=np.asarray(data['active_tick'],bool);data={k:v[active] for k,v in data.items()}
                path=jax.tree.map(lambda x:np.asarray(x[i]),paths)
                data['goal_progress']=np.full(len(data['time']),float(path.goal))
                (out/f'{name}.npz').parent.mkdir(parents=True,exist_ok=True)
                if conflict:
                    for field in ('turn_starts','turn_ends','turn_directions','turn_mask'):data[field]=np.asarray(getattr(path,field))
                    declaration=definition(name)
                    data['speed_script_values']=np.asarray(declaration['speed_values'])
                    data['speed_script_switch_times']=np.asarray(declaration['speed_switch_times'])
                np.savez(out/f'{name}.npz',**data)
                np.savez(out/f'{name}_path.npz',s=path.s,xy=path.xy,
                    heading=path.heading,curvature=path.curvature,goal=path.goal,turn_end=path.turn_end,
                    **({field:np.asarray(getattr(path,field)) for field in ('turn_starts','turn_ends','turn_directions','turn_mask')} if conflict else {}))
                traces[name]=data
            timing['file_seconds']+=time.perf_counter()-before
        try:
            for name,route,speed in declared:
                if name in cached:
                    before=time.perf_counter()
                    with np.load(out/f'{name}.npz') as archive:traces[name]={k:archive[k] for k in archive.files}
                    timing['file_seconds']+=time.perf_counter()-before
                    continue
                path=self.path_for_route(route)
                groups.setdefault(_signature(path),[]).append(((name,route,speed),path))
            for entries in groups.values():
                self.check_budget();rows=[row for row,_ in entries]
                paths=jax.tree.map(lambda *xs:j.stack(xs),*[path for _,path in entries])
                speeds=j.asarray([definition(row[0])['speed_values']+definition(row[0])['speed_switch_times'] for row in rows],j.float32) if conflict else j.asarray([row[2] for row in rows],j.float32)
                a=j.asarray([lane_alphas[row[0]] for row in rows],j.float32) if alpha_grid is not None else j.asarray(alpha,j.float32)
                dynamic_modes=None if dynamic_rows is None else j.asarray([modes[row[0]] for row in rows],j.int32)
                if dynamic_modes is not None:a=j.where(dynamic_modes==1,1.,0.)
                before=time.perf_counter()
                reset,chunk=self._compiled_group(paths,speeds,a,params,horizon,diagnostic_transform=diagnostic_transform,diagnostic_identity=diagnostic_identity,dynamic_modes=dynamic_modes)
                timing['compile_seconds']+=time.perf_counter()-before
                before=time.perf_counter();states=reset(paths,speeds,a);jax.block_until_ready(states)
                timing['reset_seconds']+=time.perf_counter()-before
                names_group=[row[0] for row in rows];chunks=[]
                group=dict(cases=names_group,path_points=int(paths.s.shape[-1]),active_lower_ticks=0,completed_intervals=0)
                timing['groups'].append(group)
                if bool(j.any(states.fault|states.domain_exit)):
                    # Reset-domain invalidity is engineering fault, not a policy sample.
                    raise RuntimeError('fixed validation reset invalid; no best selection')
                for offset in range(0,intervals,50):
                    self.check_budget()
                    self.status(state='running',stage='validation',case=','.join(names_group),completed_intervals=offset)
                    before=time.perf_counter();states,logs=chunk(states,paths,speeds,params);jax.block_until_ready(states)
                    timing['physics_seconds']+=time.perf_counter()-before
                    before=time.perf_counter();host_logs,failed,domain,fault,ticks=jax.device_get((logs,states.physical.failed,states.domain_exit,states.fault,states.tick))
                    timing['transfer_seconds']+=time.perf_counter()-before
                    chunks.append(host_logs);active_ticks=int(np.sum(host_logs['active_tick']))
                    self.account(active_ticks);group['active_lower_ticks']+=active_ticks;group['completed_intervals']=offset+50
                    if np.any(fault):
                        save_group(rows,paths,chunks)
                        raise RuntimeError('fixed validation engineering fault; no best selection')
                    if np.all(failed|domain|(ticks>=round(horizon/self.config['timing']['lower_dt_s']))):break
                save_group(rows,paths,chunks)
            timing['completed']=True
            return traces
        finally:write_timing()




def evaluate_conditional(evaluator,params,alphas,output,*,case_names=None,diagnostic_only=False):
    """Reuse existing vmap/scan50 evaluation with one parameter tree for all alpha."""
    from .path_command_selection import alpha_key,conditional_grid,summarize_conditional
    from .path_command_policy import architecture_identity,validate_actor_parameters
    from .path_command_reward import reward_version
    values=tuple(float(a) for a in alphas)
    if not diagnostic_only and values!=conditional_grid(evaluator.config):raise ValueError('conditional alpha grid mismatch')
    validate_actor_parameters(params,evaluator.config)
    out=Path(output)
    if out.exists():raise FileExistsError('conditional validation output immutable')
    out.mkdir(parents=True)
    meta=dict(schema='path_conditional_panel_v1',mode='shared_conditional_v1',
        alphas={alpha_key(a):a for a in values},config_key=config_key(evaluator.config),
        actor_parameters_sha256=None,
        architecture=architecture_identity(evaluator.config),reward_version=reward_version(evaluator.config),completed=False)
    if evaluator.config.get('preference_selection_schema') in ('path_preference_research_v1','path_conflict_preference_v1'):
        from .path_command_selection import resolve_preference_acceptance
        meta['acceptance_key']=config_key(resolve_preference_acceptance(evaluator.config))
    if evaluator.config.get('training_sampler',{}).get('version')=='path_conflict_exposure_v1':
        from .path_command_scenarios import explicit_case_declarations
        meta['case_declarations_key']=config_key(explicit_case_declarations(evaluator.config))
    atomic_json(out/'panel_identity.json',meta)
    if evaluator.config.get('evaluation_execution')=='alpha_case_vmap_v1':
        flat=evaluator.evaluate(params,0.,out,cached_names=set(),case_names=case_names,alpha_grid=values)
        result={alpha_key(a):{name.split('/',1)[1]:trace for name,trace in flat.items() if name.startswith(alpha_key(a)+'/')} for a in values}
        meta['execution']='alpha_case_vmap_v1'
    else:
        if case_names is not None:raise ValueError('case subsets require alpha_case_vmap_v1')
        result={alpha_key(a):evaluator.evaluate(params,a,out/alpha_key(a)) for a in values}
    if not diagnostic_only:
        summary=summarize_conditional(result,evaluator.config)
        atomic_json(out/'task_metrics.json',summary)
    meta['diagnostic_only']=bool(diagnostic_only)
    meta['completed']=True;atomic_json(out/'panel_identity.json',meta)
    return result


def reuse_conditional_best(root,best,config):
    """Verify and copy one saved whole-grid panel; never runs simulator physics."""
    import json,shutil
    from .path_command_selection import cases,alpha_key,conditional_grid,summarize_conditional,conditional_score_key
    from .path_command_policy import architecture_identity,validate_actor_parameters
    from .path_command_reward import reward_version
    import pickle
    root=Path(root);source_dir=Path(best['validation_dir']);source=json.loads((source_dir/'source.json').read_text())
    if source.get('source_update',source.get('update'))!=best['source_update']:
        raise ValueError('conditional validation source update identity mismatch')
    if any(source.get(k)!=v for k,v in best['logical_identity'].items()):
        raise ValueError('conditional validation source learner identity mismatch')
    meta=json.loads((source_dir/'panel_identity.json').read_text())
    if config.get('training_sampler',{}).get('version')=='path_conflict_exposure_v1':
        from .path_command_scenarios import explicit_case_declarations
        sha=config_key(explicit_case_declarations(config))
        if meta.get('case_declarations_key')!=sha or best.get('case_declarations_key')!=sha:raise ValueError('conditional case/script/turn identity mismatch')
    expected={alpha_key(a):a for a in conditional_grid(config)}
    if not meta.get('completed') or meta.get('alphas')!=expected or meta.get('config_key')!=config_key(config) or meta!=best['panel_identity']:
        raise ValueError('conditional panel identity mismatch')
    with Path(best['actor']).open('rb') as stream:params=pickle.load(stream)
    validate_actor_parameters(params,config)
    from .path_command_policy import LEGACY_ARCHITECTURE
    architecture=architecture_identity(config)
    # Historical panels predate architecture metadata; only the legacy network may use them.
    if meta.get('architecture',architecture if architecture==LEGACY_ARCHITECTURE else None)!=architecture:
        raise ValueError('conditional architecture identity mismatch')
    if meta.get('reward_version',reward_version(config))!=reward_version(config):
        raise ValueError('conditional reward identity mismatch')
    if config.get('preference_selection_schema') in ('path_preference_research_v1','path_conflict_preference_v1'):
        from .path_command_selection import resolve_preference_acceptance
        acceptance_sha=config_key(resolve_preference_acceptance(config))
        if meta.get('acceptance_key')!=acceptance_sha:
            raise ValueError('conditional preference acceptance identity mismatch')
        if best.get('selection_schema') in ('path_preference_research_v1','path_conflict_preference_v1') and best.get('acceptance_key')!=acceptance_sha:
            raise ValueError('selected preference acceptance identity mismatch')
    panels={key:{name:dict(np.load(source_dir/key/f'{name}.npz')) for name,_,_ in cases(config)} for key in expected}
    if best.get('selection_schema')==config.get('preference_selection_schema') and best.get('selection_schema') is not None:
        from .path_command_selection import summarize_preference,preference_score_key
        summary=summarize_preference(panels,config);score=preference_score_key(summary,best['source_update'])
    else:
        summary=summarize_conditional(panels,config);score=conditional_score_key(summary)
    if score!=tuple(best['score_tuple']):raise ValueError('conditional panel metrics differ from selected model')
    final=root/'final_best'
    if final.exists():
        existing=json.loads((final/'source.json').read_text())
        if existing!=best:raise ValueError('immutable conditional final source differs')
    else:final.mkdir()
    for key in expected:
        target=final/key;target.mkdir(exist_ok=True)
        for name,_,_ in cases(config):
            for suffix in ('.npz','_path.npz'):
                src=source_dir/key/(name+suffix);dest=target/(name+suffix)
                if not dest.exists():shutil.copyfile(src,dest)
    atomic_json(final/'panel_identity.json',meta);atomic_json(final/'task_metrics.json',summary)
    atomic_json(final/'source.json',best)
    return panels


def run_conflict_preflight(config_path,output):
    """One declared B0/probe panel; scientific failures are recorded, never gated."""
    import json
    from .path_command_training import load_plan,load_environment
    from .path_command_cache import reuse_conflict_baseline
    from .path_command_policy import make_path_actor
    from .path_command_scenarios import explicit_case_declarations
    from .path_command_selection import _conflict_turn_metrics
    from .runtime import configure_compilation_cache
    from .geometric_path import build_path
    plan,cfg,source=load_plan(config_path);root=Path(output)
    if root.exists():raise FileExistsError('preflight directory immutable')
    root.mkdir(parents=True);configure_compilation_cache()
    status=dict(state='initializing',stage='preflight',active_lower_ticks=0,max_lower_ticks=64000)
    atomic_json(root/'status.json',status)
    env,snapshot=load_environment(cfg,source,plan['terminal_contract'])
    definitions=explicit_case_declarations(cfg);pose=np.asarray(env.physics.helpers.pose(snapshot.data))
    paths={d['route_id']:build_path(d['route_id'],pose,cfg) for d in definitions}
    atomic_json(root/'manifest.json',dict(config=cfg,source=source,plan=plan,data_role='B0 and probe diagnostics; never PPO/teacher',scientific_failure_policy='record_and_continue_200'))
    original_plan_path=Path(config_path).resolve().parent/'path_unified_scratch_run.json'
    old_plan,_,_=load_plan(original_plan_path)
    old_parent=old_plan['baseline_parent']
    cached=reuse_conflict_baseline(old_parent,root,cfg,source,paths)
    def update_status(**kw):status.update(kw);atomic_json(root/'status.json',status)
    def account(ticks):
        status['active_lower_ticks']+=int(ticks)
        if status['active_lower_ticks']>64000:raise RuntimeError('preflight lower tick budget exceeded')
        atomic_json(root/'status.json',status)
    evaluator=BatchedPanelEvaluator(env,snapshot,cfg,status=update_status,account=account)
    params=make_path_actor(cfg).init(jax.random.PRNGKey(0),j.zeros(351))
    try:
        baseline=evaluator.evaluate(params,0.,root/'zero_upper',cached_names=cached)
        names=[d['name'] for d in definitions if d['evaluation_role']=='constant_pressure']
        def probe(z,state,path,v,environment):
            active=state.projection.progress<path.turn_end+2.
            physical=j.where(active,-.65,0.)
            scale=cfg['residual_interface']['speed_negative_scale_m_s']
            latent=j.arctanh(j.clip(physical/scale,-.999999,.999999))
            return j.stack([latent,j.float32(0.)]),dict(probe_target_speed_offset=physical)
        identity=dict(mode='fixed_slowdown_probe',delta_v_m_s=-.65,delta_delta_rad=0.,release_m_after_last_turn=2.,never_teacher=True)
        probes=evaluator.evaluate(params,0.,root/'slowdown_probe',case_names=names,diagnostic_transform=probe,diagnostic_identity=identity)
        metrics={}
        def peak(d):return float(np.max(d['peak_roll']))
        def failed(d):return bool(np.any(d['physical_failure']) or np.any(d['domain_exit']))
        for name in names:
            b,p=baseline[name],probes[name];bm=_conflict_turn_metrics(b,cfg);pm=_conflict_turn_metrics(p,cfg)
            over=np.asarray(b['peak_roll'])>.32
            edges=np.diff(np.r_[False,over,False].astype(int));durations=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
            over_s=float(durations.max(initial=0)*cfg['timing']['lower_dt_s'])
            b_conflict=failed(b) or over_s>=.25 or (bm['path_rmse_m'] is not None and bm['path_rmse_m']>.20)
            goal=bool(np.any((p['goal_section_signed_distance']>=0)&(p['path_progress']>=p['goal_progress'])))
            slowdown=None if pm['mean_actual_speed_m_s'] is None or bm['mean_actual_speed_m_s'] is None else bm['mean_actual_speed_m_s']-pm['mean_actual_speed_m_s']
            roll_reduction=peak(b)-peak(p)
            path_reduction=None if bm['path_rmse_m'] is None or pm['path_rmse_m'] is None else bm['path_rmse_m']-pm['path_rmse_m']
            qualified=bool(b_conflict and not failed(p) and goal and pm['complete'] and slowdown is not None and slowdown>=.30 and (roll_reduction>=.03 or path_reduction is not None and path_reduction>=.05))
            metrics[name]=dict(baseline=bm,probe=pm,baseline_failure=failed(b),probe_failure=failed(p),probe_goal=goal,baseline_over32_max_contiguous_s=over_s,baseline_conflict=b_conflict,actual_slowdown_m_s=slowdown,roll_peak_reduction_rad=roll_reduction,path_rmse_reduction_m=path_reduction,empirical_conflict_qualified=qualified)
        receipt=dict(state='completed',scientific_threshold_policy='diagnostic_continue_declared_200',all_conflict_conditions=all(m['empirical_conflict_qualified'] for m in metrics.values()),metrics=metrics,active_lower_ticks=status['active_lower_ticks'],max_lower_ticks=64000,reused_baseline_cases=sorted(cached),probe_identity=identity,eligible_as_parent=False)
        atomic_json(root/'receipt.json',receipt)
        from .path_command_reporting import report_conflict_preflight
        report_conflict_preflight(root,cfg,receipt)
        update_status(state='completed',stage='reported')
        return receipt
    except Exception as error:
        update_status(state='error',error=str(error));raise
