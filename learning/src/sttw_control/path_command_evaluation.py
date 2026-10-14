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
from .preference_best import atomic_json,config_hash


def _signature(tree):
    return (str(jax.tree.structure(tree)),tuple((tuple(x.shape),str(x.dtype)) for x in jax.tree.leaves(tree)))


class BatchedPanelEvaluator:
    def __init__(self,env,snapshot,config,check_budget=lambda:None,
                 status=lambda **kw:None,account=lambda ticks:None):
        self.env=env;self.snapshot=snapshot;self.config=config
        self.check_budget=check_budget;self.status=status;self.account=account
        self.compiled={}

    def _compiled_group(self,paths,speeds,alpha,params,horizon,*,capture_states=False,diagnostic_transform=None,diagnostic_identity=None,dynamic_modes=None):
        from .path_command_policy import make_path_actor,validate_actor_parameters
        env=self.env
        validate_actor_parameters(params,self.config)
        if (diagnostic_transform is None)!=(diagnostic_identity is None):
            raise ValueError('diagnostic transform requires explicit serializable identity')
        diagnostic_key=None if diagnostic_identity is None else config_hash(diagnostic_identity)
        dynamic_key=dynamic_modes is not None
        key=(dynamic_key,None if dynamic_modes is None else tuple(np.asarray(dynamic_modes).tolist()),config_hash(self.config),env.terminal_contract,_signature(paths),_signature(speeds),_signature(params),float(horizon),bool(capture_states),diagnostic_key)
        if key not in self.compiled:
            def reset(paths,speeds,alpha):
                return jax.vmap(lambda p,v,a:env.reset(self.snapshot,p,v,a))(paths,speeds,alpha) if dynamic_key else jax.vmap(lambda p,v:env.reset(self.snapshot,p,v,alpha))(paths,speeds)
            reset_fn=jax.jit(reset).lower(paths,speeds,alpha).compile()
            states=reset_fn(paths,speeds,alpha);jax.block_until_ready(states)
            example=jax.tree.map(lambda x:x[0],states)
            path=jax.tree.map(lambda x:x[0],paths)
            env.set_log_template(example,path,speeds[0])
            template=jax.eval_shape(lambda state:env.policy_step(state,j.zeros(2),path,speeds[0],horizon),example)[1]
            zero_logs=jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),template)
            if diagnostic_transform is not None:
                _,metadata=diagnostic_transform(j.zeros(2),example,path,speeds[0],env)
                lower_ticks=round(self.config['timing']['upper_dt_s']/self.config['timing']['lower_dt_s'])
                zero_logs=dict(zero_logs,**{'diagnostic_'+k:j.zeros((lower_ticks,)+v.shape,v.dtype) for k,v in metadata.items()})
            if dynamic_key:zero_logs=dict(zero_logs,alpha_request=j.zeros(4),alpha_applied=j.zeros(4))
            horizon_ticks=round(horizon/self.config['timing']['lower_dt_s'])
            actor=make_path_actor(self.config)
            def chunk(states,paths,speeds,params):
                def one(state,path,v,mode):
                    active=(state.tick<horizon_ticks)&~state.physical.failed&~state.fault&~state.domain_exit
                    def advance(state):
                        if dynamic_key:
                            from .path_command_dynamic import apply_dynamic_alpha
                            state,request=apply_dynamic_alpha(state,mode,self.config)
                        obs,_,fault=env.observation(state,path,v,horizon)
                        state=state.replace(fault=state.fault|fault)
                        z=actor.apply(params,obs)
                        if diagnostic_transform is None:
                            nxt,logs=env.policy_step(state,z,path,v,horizon)
                            if dynamic_key:logs=dict(logs,alpha_request=j.full(4,request),alpha_applied=j.full(4,state.alpha))
                            return nxt,logs
                        applied,metadata=diagnostic_transform(z,state,path,v,env)
                        # A transform cannot sanitize an original Actor fault.
                        state=state.replace(fault=state.fault|~j.all(j.isfinite(z)))
                        nxt,logs=env.policy_step(state,applied,path,v,horizon)
                        logs=dict(logs,**{'diagnostic_'+k:j.broadcast_to(value,(logs['active_tick'].shape[0],)+value.shape) for k,value in metadata.items()})
                        return nxt,logs
                    return jax.lax.cond(active,advance,lambda state:(state,zero_logs),state)
                def step(states,_):
                    nxt,logs=jax.vmap(one)(states,paths,speeds,dynamic_modes if dynamic_key else j.zeros(speeds.shape,j.int32))
                    return nxt,(logs,nxt) if capture_states else logs
                return jax.lax.scan(step,states,None,length=50)
            chunk_fn=jax.jit(chunk).lower(states,paths,speeds,params).compile()
            self.compiled[key]=(reset_fn,chunk_fn)
        return self.compiled[key]

    def evaluate(self,params,alpha,output,cached_names=None,*,case_names=None,diagnostic_transform=None,diagnostic_identity=None,dynamic_rows=None):
        from .path_command_selection import cases
        from .geometric_path import build_path
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
        horizon=self.config['training_future_phase_C']['episode_s']
        upper_dt=self.config['timing']['upper_dt_s'];intervals=round(horizon/upper_dt)
        if abs(upper_dt-.02)>1e-12 or intervals%50:
            raise ValueError('batched evaluator requires existing 50Hz whole-second horizon')
        traces={};groups={};pose=np.asarray(self.env.physics.helpers.pose(self.snapshot.data))
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
                np.savez_compressed(out/f'{name}.npz',**data)
                np.savez_compressed(out/f'{name}_path.npz',s=path.s,xy=path.xy,
                    heading=path.heading,curvature=path.curvature,goal=path.goal,turn_end=path.turn_end)
                traces[name]=data
            timing['file_seconds']+=time.perf_counter()-before
        try:
            for name,route,speed in declared:
                if name in cached:
                    before=time.perf_counter()
                    with np.load(out/f'{name}.npz') as archive:traces[name]={k:archive[k] for k in archive.files}
                    timing['file_seconds']+=time.perf_counter()-before
                    continue
                path=build_path(route,pose,self.config)
                groups.setdefault(_signature(path),[]).append(((name,route,speed),path))
            for entries in groups.values():
                self.check_budget();rows=[row for row,_ in entries]
                paths=jax.tree.map(lambda *xs:j.stack(xs),*[path for _,path in entries])
                speeds=j.asarray([row[2] for row in rows],j.float32);a=j.asarray(alpha,j.float32)
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


def _conditional_parameter_digest(params):
    import hashlib
    hasher=hashlib.sha256()
    hasher.update(str(jax.tree.structure(params)).encode())
    for value in jax.tree.leaves(params):
        array=np.asarray(value)
        hasher.update(str((array.shape,array.dtype.str)).encode());hasher.update(array.tobytes())
    return hasher.hexdigest()


def evaluate_conditional(evaluator,params,alphas,output):
    """Reuse existing vmap/scan50 evaluation with one parameter tree for all alpha."""
    from .path_command_selection import alpha_key,conditional_grid,summarize_conditional
    from .path_command_policy import architecture_identity,validate_actor_parameters
    from .path_command_reward import reward_version
    values=tuple(float(a) for a in alphas)
    if values!=conditional_grid(evaluator.config):raise ValueError('conditional alpha grid mismatch')
    validate_actor_parameters(params,evaluator.config)
    out=Path(output)
    if out.exists():raise FileExistsError('conditional validation output immutable')
    out.mkdir(parents=True)
    meta=dict(schema='path_conditional_panel_v1',mode='shared_conditional_v1',
        alphas={alpha_key(a):a for a in values},config_sha256=config_hash(evaluator.config),
        actor_parameters_sha256=_conditional_parameter_digest(params),
        architecture=architecture_identity(evaluator.config),reward_version=reward_version(evaluator.config),completed=False)
    atomic_json(out/'panel_identity.json',meta)
    result={alpha_key(a):evaluator.evaluate(params,a,out/alpha_key(a)) for a in values}
    summary=summarize_conditional(result,evaluator.config)
    atomic_json(out/'task_metrics.json',summary)
    meta['completed']=True;atomic_json(out/'panel_identity.json',meta)
    return result


def reuse_conditional_best(root,best,config):
    """Verify and copy one saved whole-grid panel; never runs simulator physics."""
    import json,shutil
    from .preference_best import digest
    from .path_command_selection import cases,alpha_key,conditional_grid,summarize_conditional,conditional_score_key
    from .path_command_policy import architecture_identity,validate_actor_parameters
    from .path_command_reward import reward_version
    import pickle
    root=Path(root);source_dir=Path(best['validation_dir']);source=json.loads((source_dir/'source.json').read_text())
    for kind in ('checkpoint','actor'):
        if digest(best[kind])!=best[kind+'_sha256'] or source.get(kind+'_sha256')!=best[kind+'_sha256']:
            raise ValueError('conditional validation source SHA mismatch')
    if source.get('source_update',source.get('update'))!=best['source_update']:
        raise ValueError('conditional validation source update identity mismatch')
    if any(source.get(k)!=v for k,v in best['logical_identity'].items()):
        raise ValueError('conditional validation source learner identity mismatch')
    meta=json.loads((source_dir/'panel_identity.json').read_text())
    expected={alpha_key(a):a for a in conditional_grid(config)}
    if not meta.get('completed') or meta.get('alphas')!=expected or meta.get('config_sha256')!=config_hash(config) or meta!=best['panel_identity']:
        raise ValueError('conditional panel identity mismatch')
    with Path(best['actor']).open('rb') as stream:params=pickle.load(stream)
    validate_actor_parameters(params,config)
    if meta.get('actor_parameters_sha256')!=_conditional_parameter_digest(params):
        raise ValueError('conditional full Actor parameters identity mismatch')
    from .path_command_policy import LEGACY_ARCHITECTURE
    architecture=architecture_identity(config)
    # Historical panels predate architecture metadata; only the legacy network may use them.
    if meta.get('architecture',architecture if architecture==LEGACY_ARCHITECTURE else None)!=architecture:
        raise ValueError('conditional architecture identity mismatch')
    if meta.get('reward_version',reward_version(config))!=reward_version(config):
        raise ValueError('conditional reward identity mismatch')
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
                if dest.exists() and digest(src)!=digest(dest):raise ValueError('immutable conditional final trace differs')
                if not dest.exists():shutil.copyfile(src,dest)
    atomic_json(final/'panel_identity.json',meta);atomic_json(final/'task_metrics.json',summary)
    atomic_json(final/'source.json',best)
    return panels
