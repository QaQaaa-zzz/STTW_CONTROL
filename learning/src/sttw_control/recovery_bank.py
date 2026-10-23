"""Collect immutable TRAIN-only parent prefixes and nominal retention anchors."""
from pathlib import Path
import json,os,pickle,time,subprocess
import numpy as np
from .preference_best import atomic_json,digest


def select_complete_states(mask, old, current):
    """Select whole closed-loop states, with the same lane mask for every leaf."""
    import jax
    import jax.numpy as j
    return jax.tree.map(lambda a,b:j.where(mask.reshape(mask.shape+(1,)*(b.ndim-mask.ndim)),b,a),old,current)


def collect_preference(declaration_path,output):
    """New authentic TRAIN prefixes from the actual conditional parent.

    Exact alpha values are repeated across independent contexts. Development
    screen snapshots are never read. Original clocks and full states survive.
    """
    import jax
    import jax.numpy as j
    import torch
    from .path_command_training import load_environment
    from .path_command_scenarios import sample_curriculum_route,validate_recovery_coverage
    from .path_command_policy import make_path_actor,validate_actor_parameters
    from .runtime import configure_compilation_cache
    declaration=json.loads(Path(declaration_path).read_text());out=Path(output)
    coverage=declaration.get('route_coverage')
    validate_recovery_coverage(coverage)
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();ticks=0
    n=sum(declaration['alpha_counts']);seconds=declaration['nominal_collection_s'];h=declaration['config']['training_future_phase_C']['episode_s']
    steps=round(seconds/.02);maximum=n*round(seconds/.005)
    if steps%50 or seconds>h or maximum!=declaration['maximum_lower_ticks']:raise ValueError('bank timing/accounting mismatch')
    state=dict(state='running',phase='running',pid=os.getpid(),training_steps=0,declared_contexts=n,maximum_lower_ticks=maximum)
    def update(**kw):
        state.update(kw);state.update(checked_unix=time.time(),elapsed_s=time.monotonic()-started,active_lower_ticks=ticks)
        atomic_json(out/'status.json',state)
    update(stage='loading')
    try:
        record=declaration['parent'];cfg=declaration['config'];source=declaration['source']
        for key in ('actor','checkpoint'):
            if digest(record[key])!=record[key+'_sha256']:raise ValueError('conditional parent identity changed')
        cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,snapshot=load_environment(cfg,source,'terminal_contract_v2');pose=env.physics.helpers.pose(snapshot.data)
        ids=np.concatenate([np.linspace(0,511,count,dtype=np.int32) for count in declaration['alpha_counts']])
        alpha=np.repeat(declaration['alphas'],declaration['alpha_counts']).astype(np.float32)
        routes=jax.jit(jax.vmap(lambda eid:sample_curriculum_route(declaration['route_seed'],eid,declaration['route_episode'],pose,cfg,declaration['stage_index'],coverage)))(j.asarray(ids))
        actual=jax.device_get(routes)
        atomic_json(out/'route_coverage.json',dict(declared=coverage,alpha=alpha.tolist(),route_ids=ids.tolist(),
            speed_m_s=np.asarray(actual.speed).tolist(),radius_m=np.asarray(actual.radius).tolist(),
            signed_angle_deg=np.rad2deg(np.asarray(actual.angle)).tolist(),family=np.asarray(actual.family).tolist(),
            interpretation='sampled contexts before prefix validity filtering; valid mask and counts in final bank receipt'))
        states=jax.jit(jax.vmap(lambda path,v,a:env.reset(snapshot,path,v,a)))(routes.path,routes.speed,j.asarray(alpha))
        env.set_log_template(jax.tree.map(lambda x:x[0],states),jax.tree.map(lambda x:x[0],routes.path),routes.speed[0])
        actor=make_path_actor(cfg)
        with Path(record['actor']).open('rb') as stream:params=jax.tree.map(j.asarray,pickle.load(stream))
        validate_actor_parameters(params,cfg)
        saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False);log_std=saved['policy']['log_std'].detach().numpy()
        keys=jax.vmap(lambda eid:jax.random.fold_in(jax.random.PRNGKey(declaration['onset_seed']),eid))(j.arange(n))
        low,high=declaration['onset_intervals_inclusive']
        if low<0 or high>=steps:raise ValueError('prefix onset outside collection')
        onsets=jax.vmap(lambda key:jax.random.randint(key,(),low,high+1))(keys)
        def one(st,path,v):
            obs,_,fault=env.observation(st,path,v,h);mu=actor.apply(params,obs)
            end,logs=env.policy_step(st,mu,path,v,h)
            bad=end.physical.failed|end.domain_exit|end.fault
            return end,obs,mu,bad,fault|j.any(logs['policy_fault']|logs['lower_fault']),j.sum(logs['active_tick'])
        def chunk(carry,start):
            def step(c,offset):
                st,snap,bad,valid=c;i=start+offset;choose=i==onsets
                snap=select_complete_states(choose,snap,st);valid=j.where(choose,~bad,valid)
                end,obs,mu,newbad,fault,active=jax.vmap(one)(st,routes.path,routes.speed)
                return (end,snap,bad|newbad,valid),(obs,mu,~bad&~newbad,fault,active)
            return jax.lax.scan(step,carry,j.arange(50,dtype=j.int32))
        execute=jax.jit(chunk);carry=(states,states,j.zeros(n,j.bool_),j.zeros(n,j.bool_))
        observations=[];means=[];anchor_valid=[]
        atomic_json(out/'manifest.json',dict(declaration=declaration,config=cfg,source=source,parent=record,
            engineering_only=False,data_role='new random TRAIN prefixes; never DEV states',
            original_task_horizon_s=h,state_semantics='real parent prefix; original physical clocks/history/ESO/filter/RNG state, no qpos injection',
            valid_rule='no physical/domain/engineering failure before capture; working roll threshold is diagnostic only',
            route_ids=ids.tolist(),alpha_values=alpha.tolist(),cache=str(cache),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        for start in range(0,steps,50):
            update(stage='collecting',task_seconds=start*.02)
            carry,logs=execute(carry,j.int32(start));jax.block_until_ready(carry)
            ob,mu,valid,fault,active=jax.device_get(logs);ticks+=int(active.sum())
            if fault.any():raise RuntimeError('engineering fault during conditional TRAIN collection')
            observations.append(ob[::5].reshape(-1,351));means.append(mu[::5].reshape(-1,2));anchor_valid.append(valid[::5].reshape(-1))
            update(stage='collecting',task_seconds=(start+50)*.02)
        final,captured,bad,valid=jax.device_get(carry);valid=np.asarray(valid)
        if not valid.any():raise RuntimeError('no valid conditional TRAIN prefix')
        with (out/'bank.pkl').open('wb') as stream:pickle.dump(dict(states=captured,routes=jax.device_get(routes),valid=valid,onset_intervals=np.asarray(onsets),route_ids=ids),stream)
        mask=np.concatenate(anchor_valid);obs=np.concatenate(observations)[mask];mu=np.concatenate(means)[mask]
        np.savez_compressed(out/'anchors.npz',observations=obs,means=mu,log_std=log_std)
        counts={str(a):int(np.sum(valid&(alpha==a))) for a in declaration['alphas']}
        atomic_json(out/'receipt.json',dict(bank_sha256=digest(out/'bank.pkl'),anchors_sha256=digest(out/'anchors.npz'),valid_contexts=int(valid.sum()),rejected_contexts=int((~valid).sum()),valid_by_alpha=counts,anchors=len(obs),actual_lower_ticks=ticks,
            nominal_complete_contexts=int(np.sum(np.asarray(final.tick)>=round(seconds/.005))),
            nominal_physical_failures=int(np.sum(final.physical.failed)),nominal_domain_failures=int(np.sum(final.domain_exit))))
        update(state='completed',phase='completed',stage='bank_ready',valid_contexts=int(valid.sum()),valid_by_alpha=counts)
    except BaseException as exc:
        update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise


def load_parent(campaign, alpha):
    campaign=Path(campaign)
    plan=json.loads((campaign/f'uniform_alpha{alpha}/plan.json').read_text())
    from .tolerance_comparison import resolve_comparison_actors
    record=resolve_comparison_actors(campaign)[('parent',alpha)];manifest=json.loads(Path(plan['parent_manifest']).read_text())
    for key in ('actor','checkpoint'):
        if digest(record[key])!=record[key+'_sha256']:raise ValueError('parent identity changed')
    if record['identity']!=manifest['identity'] or record['identity']['alpha']!=alpha:
        raise ValueError('parent endpoint mismatch')
    return plan,record,manifest,json.loads(Path(plan['config']).read_text())


def collect(campaign,alpha,output,*,smoke=False):
    import jax
    import jax.numpy as j
    from .path_command_training import load_environment
    from .path_command_scenarios import sample_training_route
    from .path_command_policy import PathCommandActor
    from .runtime import configure_compilation_cache
    import torch
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    count=8 if smoke else 512;seconds=1.2 if smoke else 8.;intervals=round(seconds/.02)
    status=dict(phase='running',state='running',pid=os.getpid(),alpha=alpha,
                collected_contexts=0,declared_contexts=count,active_lower_ticks=0,
                maximum_lower_ticks=count*round(seconds/.005),training_steps=0)
    def update(**kw):
        status.update(kw);status['checked_unix']=time.time();status['elapsed_s']=time.monotonic()-started
        atomic_json(out/'status.json',status)
    update(stage='loading')
    try:
        plan,record,parent,cfg=load_parent(campaign,alpha)
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,snapshot=load_environment(cfg,parent['source'],'terminal_contract_v2')
        ids=j.asarray(np.linspace(0,511,count,dtype=np.int32));episode=100
        pose=env.physics.helpers.pose(snapshot.data)
        routes=jax.jit(jax.vmap(lambda eid:sample_training_route(910501,eid,episode,pose,cfg)))(ids)
        states=jax.jit(jax.vmap(lambda path,v:env.reset(snapshot,path,v,j.float32(alpha))))(routes.path,routes.speed)
        env.set_log_template(jax.tree.map(lambda x:x[0],states),jax.tree.map(lambda x:x[0],routes.path),routes.speed[0])
        keys=jax.vmap(lambda eid:jax.random.fold_in(jax.random.PRNGKey(910502),eid))(ids)
        onset_intervals=jax.vmap(lambda key:jax.random.randint(key,(),40,51 if smoke else 176))(keys)
        with Path(record['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
        actor=PathCommandActor()
        saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        if saved['identity']!=record['identity']:raise ValueError('parent checkpoint identity mismatch')
        log_std=saved['policy']['log_std'].detach().numpy()
        atomic_json(out/'manifest.json',dict(experiment='train_parent_prefix_bank_v1',engineering_only=smoke,
            parent=record,source=parent['source'],config=cfg,route_episode=episode,
            route_ids=np.asarray(ids).tolist(),route_seed_effective=cfg['training_sampler']['route_seed'],
            onset_seed=910502,onset_grid_s=[.8,1. if smoke else 3.5,.02],
            nominal_collection_s=seconds,original_task_horizon_s=20.,
            state_semantics='complete real parent zero-force state; original clocks/history/ESO/filters retained',
            data_role='new random TRAIN prefixes; never DEV states',
            anchor_stride_policy_steps=5,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        def one(st,path,v):
            obs,_,fault=env.observation(st,path,v,20.)
            z=actor.apply(params,obs)
            end,logs=env.policy_step(st,z,path,v,20.)
            bad=end.physical.failed|end.domain_exit|end.fault|j.any(logs['peak_roll']>cfg['stage_B_baseline']['working_roll_rad'])
            return end,obs,z,bad,fault|j.any(logs['policy_fault']|logs['lower_fault']),j.sum(logs['active_tick'])
        advance=jax.vmap(one)
        def chunk(carry,start):
            def step(c,offset):
                st,snap,bad,valid=c;i=start+offset
                choose=i==onset_intervals
                snap=select_complete_states(choose,snap,st)
                valid=j.where(choose,~bad,valid)
                end,obs,mu,newbad,fault,active=advance(st,routes.path,routes.speed)
                return (end,snap,bad|newbad,valid),(obs,mu,~bad&~newbad,fault,active)
            return jax.lax.scan(step,carry,j.arange(10,dtype=j.int32))
        execute=jax.jit(chunk);carry=(states,states,j.zeros(count,j.bool_),j.zeros(count,j.bool_))
        observations=[];means=[];anchor_valid=[];active_total=0
        update(stage='collecting')
        for start in range(0,intervals,10):
            carry,logs=execute(carry,j.int32(start));jax.block_until_ready(carry)
            ob,mu,valid,fault,active=jax.device_get(logs)
            if fault.any():raise RuntimeError('engineering fault during parent collection')
            observations.append(ob[::5].reshape(-1,351));means.append(mu[::5].reshape(-1,2));anchor_valid.append(valid[::5].reshape(-1))
            active_total+=int(active.sum());update(active_lower_ticks=active_total,seconds=(start+10)*.02)
        _,captured,_,valid=jax.device_get(carry);valid=np.asarray(valid)
        if not valid.any():raise RuntimeError('no valid TRAIN prefix contexts')
        with (out/'bank.pkl').open('wb') as f:pickle.dump(dict(states=captured,routes=jax.device_get(routes),valid=valid,
            onset_intervals=np.asarray(onset_intervals),route_ids=np.asarray(ids)),f)
        mask=np.concatenate(anchor_valid);obs=np.concatenate(observations)[mask];mu=np.concatenate(means)[mask]
        np.savez_compressed(out/'anchors.npz',observations=obs,means=mu,log_std=log_std)
        atomic_json(out/'receipt.json',dict(bank_sha256=digest(out/'bank.pkl'),anchors_sha256=digest(out/'anchors.npz'),
            valid_contexts=int(valid.sum()),rejected_contexts=int((~valid).sum()),anchors=len(obs),
            active_lower_ticks=active_total,declared_contexts=count,parent_actor_sha256=record['actor_sha256']))
        update(phase='completed',state='completed',stage='bank_ready',collected_contexts=count,
               valid_contexts=int(valid.sum()),rejected_contexts=int((~valid).sum()),anchors=len(obs))
    except BaseException as exc:
        update(phase='error',state='error',error=f'{type(exc).__name__}: {exc}');raise
