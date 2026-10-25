"""Finite fresh-policy live collector engineering; zero optimizer updates."""
import json,os,pickle,time
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest


def prepare(root):
    import traceback
    from prepare_short_recovery_campaign import register
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    state=dict(phase='running',pid=os.getpid(),physics_steps=0,optimizer_updates=0)
    atomic_json(root/'prepare_status.json',state)
    register(root/'ACTIVE_PREPARE_RUN.json','STTW_CONTROL 实时偏好零优化采样准备',root/'prepare_status.json')
    try:
        _prepare(root)
        state.update(phase='completed',checked_unix=time.time());atomic_json(root/'prepare_status.json',state)
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc());state.update(phase='error',error=repr(exc),checked_unix=time.time());atomic_json(root/'prepare_status.json',state);raise


def _prepare(root):
    from prepare_short_recovery_campaign import register
    base=Path.cwd().resolve();root=root.resolve()
    source=base/'runs/early_support_prospective_v1';old=json.loads((source/'plan.json').read_text())
    assert json.loads((source/'status.json').read_text())['phase']=='completed'
    context={};cases=[];model=None
    for ri,short in enumerate(['left','right']):
        local=Path(old['route_run_paths'][short]);p=json.loads((local/'plan.json').read_text())
        assert digest(local/'plan.json')==old['route_plan_sha256'][short]
        auth=json.loads((local/'authenticated_models.json').read_text())['models']['task_only']
        if model is None:model=auth
        else:assert model==auth
        for k in ['actor','checkpoint']:assert digest(auth[k])==auth[k+'_sha256']
        cfg=(local/'common_environment_config.json').read_bytes()
        if short=='left':(root/'effective_config.json').write_bytes(cfg);(root/'source_declaration.json').write_bytes((local/'source_declaration.json').read_bytes())
        else:assert cfg==(root/'effective_config.json').read_bytes()
        for key,sha in [('source_initial_state','source_initial_state_sha256'),('path_file','path_sha256')]:assert digest(p[key])==p[sha]
        template=json.loads((local/'inventory.json').read_text())['cases'][0]
        context[short]=dict(initial=p['source_initial_state'],initial_sha256=p['source_initial_state_sha256'],path=p['path_file'],path_sha256=p['path_sha256'])
        for ei,force in enumerate([0.,-32.,32.]):
            for profile in [0,7]:
                cases.append(dict(route=template['route'],context=short,speed=template['requested_speed_m_s'],alpha=.5,force_N=force,
                    world_direction=template['world_direction'],pulse_onset_task_tick=480 if short=='left' else 456,
                    pulse_duration_s=.1,profile=profile,rng_fold_in=ri*3+ei))
    sources={}
    for f in list((base/'learning/src/sttw_control').glob('*.py'))+[base/'learning/cli/collect_live_preference_smoke.py']:
        sources[str(f)]=digest(f);dest=root/'source_at_launch'/f.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    plan=dict(project='STTW_CONTROL',purpose='Engineering only: raw densities, complete live rewards and legal reentry; not controller efficacy.',
        model=model,contexts=context,cases=cases,config_sha256=digest(root/'effective_config.json'),declaration_sha256=digest(root/'source_declaration.json'),
        source_sha256=sources,declared_cases=12,horizon_s=10.,maximum_active_lower_ticks=24000,optimizer_updates=0,
        seed=2026102521,acceleration=['vmap12 actual per-lane parameter layout','scan50','persistent JAX cache'],
        stop='One12-case engineering collection; no optimizer, automatic retry, extra cases or training. Unknown/fault retains raw and stops.',
        limitation='Old two routes, one executedalpha, new stochastic sampling. Unseen branches remain unverified; no control improvement or independent test claim.')
    atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=12,completed_cases=0,active_lower_ticks=0,optimizer_updates=0))
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 实时偏好采样12回合零优化工程检查',root/'status.json')
    register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 实时偏好采样形状检查',root/'shape/status.json')


def execute(root,*,shape_only=False):
    import traceback
    root=root.resolve();directory=root/'shape' if shape_only else root/'execution';directory.mkdir(exist_ok=False)
    previous=json.loads((root/'status.json').read_text())
    if previous['phase']!='prepared':raise ValueError('fresh prepared engineering stage only')
    status=dict(phase='running',stage='authenticate',pid=os.getpid(),declared_cases=12,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0)
    status_file=directory/'status.json' if shape_only else root/'status.json'
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(status_file,status)
    update()
    try:
        import jax
        import jax.numpy as j
        import torch
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor,make_path_critic
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate
        from .survival_online_learner import OnlineSurvivalLearner
        from .survival_live_collection import LiveSurvivalCollectionState,make_live_survival_collection_step
        from .survival_live_batch import assemble_live_arm
        from .anticipatory_run import flatten_ticks,physical_outcome
        p=json.loads((root/'plan.json').read_text());assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        for file,sha in p['source_sha256'].items():assert digest(file)==sha,file
        for n,k in [('effective_config.json','config_sha256'),('source_declaration.json','declaration_sha256')]:assert digest(root/n)==p[k]
        for k in ['actor','checkpoint']:assert digest(p['model'][k])==p['model'][k+'_sha256']
        cfg=json.loads((root/'effective_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        torch.set_num_threads(4)
        learner=OnlineSurvivalLearner.restore(cfg,torch.load(p['model']['checkpoint'],map_location='cpu',weights_only=False))
        frozen=learner.freeze_collection();fingerprint=learner.fingerprint();counters=list(learner.optimizer_steps)
        params=learner.sampling_parameters()
        with Path(p['model']['actor']).open('rb') as f:saved=pickle.load(f)
        assert jax.tree.structure(params['actor'])==jax.tree.structure(saved)
        for a,b in zip(jax.tree.leaves(params['actor']),jax.tree.leaves(saved)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2');paths={};origins={}
        for short,c in p['contexts'].items():
            assert digest(c['initial'])==c['initial_sha256'] and digest(c['path'])==c['path_sha256']
            with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
            origins[short]=jax.tree.map(lambda x:j.asarray(x[0]),bank).replace(alpha=j.float32(.5))
            s=origins[short];assert int(s.tick)==0 and int(s.physical.physical_tick)==700 and not bool(s.physical.failed|s.fault|s.domain_exit)
            with np.load(c['path']) as z:paths[short]=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs);cases=p['cases']
        states=[origins[c['context']] for c in cases]
        carry=stack([LiveSurvivalCollectionState(s,jax.random.fold_in(jax.random.PRNGKey(p['seed']),c['rng_fold_in']),initial_gate(),j.int32(0)) for s,c in zip(states,cases)])
        path_batch=stack([paths[c['context']] for c in cases]);speeds=j.asarray([c['speed'] for c in cases],dtype=j.float32)
        pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+.1,*[c['force_N']*d for d in c['world_direction']]] for c in cases],dtype=j.float32)
        profiles=j.asarray([c['profile'] for c in cases],dtype=j.int32)
        env.set_log_template(states[0],paths[cases[0]['context']],speeds[0])
        one=make_live_survival_collection_step(env,make_path_actor(cfg),horizon_s=10.,task_critic=make_path_critic(cfg))
        @jax.jit
        def chunk(carry,parameters,logstd):
            return jax.lax.scan(lambda st,_:jax.vmap(one)(st,parameters,logstd,path_batch,speeds,pulses,profiles),carry,None,length=50)
        parameters=stack([params for _ in cases]);logstd=j.stack([j.asarray(frozen['log_std']) for _ in cases])
        if shape_only:
            shaped=jax.eval_shape(chunk,carry,parameters,logstd)
            assert shaped[1][0]['valid'].shape==(50,12) and shaped[1][1]['active_tick'].shape==(50,12,4)
            atomic_json(directory/'receipt.json',dict(decision_shape=[50,12],lower_shape=[50,12,4],physics_steps=0,optimizer_updates=0));update(phase='completed',stage='shape_complete');return
        assert json.loads((root/'shape/status.json').read_text())['phase']=='completed'
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        with (directory/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        decisions=[];physical=[];started=time.monotonic()
        for block in range(10):
            update(stage='collecting',block=block)
            carry,(d,logs)=chunk(carry,parameters,logstd);jax.block_until_ready(carry);host,d,logs=jax.device_get((carry,d,logs));logs=flatten_ticks(logs)
            decisions.append(d);physical.append(logs)
            np.savez_compressed(directory/f'chunk_{block:02d}.npz',**{'decision_'+k:v for k,v in d.items()},**{'lower_'+k:v for k,v in logs.items()})
            with (directory/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
            update(active_lower_ticks=status['active_lower_ticks']+int(logs['active_tick'].sum()),allocated_lower_slots=status['allocated_lower_slots']+2400,elapsed_s=time.monotonic()-started)
            if status['active_lower_ticks']>24000:raise ValueError('engineering integration budget exceeded')
            if np.any(host.state.fault):raise ValueError('engineering fault; complete raw evidence preserved')
            if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
        d={k:np.concatenate([x[k] for x in decisions]) for k in decisions[0]};physical={k:np.concatenate([x[k] for x in physical]) for k in physical[0]}
        np.savez_compressed(directory/'decisions.npz',**d);np.savez_compressed(directory/'physical.npz',**physical)
        summaries=[];density=[]
        for profile in [0,7]:
            ids=[i for i,c in enumerate(cases) if c['profile']==profile]
            batch,rows=assemble_live_arm(d,physical,cases,ids,collection_id=frozen['collection_id'],dt_s=.005,horizon_s=10.,gamma=cfg['training_future_phase_C']['gamma'],gae_lambda=cfg['training_future_phase_C']['gae_lambda'])
            summaries.extend(rows)
            with torch.no_grad():
                mean=learner.actor(batch['obs']);logp=torch.distributions.Normal(mean,learner.log_std.exp()).log_prob(batch['action']).sum(-1)
                assert torch.allclose(mean,batch['old_mean'],rtol=1e-5,atol=1e-4)
                assert torch.allclose(logp,batch['old_logp'],rtol=0,atol=1e-3)
                value=learner.task_critic(batch['critic']).squeeze(-1)
                density.append(dict(profile=profile,samples=len(mean),max_mean_error=float((mean-batch['old_mean']).abs().max()),max_logp_error=float((logp-batch['old_logp']).abs().max()),max_value_error=float((value-batch['old_task']).abs().max())))
            torch.save(batch,directory/f'profile{profile}_fresh_batch.pt')
        assert learner.fingerprint()==fingerprint and learner.optimizer_steps==counters
        outcomes=[]
        for i,c in enumerate(cases):
            pp={k:v[:,i] for k,v in physical.items()};a=pp['active_tick'];outcome=physical_outcome(a,pp['contact_failure'],pp['nonfinite']|pp['lower_fault']|pp['policy_fault'],2000)
            if outcome=='unknown':raise ValueError('incomplete or unknown engineering episode')
            outcomes.append(dict(c,physical_outcome=outcome,actual_ticks=int(a.sum())))
        atomic_json(directory/'results.json',outcomes);atomic_json(directory/'batch_summary.json',summaries)
        atomic_json(directory/'receipt.json',dict(density_checks=density,optimizer_updates=0,parameter_fingerprint_unchanged=True,collection_id=frozen['collection_id'],actor_decisions=int(d['valid'].sum()),fallback_ticks=int(physical['fallback_active'].sum()),reentry_decisions=sum(x['reentry_decisions'] for x in summaries),raw_decisions_sha256=digest(directory/'decisions.npz'),raw_physical_sha256=digest(directory/'physical.npz'),scope='Engineering only; no optimizer or policy adoption. Unseen branches are not certified.'))
        update(phase='completed',stage='engineering_numeric_complete_graphs_pending',completed_cases=12)
    except BaseException as exc:
        (directory/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
