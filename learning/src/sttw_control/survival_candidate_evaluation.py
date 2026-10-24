"""Paired fresh-noise full-task evaluation of parent and two updated candidates."""
import json,os,pickle,subprocess,sys,time
from pathlib import Path
from .preference_best import atomic_json,digest


def candidate_policy_view(candidate):
    """Evaluation-only native policy view; never pretend to resume training."""
    if not all(k in candidate for k in ['actor','task_critic','log_std']):
        raise ValueError('updated actor, task Critic and logstd required')
    policy={'actor.'+k:v for k,v in candidate['actor'].items()}
    policy.update({'critic.'+k:v for k,v in candidate['task_critic'].items()})
    policy['log_std']=candidate['log_std']
    return dict(policy=policy,exact_training_resume=False,scope='evaluation view of authenticated updated candidate')


def prepare(root):
    import torch
    root.mkdir(parents=True,exist_ok=False)
    source=Path('runs/fresh_survival_collector_v1').resolve()
    updates=Path('runs/dual_survival_update_engineering_v1').resolve()
    old=json.loads((source/'plan.json').read_text());result=json.loads((updates/'results.json').read_text())
    assert json.loads((updates/'status.json').read_text())['phase']=='completed'
    models={'parent':old['model']}
    for row in result['arms']:
        arm=row['arm'];job=updates/arm
        assert digest(job/'checkpoint.pt')==row['checkpoint_sha256']
        assert digest(job/'actor.pkl')==row['actor_sha256']
        checkpoint=torch.load(job/'checkpoint.pt',map_location='cpu',weights_only=False)
        target=root/(arm+'_evaluation_view.pt');torch.save(candidate_policy_view(checkpoint),target)
        models[arm]=dict(actor=str(job/'actor.pkl'),actor_sha256=row['actor_sha256'],
            checkpoint=str(target),checkpoint_sha256=digest(target),
            original_training_checkpoint=str(job/'checkpoint.pt'),original_training_checkpoint_sha256=row['checkpoint_sha256'])
    cases=[dict(c,model=name,source_case=i) for name in models for i,c in enumerate(old['cases'])]
    sources={}
    for p in list(Path('learning/src/sttw_control').rglob('*.py'))+[Path('learning/cli/evaluate_survival_candidates.py')]:
        p=p.resolve();sources[str(p)]=digest(p)
        target=root/'source_at_launch'/p.relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    plan=dict(project='STTW_CONTROL',purpose='First actual post-update paired development evaluation, not independent efficacy confirmation',
        source_plan=str(source/'plan.json'),source_plan_sha256=digest(source/'plan.json'),
        training_result=str(updates/'results.json'),training_result_sha256=digest(updates/'results.json'),
        source_declaration=old['source_declaration'],source_declaration_sha256=old['source_declaration_sha256'],
        path_evidence_directory=old['path_evidence_directory'],models=models,cases=cases,seed=2026102411,
        horizon_s=10.,maximum_lower_ticks=24000,declared_cases=12,optimizer_updates=0,
        matched='same initial complete task0 state, common Gaussian noise across models within each force/context pair',
        external_force='fixed world direction and task time from original plan, .1s, not body-aligned to new onset',
        scope='alpha0 only, two low-speed tight turns, one new common-noise seed; all development data',
        stop='Exactly12 episodes up to contact or original10s, no force/seed/threshold sweep or automatic retry',
        source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',completed_cases=0,declared_cases=12,active_lower_ticks=0,optimizer_updates=0))
    sys.path.insert(0,str(Path('learning/cli').resolve()))
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 双目标候选配对完整轨迹评测',root/'status.json')
    print(str(root/'plan.json'))


def execute(root):
    state=dict(phase='running',stage='loading',pid=os.getpid(),completed_cases=0,declared_cases=12,active_lower_ticks=0,optimizer_updates=0)
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
    update()
    try:
        import numpy as np
        import torch
        import jax
        import jax.numpy as j
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor
        from .geometric_path import build_path
        from .survival_collection import SurvivalCollectionState,make_survival_collection_step
        from .survival_collection_run import analyze
        plan=json.loads((root/'plan.json').read_text())
        assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        for p,sha in plan['source_sha256'].items():assert digest(p)==sha,p
        for k in ['source_plan','training_result','source_declaration']:assert digest(plan[k])==plan[k+'_sha256']
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        decl=json.loads(Path(plan['source_declaration']).read_text());cfg=decl['config']
        env,initial=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2')
        pose=np.asarray(env.physics.helpers.pose(initial.data));paths={r['name']:build_path(r['builder'],pose,cfg) for r in decl['routes']}
        cases=plan['cases'];stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        params={};stds={}
        for name,model in plan['models'].items():
            for kind in ['actor','checkpoint']:assert digest(model[kind])==model[kind+'_sha256']
            with Path(model['actor']).open('rb') as stream:params[name]=jax.tree.map(j.asarray,pickle.load(stream))
            checkpoint=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
            stds[name]=j.asarray(checkpoint['policy']['log_std'].numpy())
        initial_states=[env.reset(initial,paths[c['route']],j.float32(c['speed']),j.float32(c['alpha'])) for c in cases]
        assert all(int(s.tick)==0 and not bool(s.fault|s.domain_exit|s.physical.failed) for s in initial_states)
        atomic_json(root/'manifest.json',dict(config=cfg,source=decl['source'],physical_identity=env.physics.bundle.identity,
            model_identities=plan['models'],optimizer_updates=0,seed=plan['seed']))
        carry=stack([SurvivalCollectionState(s,jax.random.fold_in(jax.random.PRNGKey(plan['seed']),c['rng_fold_in']),j.zeros(2),j.bool_(False))
            for s,c in zip(initial_states,cases)])
        first=jax.device_get(carry)
        with (root/'initial_carry.pkl').open('wb') as stream:pickle.dump(first,stream)
        batch_params=stack([params[c['model']] for c in cases]);batch_std=j.stack([stds[c['model']] for c in cases])
        batch_path=stack([paths[c['route']] for c in cases]);speeds=j.asarray([c['speed'] for c in cases])
        start=float(initial.physical_tick)*env.cc.dt
        pulses=j.asarray([[start+c['pulse_onset_task_tick']*.005,start+c['pulse_onset_task_tick']*.005+.1,
            *[c['force_N']*d for d in c['world_direction']]] for c in cases],dtype=j.float32)
        env.set_log_template(initial_states[0],paths[cases[0]['route']],speeds[0])
        one=make_survival_collection_step(env,make_path_actor(cfg),horizon_s=10.)
        @jax.jit
        def chunk(carry):
            return jax.lax.scan(lambda c,_:jax.vmap(one)(c,batch_params,batch_std,batch_path,speeds,pulses),carry,None,length=50)
        decisions=[];logs=[];started=time.monotonic();update(stage='compile_and_collect')
        for block in range(11):
            carry,(decision,log)=chunk(carry);jax.block_until_ready(carry)
            decision,log=jax.device_get((decision,log))
            flat={k:np.swapaxes(v,1,2).reshape((200,len(cases))+v.shape[3:]) for k,v in log.items()}
            decisions.append(decision);logs.append(flat)
            np.savez_compressed(root/f'chunk_{block:02d}.npz',**{'decision_'+k:v for k,v in decision.items()},**{'lower_'+k:v for k,v in flat.items()})
            state['active_lower_ticks']+=int(flat['active_tick'].sum());assert state['active_lower_ticks']<=plan['maximum_lower_ticks']
            update(stage='collecting',completed_blocks=block+1,elapsed_s=time.monotonic()-started)
            if bool(j.any(carry.state.fault)):raise RuntimeError('engineering fault; raw saved, no retry')
            if bool(j.all(carry.state.physical.failed|(carry.state.tick>=2000))):break
        assert bool(j.all(carry.state.physical.failed|(carry.state.tick>=2000)))
        physical={k:np.concatenate([x[k] for x in logs]) for k in logs[0]}
        decision={k:np.concatenate([x[k] for x in decisions]) for k in decisions[0]}
        final=jax.device_get(carry)
        with (root/'final_carry.pkl').open('wb') as stream:pickle.dump(final,stream)
        summaries={}
        for group,(name,model) in enumerate(plan['models'].items()):
            job=root/name;job.mkdir();indices=np.arange(group*4,group*4+4)
            for label,c in [('initial',first),('final',final)]:
                with (job/(label+'_carry.pkl')).open('wb') as stream:pickle.dump(jax.tree.map(lambda x:x[indices],c),stream)
            for i,index in enumerate(indices):
                np.savez_compressed(job/f'case_{i:03d}.npz',**{k:v[:,index] for k,v in physical.items()},**{'decision_'+k:v[:,index] for k,v in decision.items()})
            for route,path in paths.items():np.savez_compressed(job/(route+'_path.npz'),xy=np.asarray(path.xy),s=np.asarray(path.s),goal=np.asarray(path.goal))
            subplan=dict(plan,model=model,cases=[{k:v for k,v in cases[index].items() if k not in ['model','source_case']} for index in indices],
                collection_policy_id=digest(model['checkpoint']),sampling=dict(log_std=np.asarray(stds[name]).tolist()),
                declared_cases=4,maximum_lower_ticks=8000)
            atomic_json(job/'plan.json',subplan);analyze(job,subplan,cfg)
            summaries[name]=json.loads((job/'results.json').read_text())['cases']
        comparison=[]
        for name in ['task_only','physical_credit']:
            for parent,candidate in zip(summaries['parent'],summaries[name]):
                comparison.append(dict(model=name,case=candidate['case'],parent_outcome=parent['outcome'],candidate_outcome=candidate['outcome'],
                    contact_gain=parent['outcome']=='contact' and candidate['outcome']=='survived',
                    contact_loss=parent['outcome']=='survived' and candidate['outcome']=='contact',
                    parent_hold_ticks=parent['hold_ticks'],candidate_hold_ticks=candidate['hold_ticks'],
                    task_return_change=candidate['task_scored_return']-parent['task_scored_return']))
        atomic_json(root/'results.json',dict(models=summaries,paired_comparison=comparison,
            active_lower_ticks=state['active_lower_ticks'],optimizer_updates=0,
            scope='Single fresh-noise DEV comparison after one shared training batch, not a converged-policy or multi-seed result.'))
        (root/'REPORT.md').write_text('# First matched post-update development evaluation\n\n'+
            '\n'.join(f'[{name} all-case XY]({name}/all_xy.png)' for name in summaries)+'\n\n'+
            'All12 cases complete; paired contact/hold/task-return differences are in results.json. These are single-batch candidates, alpha0 only; no efficacy or novelty conclusion from one noise seed.\n')
        update(phase='completed',stage='paired_trajectories_and_reports_ready',completed_cases=12)
    except BaseException as exc:update(phase='error',error=repr(exc));raise
