"""Fixed online matched continuation; no replay, auto-extension or adoption."""
import json,os,pickle,subprocess,sys,time,socket,shutil
from pathlib import Path
from .preference_best import atomic_json,digest,config_hash


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    from .survival_online_batch import make_online_cases
    source=Path('runs/fresh_survival_collector_v1/plan.json').resolve();old=json.loads(source.read_text())
    assert digest(old['source_declaration'])==old['source_declaration_sha256']
    for kind in ['actor','checkpoint']:assert digest(old['model'][kind])==old['model'][kind+'_sha256']
    cases=make_online_cases(old['cases'],alphas=(0.,.5,1.),replicates=4)
    design=Path('docs/superpowers/plans/2026-10-24-online-preference-survival-training.md').resolve()
    sources={}
    for p in list(Path('learning/src/sttw_control').rglob('*.py'))+[Path('learning/cli/train_survival_online.py'),design]:
        p=p.resolve();sources[str(p)]=digest(p)
        dest=root/'source_at_launch'/p.relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    declaration=json.loads(Path(old['source_declaration']).read_text())
    atomic_json(root/'effective_config.json',declaration['config'])
    plan=dict(project='STTW_CONTROL',experiment='online_preference_survival_v1',
        question='Does full physical-tail credit improve contact avoidance beyond matched task-only continuation while preserving alpha uses?',
        source_plan=str(source),source_plan_sha256=digest(source),source_declaration=old['source_declaration'],
        source_declaration_sha256=old['source_declaration_sha256'],model=old['model'],
        effective_config_sha256=digest(root/'effective_config.json'),cases=cases,
        arms={'task_only':0.,'physical_credit':1.},seed=2026102417,learner_seed=2026102418,
        updates=64,critic_only_updates=2,epochs=4,episodes_per_arm=48,total_envs=96,horizon_s=10.,
        maximum_active_lower_ticks=64*96*2000,maximum_allocated_lower_slots=64*96*2000,
        actor_lr=1e-4,critic_lr=3e-4,clip=.2,entropy=.001,grad_clip=1.,fixed_denominator=24000,
        actor_steps_per_arm=248,critic_steps_per_head_per_arm=256,torch_threads=4,
        initialization='common original H75; independent new Adam states; zero physical sigmoid Linear352',
        normalizer='original Actor observation scaling; per-complete-batch task advantage mean/std frozen once',
        actor_architecture='unchanged Actor351',physical_value='separate Linear352 sigmoid; remaining horizon supplied',
        snapshot='same actual complete prepared bank3 task0 on every episode; stochastic actions only',
        terminal='task domain/contact/window versus full physical contact/window; irreversible held-reference tail',
        stop='exact64 collections, no wall-time cap or automatic extension; engineering errors retain raw and stop',
        adoption='none; final endpoint models require paired evaluation and nominal preference analysis',
        acceleration=['vmap96','scan50','one dynamic-parameter JIT reused across updates','persistent JAX cache'],
        source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',completed_updates=0,declared_updates=64,
        actor_steps=0,critic_steps_per_head=0,active_lower_ticks=0,allocated_lower_slots=0,
        legal_decisions_collected=0,actor_training_samples=0,tensorboard_reward_verified=False))
    sys.path.insert(0,str(Path('learning/cli').resolve()))
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 多α普通任务与完整防摔信用匹配训练',root/'status.json')
    print(str(root/'plan.json'))


def execute(root):
    previous=json.loads((root/'status.json').read_text())
    if previous['phase']!='prepared':raise ValueError('existing execution cannot be restarted; preserve history and use a reviewed boundary handoff')
    state=dict(previous,phase='running',stage='loading',pid=os.getpid())
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
    update();writers={}
    try:
        import numpy as np
        import torch
        import jax
        import jax.numpy as j
        import urllib.request
        from torch.utils.tensorboard import SummaryWriter
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor,make_path_critic,export_torch_actor
        from .geometric_path import build_path
        from .survival_collection import SurvivalCollectionState,make_survival_collection_step,evaluate_collection_batch
        from .survival_online_learner import OnlineSurvivalLearner
        from .survival_online_batch import assemble_online_arm
        plan=json.loads((root/'plan.json').read_text())
        assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        for p,sha in plan['source_sha256'].items():assert digest(p)==sha,p
        for k in ['source_plan','source_declaration']:assert digest(plan[k])==plan[k+'_sha256']
        assert digest(root/'effective_config.json')==plan['effective_config_sha256']
        for kind in ['actor','checkpoint']:assert digest(plan['model'][kind])==plan['model'][kind+'_sha256']
        cfg=json.loads((root/'effective_config.json').read_text());decl=json.loads(Path(plan['source_declaration']).read_text())
        assert config_hash(cfg)==config_hash(decl['config'])
        torch.set_num_threads(plan['torch_threads'])
        checkpoint=torch.load(plan['model']['checkpoint'],map_location='cpu',weights_only=False)
        learners={name:OnlineSurvivalLearner(cfg,checkpoint['policy'],beta=beta,seed=plan['learner_seed'],
            epochs=plan['epochs'],fixed_denominator=plan['fixed_denominator'],actor_lr=plan['actor_lr'],
            critic_lr=plan['critic_lr'],clip=plan['clip'],entropy=plan['entropy'],grad_clip=plan['grad_clip'])
            for name,beta in plan['arms'].items()}
        # Real saved H75 inputs check all exported heads before expensive physics.
        check_raw=Path(plan['source_plan']).parent/'case_000.npz'
        with np.load(check_raw) as z:
            check_obs=z['decision_observation'][z['decision_valid']][:16]
            check_ticks=z['decision_task_tick'][z['decision_valid']][:16]
        from .survival_task_credit import reconstruct_task_critic
        check_view=reconstruct_task_critic(check_obs,check_ticks,np.ones(len(check_obs),dtype=bool),dt_s=.005,horizon_s=10.)
        check_learner=learners['task_only'];check_params=check_learner.sampling_parameters()
        unique_observations=len(check_obs)
        check_obs=np.tile(check_obs,(plan['total_envs']//unique_observations,1))
        check_view=np.tile(check_view,(plan['total_envs']//unique_observations,1))
        batched_check=jax.tree.map(lambda x:j.stack([x]*plan['total_envs']),check_params)
        values=jax.jit(lambda p,x,v:evaluate_collection_batch(make_path_actor(cfg),make_path_critic(cfg),p,x,v))(
            batched_check,j.asarray(check_obs),j.asarray(check_view))
        with torch.no_grad():expected=(check_learner.actor(torch.tensor(check_obs)).numpy(),
            check_learner.task_critic(torch.tensor(check_view)).squeeze(-1).numpy(),
            check_learner.physical_critic(torch.tensor(check_view)).squeeze(-1).numpy())
        for actual,target in zip(values,expected):np.testing.assert_allclose(np.asarray(actual),target,atol=1e-4,rtol=1e-5)
        atomic_json(root/'prelaunch_value_check.json',dict(source_raw_sha256=digest(check_raw),actual_unique_saved_observations=unique_observations,
            batched_rows=len(check_obs),parameter_layout='per_environment_parameters_matching_production',
            head_max_errors=[float(np.max(np.abs(np.asarray(a)-b))) for a,b in zip(values,expected)],new_physics_ticks=0,optimizer_steps=0))
        for port in range(6011,6031):
            with socket.socket() as sock:
                try:sock.bind(('127.0.0.1',port));break
                except OSError:continue
        else:raise RuntimeError('no free TensorBoard port')
        with (root/'tensorboard.log').open('x') as stream:
            tb=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),
                '--port',str(port),'--host','127.0.0.1','--reload_interval','1'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        url=f'http://localhost:{port}'
        for attempt in range(30):
            if tb.poll() is not None:raise RuntimeError('TensorBoard exited')
            try:
                with urllib.request.urlopen(url,timeout=2) as response:
                    if response.status==200:break
            except OSError:pass
            time.sleep(1)
        else:raise RuntimeError('TensorBoard HTTP unavailable')
        atomic_json(root/'tensorboard_server.json',dict(pid=tb.pid,url=url,port=port,logdir=str(root/'tensorboard'),reward_scalar='pending first actual collection'))
        writers={name:SummaryWriter(str(root/'tensorboard'/name)) for name in learners}
        update(tensorboard_url=url,stage='loading_physics')
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,initial=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2')
        dt=env.cc.dt;assert abs(dt-.005)<1e-10
        pose=np.asarray(env.physics.helpers.pose(initial.data))
        paths={r['name']:build_path(r['builder'],pose,cfg) for r in decl['routes']}
        for route,path in paths.items():np.savez_compressed(root/(route+'_path.npz'),xy=np.asarray(path.xy),s=np.asarray(path.s),goal=np.asarray(path.goal))
        cases=plan['cases'];stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        base_states=[env.reset(initial,paths[c['route']],j.float32(c['speed']),j.float32(c['alpha'])) for c in cases]
        assert all(int(s.tick)==0 and not bool(s.fault|s.domain_exit|s.physical.failed) for s in base_states)
        batch_path=stack([paths[c['route']] for c in cases]);speeds=j.asarray([c['speed'] for c in cases],dtype=j.float32)
        start=float(initial.physical_tick)*dt
        pulses=j.asarray([[start+c['pulse_onset_task_tick']*dt,start+c['pulse_onset_task_tick']*dt+c['pulse_duration_s'],
            *[c['force_N']*d for d in c['world_direction']]] for c in cases],dtype=j.float32)
        env.set_log_template(base_states[0],paths[cases[0]['route']],speeds[0])
        one=make_survival_collection_step(env,make_path_actor(cfg),horizon_s=plan['horizon_s'],task_critic=make_path_critic(cfg))
        @jax.jit
        def chunk(carry,parameters,log_std):
            return jax.lax.scan(lambda c,_:jax.vmap(one)(c,parameters,log_std,batch_path,speeds,pulses),carry,None,length=50)
        atomic_json(root/'manifest.json',dict(physical_identity=env.physics.bundle.identity,source=decl['source'],config=cfg,
            simultaneous_envs=96,original_models_unchanged=True,actual_old_value_heads=True,randomization='actions only; paired keys'))
        def save_pair(index):
            records=[]
            for name,obj in learners.items():
                directory=root/name/'checkpoints';directory.mkdir(parents=True,exist_ok=True)
                pt=directory/f'learner_{index:04d}.pt';actor=directory/f'actor_{index:04d}.pkl'
                torch.save(obj.checkpoint(),pt)
                with actor.open('wb') as stream:pickle.dump(jax.device_get(export_torch_actor(obj.actor,cfg)),stream)
                records.append(dict(arm=name,checkpoint=str(pt),checkpoint_sha256=digest(pt),actor=str(actor),actor_sha256=digest(actor),
                    state_sha256=obj.fingerprint(),sampling_sha256=obj.fingerprint(sampling_only=True),next_collection=obj.next_collection))
            receipt=dict(completed_updates=index,models=records,plan_sha256=digest(root/'plan.json'),
                source_initialization_sha256=plan['model']['checkpoint_sha256'],next_noise_update_index=index,
                base_noise_seed=plan['seed'],cost_counters={k:state[k] for k in ['active_lower_ticks','allocated_lower_slots','legal_decisions_collected','actor_training_samples']},
                resume_scope='exact learner/pair boundary at next task0; no mid-collection or automatic restart')
            target=root/f'checkpoint_pair_{index:04d}.json';atomic_json(target,receipt)
            atomic_json(root/'latest_pair.json',dict(path=str(target),sha256=digest(target)))
            return receipt
        pair=save_pair(0);total_started=time.monotonic()
        for iteration in range(plan['updates']):
            directory=root/'collections'/f'update_{iteration+1:04d}';directory.mkdir(parents=True,exist_ok=False)
            warmup=iteration<plan['critic_only_updates']
            snapshots={name:obj.freeze_collection() for name,obj in learners.items()}
            atomic_json(directory/'sampling_identity.json',dict(models=snapshots,source_pair=pair,noise_update_index=iteration,
                warmup=warmup,plan_sha256=digest(root/'plan.json')))
            sampling={name:obj.sampling_parameters() for name,obj in learners.items()}
            parameters=stack([sampling[c['arm']] for c in cases])
            log_std=j.stack([j.asarray(snapshots[c['arm']]['log_std'],dtype=j.float32) for c in cases])
            key=jax.random.fold_in(jax.random.PRNGKey(plan['seed']),iteration)
            carry=stack([SurvivalCollectionState(s,jax.random.fold_in(key,c['rng_fold_in']),j.zeros(2),j.bool_(False)) for s,c in zip(base_states,cases)])
            with (directory/'initial_carry.pkl').open('wb') as stream:pickle.dump(jax.device_get(carry),stream)
            decisions=[];logs=[];started=time.monotonic()
            update(stage='collecting',current_update=iteration+1,training_phase='critic_warmup' if warmup else 'ppo',
                completed_blocks_current_update=0,completed_episodes_current_update=0)
            for block in range(11):
                carry,(decision,log)=chunk(carry,parameters,log_std);jax.block_until_ready(carry)
                decision,log=jax.device_get((decision,log))
                flat={k:np.swapaxes(v,1,2).reshape((200,len(cases))+v.shape[3:]) for k,v in log.items()}
                decisions.append(decision);logs.append(flat)
                np.savez_compressed(directory/f'chunk_{block:02d}.npz',**{'decision_'+k:v for k,v in decision.items()},**{'lower_'+k:v for k,v in flat.items()})
                state['active_lower_ticks']+=int(flat['active_tick'].sum());state['allocated_lower_slots']+=200*len(cases)
                assert state['active_lower_ticks']<=plan['maximum_active_lower_ticks']
                assert state['allocated_lower_slots']<=plan['maximum_allocated_lower_slots']
                if bool(j.any(carry.state.fault)):raise RuntimeError('collection engineering fault; raw retained, no update or retry')
                stopped=carry.state.physical.failed|(carry.state.tick>=2000)
                update(completed_blocks_current_update=block+1,completed_episodes_current_update=int(stopped.sum()),
                    current_collection_elapsed_s=time.monotonic()-started)
                if bool(j.all(stopped)):break
            assert bool(j.all(carry.state.physical.failed|(carry.state.tick>=2000)))
            physical={k:np.concatenate([x[k] for x in logs]) for k in logs[0]}
            decision={k:np.concatenate([x[k] for x in decisions]) for k in decisions[0]}
            with (directory/'final_carry.pkl').open('wb') as stream:pickle.dump(jax.device_get(carry),stream)
            batches={};rows={}
            for name,obj in learners.items():
                indices=[i for i,c in enumerate(cases) if c['arm']==name]
                batches[name],rows[name]=assemble_online_arm(decision,physical,cases,indices,collection_id=snapshots[name]['collection_id'],
                    dt_s=dt,horizon_s=plan['horizon_s'],gamma=cfg['training_future_phase_C']['gamma'],gae_lambda=cfg['training_future_phase_C']['gae_lambda'])
                np.savez_compressed(directory/(name+'_dataset.npz'),**{k:v.numpy() for k,v in batches[name].items() if k!='collection_id'})
                assert obj.fingerprint()==snapshots[name]['state_sha256']
                state['legal_decisions_collected']+=len(batches[name]['obs'])
                if not warmup:state['actor_training_samples']+=len(batches[name]['obs'])
            atomic_json(directory/'collection_complete.json',dict(case_summaries=rows,physical_ticks=int(physical['active_tick'].sum()),
                allocated_lower_slots=int(physical['active_tick'].size),elapsed_collection_s=time.monotonic()-started,
                raw_chunks={p.name:digest(p) for p in directory.glob('chunk_*.npz')},
                dataset_sha256={name:digest(directory/(name+'_dataset.npz')) for name in learners},all_physical_tails_complete=True))
            update(stage='updating')
            learner_results={}
            for name,obj in learners.items():
                batches[name]['case_ranges']=[dict(case_index=r['case_index'],start=r['vector_start'],end=r['vector_end']) for r in rows[name]]
                result=obj.update(batches[name],warmup=warmup);learner_results[name]=result
                atomic_json(directory/(name+'_learner_result.json'),result)
                torch.save(obj.checkpoint(),directory/(name+'_post_update.pt'))
                update(actor_steps=sum(o.actor_steps for o in learners.values()),
                    critic_steps_per_head=sum(o.critic_steps for o in learners.values()),
                    optimizer_steps_by_arm={k:o.optimizer_steps for k,o in learners.items()})
                writer=writers[name];records=rows[name]
                writer.add_scalar('rollout/reward',np.mean([r['task_scored_return'] for r in records]),iteration+1)
                writer.add_scalar('rollout/episode_survival',np.mean([r['outcome']=='survived' for r in records]),iteration+1)
                writer.add_scalar('rollout/mean_hold_s',np.mean([r['hold_ticks']*dt for r in records]),iteration+1)
                writer.add_scalar('rollout/legal_actor_decisions',result['samples'],iteration+1)
                writer.add_scalar('rollout/actual_full_pulse_exposure_fraction',np.mean([r['full_pulse_exposure'] for r in records]),iteration+1)
                writer.add_scalar('train/critic_only',int(warmup),iteration+1)
                writer.add_scalar('train/actor_steps',obj.actor_steps,iteration+1)
                writer.add_scalar('train/task_advantage_scale',result['task_advantage_scale'],iteration+1)
                writer.add_scalar('train/exact_gaussian_KL',result['exact_gaussian_KL_mean'],iteration+1)
                writer.add_scalar('train/ratio_clip_fraction',result['likelihood_ratio_clip_fraction'],iteration+1)
                for k,v in result['history'][-1].items():writer.add_scalar('train/'+k,v,iteration+1)
                for diagnostic in result['case_shift']:
                    local_case=diagnostic['case_index']%plan['episodes_per_arm']
                    writer.add_scalar(f'case_{local_case:02d}/exact_gaussian_KL',diagnostic['exact_gaussian_KL_mean'],iteration+1)
                    writer.add_scalar(f'case_{local_case:02d}/ratio_clip_fraction',diagnostic['ratio_clip_fraction'],iteration+1)
                for alpha in [0.,.5,1.]:
                    selected=[r for r in records if r['alpha']==alpha];tag='alpha_'+str(alpha).replace('.','p')
                    for key,value in dict(reward=np.mean([r['task_scored_return'] for r in selected]),
                        survival=np.mean([r['outcome']=='survived' for r in selected]),
                        hold_s=np.mean([r['hold_ticks']*dt for r in selected]),
                        actual_samples=sum(r['actor_decisions'] for r in selected),
                        mean_true_speed=np.mean([r['mean_true_speed'] for r in selected]),
                        max_task_path_error=np.mean([r['max_task_path_error'] for r in selected])).items():writer.add_scalar(tag+'/'+key,value,iteration+1)
                writer.flush()
            atomic_json(directory/'learner_results.json',learner_results)
            pair=save_pair(iteration+1)
            if iteration==0 or iteration==plan['updates']-1:
                for attempt in range(30):
                    if tb.poll() is not None:raise RuntimeError('TensorBoard exited')
                    try:
                        with urllib.request.urlopen(url+'/data/plugin/scalars/tags',timeout=2) as response:tags=json.load(response)
                        if all('rollout/reward' in tags.get(name,{}) for name in learners):break
                    except (OSError,ValueError):pass
                    time.sleep(1)
                else:raise RuntimeError('genuine current reward scalar missing through HTTP')
                for name in learners:
                    with urllib.request.urlopen(url+f'/data/plugin/scalars/scalars?run={name}&tag=rollout%2Freward',timeout=3) as response:scalar=json.load(response)
                    assert any(v[1]==iteration+1 and np.isclose(v[2],np.mean([r['task_scored_return'] for r in rows[name]]),rtol=1e-6) for v in scalar)
                atomic_json(root/'tensorboard_verified.json',dict(url=url,reward_scalar_loaded=True,checked_collection=iteration+1,checked_unix=time.time()))
                state['tensorboard_reward_verified']=True
            update(stage='collection_boundary',completed_updates=iteration+1,
                actor_steps=sum(o.actor_steps for o in learners.values()),critic_steps_per_head=sum(o.critic_steps for o in learners.values()),
                elapsed_training_s=time.monotonic()-total_started,latest_pair=str(root/f'checkpoint_pair_{iteration+1:04d}.json'))
        arms=[]
        for record in pair['models']:
            name=record['arm'];shutil.copyfile(record['checkpoint'],root/name/'checkpoint.pt');shutil.copyfile(record['actor'],root/name/'actor.pkl')
            arms.append(dict(arm=name,beta=plan['arms'][name],checkpoint_sha256=digest(root/name/'checkpoint.pt'),actor_sha256=digest(root/name/'actor.pkl'),
                actor_steps=learners[name].actor_steps,critic_steps_per_head=learners[name].critic_steps))
        for kind in ['actor','checkpoint']:assert digest(plan['model'][kind])==plan['model'][kind+'_sha256']
        atomic_json(root/'results.json',dict(arms=arms,completed_updates=64,episodes_per_arm=64*48,
            original_model_unchanged=True,active_lower_ticks=state['active_lower_ticks'],allocated_lower_slots=state['allocated_lower_slots'],
            post_evaluation='not_started',adoption='none',scope='Completed development training; terminal model efficacy still requires matched fresh evaluation and nominal preference checks.'))
        update(phase='completed',stage='training_complete_post_evaluation_separate')
    except BaseException as exc:
        if 'learners' in locals():
            try:
                torch.save({name:dict(actor=obj.actor.state_dict(),task_critic=obj.task_critic.state_dict(),
                    physical_critic=obj.physical_critic.state_dict(),log_std=obj.log_std.detach(),
                    optimizers=[o.state_dict() for o in obj.optimizers],pending_collection=obj._pending,
                    optimizer_steps=obj.optimizer_steps,exact_boundary_resume=False) for name,obj in learners.items()},root/'error_diagnostic_state.pt')
                state['optimizer_steps_by_arm']={k:o.optimizer_steps for k,o in learners.items()}
            except BaseException as diagnostic_error:state['diagnostic_save_error']=type(diagnostic_error).__name__
        update(phase='error',error=repr(exc));raise
    finally:
        for writer in writers.values():writer.close()
