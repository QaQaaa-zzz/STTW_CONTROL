"""Fresh matched live-task training, fixed endpoints, no automatic restart."""
import json,os,pickle,shutil,socket,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .live_ppo_schedule import make_schedule,make_evaluation_schedule,fresh_learners


def prepare(root):
    from prepare_short_recovery_campaign import register
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 同合同普通PPO对照准备',root/'prepare_status.json')
    try:
        source=Path('runs/live_preference_collector_v1_attempt2').resolve();old=json.loads((source/'plan.json').read_text())
        assert json.loads((source/'status.json').read_text())['phase']=='completed'
        assert digest(source/'plan.json')==json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']
        for k in ['actor','checkpoint']:assert digest(old['model'][k])==old['model'][k+'_sha256']
        templates=[next(c for c in old['cases'] if c['context']==n) for n in ['left','right']]
        train=make_schedule(templates,seed=2026102531,updates=64)
        evaluation=make_evaluation_schedule(templates,seed=2026102533)
        for name in ['effective_config.json','source_declaration.json']:shutil.copyfile(source/name,root/name)
        atomic_json(root/'training_cases.json',train);atomic_json(root/'evaluation_cases.json',evaluation)
        sources={};design=Path('docs/superpowers/plans/2026-10-25-matched-live-ppo-design.md')
        for f in list(Path('learning/src/sttw_control').glob('*.py'))+[Path('learning/cli/train_live_preference.py'),design]:
            sources[str(f.resolve())]=digest(f);dest=root/'source_at_launch'/f
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
        plan=dict(project='STTW_CONTROL',experiment='matched_live_ppo_v1',model=old['model'],contexts=old['contexts'],
            source_plan=str(source/'plan.json'),source_plan_sha256=digest(source/'plan.json'),
            updates=64,critic_only_updates=2,epochs=4,episodes_per_arm=60,total_envs=120,horizon_s=10.,
            seed=2026102534,learner_seed=2026102532,arms={'plain':0,'support':7},
            maximum_active_lower_ticks=15360000,maximum_allocated_lower_slots=15360000,gamma=.997,gae_lambda=.99,
            initialization='weights_only; common native task64 heads/std; fresh Adam',
            outer_settings=dict(beta=0.,epochs=4,fixed_denominator=30000,actor_lr=.0001,critic_lr=.0003,clip=.2,entropy=.001,grad_clip=1.,entropy_normalization='fixed_slots'),
            source_sha256=sources,files_sha256={n:digest(root/n) for n in ['effective_config.json','source_declaration.json','training_cases.json','evaluation_cases.json']},
            code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            adoption='none; final64 endpoint only; development optimization repeat, not final independent test',
            stop='64 complete collections; error/unknown preserves and stops; no automatic retry/extension or best selection',
            acceleration=['vmap120 real per-lane parameters','dynamic pulse/parameters scan50 JIT','persistent cache'],
            first_chunk='first real50 upper steps belong to collection1; no separate repeated engineering physics')
        atomic_json(root/'effective_training_contract.json',dict(environment=json.loads((root/'effective_config.json').read_text()),
            outer_settings=plan['outer_settings'],gamma=.997,gae_lambda=.99,updates=64,critic_only_updates=2,
            live_reentry=True,horizon_s=10.,sampling_schedule_sha256=digest(root/'training_cases.json')))
        plan['files_sha256']['effective_training_contract.json']=digest(root/'effective_training_contract.json')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',completed_updates=0,declared_updates=64,active_lower_ticks=0,
            allocated_lower_slots=0,legal_decisions_collected=0,actor_training_samples=0,tensorboard_reward_verified=False,
            optimizer_steps_by_arm={'plain':[0,0,0],'support':[0,0,0]}))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 实时偏好普通PPO有无速度支持64轮',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 实时偏好训练120路形状检查',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),checked_unix=time.time()))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',error=repr(exc),pid=os.getpid(),checked_unix=time.time()));raise


def _tensorboard(root):
    from torch.utils.tensorboard import SummaryWriter
    import urllib.request
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    with (root/'tensorboard.log').open('x') as stream:
        process=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),
            '--port',str(port),'--host','127.0.0.1','--reload_interval','1'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    url=f'http://localhost:{port}'
    for _ in range(30):
        if process.poll() is not None:raise RuntimeError('TensorBoard process exited')
        try:
            with urllib.request.urlopen(url,timeout=2) as response:
                if response.status==200:break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('TensorBoard HTTP unavailable')
    atomic_json(root/'tensorboard_server.json',dict(pid=process.pid,url=url,logdir=str(root/'tensorboard'),reward_scalar_loaded=False))
    return url,{n:SummaryWriter(str(root/'tensorboard'/n)) for n in ['plain','support']}


def _verify_reward(root,url,iteration,expected):
    import urllib.request
    from urllib.parse import urlencode
    for _ in range(30):
        try:
            actual={}
            for n in expected:
                query=urlencode(dict(run=n,tag='rollout/reward'))
                with urllib.request.urlopen(url+'/data/plugin/scalars/scalars?'+query,timeout=2) as r:actual[n]=json.load(r)
            if all(any(x[1]==iteration and np.isclose(x[2],v,rtol=1e-6) for x in actual[n]) for n,v in expected.items()):break
        except (OSError,ValueError):pass
        time.sleep(1)
    else:raise RuntimeError('actual current reward scalar missing through TensorBoard HTTP')
    atomic_json(root/'tensorboard_verified.json',dict(url=url,reward_scalar_loaded=True,checked_collection=iteration,checked_unix=time.time()))


def execute(root,*,shape_only=False):
    root=root.resolve();previous=json.loads((root/'status.json').read_text())
    if previous['phase']!='prepared':raise ValueError('fresh prepared execution only; no implicit restart')
    directory=root/'shape' if shape_only else root/'execution';directory.mkdir(exist_ok=False)
    state=dict(previous,phase='running',stage='authenticating',pid=os.getpid());writers={}
    status_file=directory/'status.json' if shape_only else root/'status.json'
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(status_file,state)
    update()
    try:
        import torch,jax
        import jax.numpy as j
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor,make_path_critic,export_torch_actor
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate
        from .survival_live_collection import LiveSurvivalCollectionState,make_live_survival_collection_step
        from .survival_live_batch import assemble_live_arm
        from .anticipatory_run import flatten_ticks
        p=json.loads((root/'plan.json').read_text())
        assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        for f,sha in p['source_sha256'].items():assert digest(f)==sha,f
        for f,sha in p['files_sha256'].items():assert digest(root/f)==sha,f
        assert digest(p['source_plan'])==p['source_plan_sha256']
        for k in ['actor','checkpoint']:assert digest(p['model'][k])==p['model'][k+'_sha256']
        cfg=json.loads((root/'effective_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        assert cfg['training_future_phase_C']['gamma']==p['gamma'] and cfg['training_future_phase_C']['gae_lambda']==p['gae_lambda']
        torch.set_num_threads(4)
        learners=fresh_learners(cfg,torch.load(p['model']['checkpoint'],map_location='cpu',weights_only=False),seed=p['learner_seed'])
        assert learners['plain'].fingerprint()==learners['support'].fingerprint()
        with Path(p['model']['actor']).open('rb') as f:old_actor=pickle.load(f)
        new_actor=learners['plain'].sampling_parameters()['actor']
        assert jax.tree.structure(new_actor)==jax.tree.structure(old_actor)
        for a,b in zip(jax.tree.leaves(new_actor),jax.tree.leaves(old_actor)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        atomic_json(directory/'initialization_receipt.json',dict(parameter_fingerprint=learners['plain'].fingerprint(),both_arms_identical=True,
            source_actor_sha256=p['model']['actor_sha256'],fresh_adam=True,optimizer_steps=[o.optimizer_steps for o in learners.values()],actual_settings=learners['plain'].settings))
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2');paths={};origins={}
        for name,c in p['contexts'].items():
            assert digest(c['initial'])==c['initial_sha256'] and digest(c['path'])==c['path_sha256']
            with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
            s=jax.tree.map(lambda x:j.asarray(x[0]),bank)
            assert int(s.tick)==0 and int(s.physical.physical_tick)==700 and not bool(s.physical.failed|s.fault|s.domain_exit)
            origins[name]=s
            with np.load(c['path'],allow_pickle=False) as z:paths[name]=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        schedule=json.loads((root/'training_cases.json').read_text());cases=schedule[0]
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        states=[origins[c['context']].replace(alpha=j.float32(c['alpha'])) for c in cases]
        batch_path=stack([paths[c['context']] for c in cases]);speeds=j.asarray([c['speed'] for c in cases],j.float32)
        profiles=j.asarray([c['profile'] for c in cases],j.int32)
        env.set_log_template(states[0],paths[cases[0]['context']],speeds[0])
        one=make_live_survival_collection_step(env,make_path_actor(cfg),horizon_s=10.,task_critic=make_path_critic(cfg))
        @jax.jit
        def chunk(carry,params,logstd,pulses):
            return jax.lax.scan(lambda s,_:jax.vmap(one)(s,params,logstd,batch_path,speeds,pulses,profiles),carry,None,length=50)
        def collection_inputs(index):
            cases=schedule[index];key=jax.random.fold_in(jax.random.PRNGKey(p['seed']),index)
            carry=stack([LiveSurvivalCollectionState(s,jax.random.fold_in(key,c['rng_fold_in']),initial_gate(),j.int32(0)) for s,c in zip(states,cases)])
            params={n:o.sampling_parameters() for n,o in learners.items()}
            params=stack([params[c['arm']] for c in cases])
            logstd=j.stack([j.asarray(learners[c['arm']].log_std.detach().numpy()) for c in cases])
            pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+.1,
                *[c['force_N']*d for d in c['world_direction']]] for c in cases],j.float32)
            return carry,params,logstd,pulses
        if shape_only:
            shaped=jax.eval_shape(chunk,*collection_inputs(0))
            assert shaped[1][0]['valid'].shape==(50,120) and shaped[1][1]['active_tick'].shape==(50,120,4)
            atomic_json(directory/'receipt.json',dict(decision_shape=[50,120],lower_shape=[50,120,4],physics_steps=0,optimizer_updates=0))
            update(phase='completed',stage='shape_complete');return
        assert json.loads((root/'shape/status.json').read_text())['phase']=='completed'
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        url,writers=_tensorboard(root);update(tensorboard_url=url,stage='starting_first_collection')
        def save_pair(index):
            records=[]
            for name,obj in learners.items():
                assert obj.next_collection==index
                d=root/name/'checkpoints';d.mkdir(parents=True,exist_ok=True)
                pt=d/f'learner_{index:04d}.pt';actor=d/f'actor_{index:04d}.pkl'
                torch.save(obj.checkpoint(),pt)
                with actor.open('wb') as f:pickle.dump(jax.device_get(export_torch_actor(obj.actor,cfg)),f)
                records.append(dict(arm=name,profile=p['arms'][name],checkpoint=str(pt),checkpoint_sha256=digest(pt),actor=str(actor),actor_sha256=digest(actor),fingerprint=obj.fingerprint()))
            atomic_json(root/f'checkpoint_pair_{index:04d}.json',dict(completed_updates=index,models=records,plan_sha256=digest(root/'plan.json'),
                active_lower_ticks=state['active_lower_ticks'],allocated_lower_slots=state['allocated_lower_slots']))
        save_pair(0);started=time.monotonic()
        for iteration,cases in enumerate(schedule):
            droot=directory/f'update_{iteration+1:04d}';droot.mkdir(exist_ok=False)
            frozen={n:o.freeze_collection() for n,o in learners.items()}
            atomic_json(droot/'sampling_identity.json',dict(models=frozen,cases=cases,plan_sha256=digest(root/'plan.json')))
            carry,params,logstd,pulses=collection_inputs(iteration)
            with (droot/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            decisions=[];physical=[];warmup=iteration<p['critic_only_updates']
            update(stage='collecting',current_update=iteration+1,training_phase='critic_warmup' if warmup else 'ppo',completed_blocks_current_update=0)
            for block in range(10):
                carry,(d,logs)=chunk(carry,params,logstd,pulses);jax.block_until_ready(carry)
                host,d,logs=jax.device_get((carry,d,logs));logs=flatten_ticks(logs)
                decisions.append(d);physical.append(logs)
                np.savez_compressed(droot/f'chunk_{block:02d}.npz',**{'decision_'+k:v for k,v in d.items()},**{'lower_'+k:v for k,v in logs.items()})
                state['active_lower_ticks']+=int(logs['active_tick'].sum());state['allocated_lower_slots']+=24000
                assert state['active_lower_ticks']<=p['maximum_active_lower_ticks'] and state['allocated_lower_slots']<=p['maximum_allocated_lower_slots']
                update(completed_blocks_current_update=block+1,elapsed_training_s=time.monotonic()-started,
                    completed_episodes_current_update=int(np.sum(host.state.physical.failed|(host.state.tick>=2000))))
                if np.any(host.state.fault):raise RuntimeError('engineering fault; raw preserved, no automatic retry')
                if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
            with (droot/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
            decision={k:np.concatenate([x[k] for x in decisions]) for k in decisions[0]};phys={k:np.concatenate([x[k] for x in physical]) for k in physical[0]}
            np.savez_compressed(droot/'decisions.npz',**decision);np.savez_compressed(droot/'physical.npz',**phys)
            batches={};rows={}
            for name in learners:
                ids=[i for i,c in enumerate(cases) if c['arm']==name]
                batches[name],rows[name]=assemble_live_arm(decision,phys,cases,ids,collection_id=frozen[name]['collection_id'],dt_s=.005,horizon_s=10.,gamma=p['gamma'],gae_lambda=p['gae_lambda'])
                assert len(rows[name])==60
            atomic_json(droot/'collection_complete.json',dict(case_summaries=rows,raw_decisions_sha256=digest(droot/'decisions.npz'),raw_physical_sha256=digest(droot/'physical.npz'),all_known=True))
            expected={};update(stage='updating')
            for name,obj in learners.items():
                result=obj.update(batches[name],warmup=warmup);atomic_json(droot/(name+'_learner_result.json'),result)
                torch.save(obj.checkpoint(),droot/(name+'_post_update.pt'))
                records=rows[name];writer=writers[name];expected[name]=float(np.mean([r['task_scored_return'] for r in records]))
                for k,v in dict(reward=expected[name],episode_survival=np.mean([r['outcome']=='survived' for r in records]),legal_actor_decisions=result['samples'],
                    fallback_s=sum(r['fallback_ticks'] for r in records)*.005,mean_full_pulse_exposure=np.mean([r['full_pulse_exposure'] for r in records]),actual_lower_ticks=sum(r['physical_ticks'] for r in records)).items():writer.add_scalar('rollout/'+k,v,iteration+1)
                for alpha in [0.,.25,.5,.75,1.]:
                    chosen=[r for r in records if r['alpha']==alpha];tag=f'alpha_{alpha:g}'
                    for k,v in dict(reward=np.mean([r['task_scored_return'] for r in chosen]),survival=np.mean([r['outcome']=='survived' for r in chosen]),
                        legal_actor_decisions=sum(r['actor_decisions'] for r in chosen),max_path_error=np.mean([r['max_task_path_error'] for r in chosen])).items():writer.add_scalar(tag+'/'+k,v,iteration+1)
                for context in ['left','right']:
                    for force in [0.,-32.,32.]:
                        chosen=[r for r in records if cases[r['case_index']]['context']==context and r['force_N']==force]
                        writer.add_scalar(f'exposure/{context}_F{force:g}_legal_decisions',sum(r['actor_decisions'] for r in chosen),iteration+1)
                        writer.add_scalar(f'exposure/{context}_F{force:g}_actual_ticks',sum(r['physical_ticks'] for r in chosen),iteration+1)
                for k in ['exact_gaussian_KL_mean','likelihood_ratio_clip_fraction','task_advantage_scale','actor_steps','critic_steps']:writer.add_scalar('train/'+k,result[k],iteration+1)
                writer.add_scalar('train/critic_only',int(warmup),iteration+1)
                for k,v in result['history'][-1].items():writer.add_scalar('train/'+k,v,iteration+1)
                writer.flush();state['legal_decisions_collected']+=result['samples']
                if not warmup:state['actor_training_samples']+=result['samples']
            save_pair(iteration+1)
            if iteration in [0,63]:_verify_reward(root,url,iteration+1,expected);state['tensorboard_reward_verified']=True
            update(stage='collection_boundary',completed_updates=iteration+1,optimizer_steps_by_arm={n:o.optimizer_steps for n,o in learners.items()},elapsed_training_s=time.monotonic()-started)
        atomic_json(root/'results.json',dict(completed_updates=64,episodes_per_arm=3840,active_lower_ticks=state['active_lower_ticks'],allocated_lower_slots=state['allocated_lower_slots'],
            evaluation_status='not_started',adoption='none',fixed_endpoint=str(root/'checkpoint_pair_0064.json')))
        for k in ['actor','checkpoint']:assert digest(p['model'][k])==p['model'][k+'_sha256']
        update(phase='completed',stage='training_complete_evaluation_separate')
    except BaseException as exc:
        original_traceback=traceback.format_exc()
        if 'learners' in locals():
            state['optimizer_steps_by_arm']={n:o.optimizer_steps for n,o in learners.items()}
            state['learner_journal']={n:dict(next_collection=o.next_collection,pending_collection=o._pending) for n,o in learners.items()}
            try:
                # May contain a partially applied epoch: diagnostic only, never
                # label this a resumable paired boundary or replay its update.
                torch.save({n:dict(actor=o.actor.state_dict(),task_critic=o.task_critic.state_dict(),
                    physical_critic=o.physical_critic.state_dict(),log_std=o.log_std.detach(),
                    optimizers=[x.state_dict() for x in o.optimizers],optimizer_steps=o.optimizer_steps,
                    next_collection=o.next_collection,pending_collection=o._pending,exact_boundary_resume=False)
                    for n,o in learners.items()},directory/'error_diagnostic_state.pt')
            except BaseException as diagnostic_error:state['diagnostic_write_error']=repr(diagnostic_error)
        (directory/'traceback.txt').write_text(original_traceback)
        update(phase='error',error=repr(exc));raise
    finally:
        for writer in writers.values():writer.close()
