"""One matched fresh-batch dual-objective optimizer engineering round.

This is not an efficacy campaign or automatic continuation. Original policies,
raw collection, and derived task data stay immutable; candidates live separately.
"""
import json,os,pickle,subprocess,sys,time,socket
from pathlib import Path
from .preference_best import atomic_json,digest


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    collection=Path('runs/fresh_survival_collector_v1').resolve()
    task=collection/'task_credit_with_geometry'
    old=json.loads((collection/'plan.json').read_text())
    files=list(Path('learning/src/sttw_control').rglob('*.py'))+[Path('learning/cli/update_survival_objectives.py')]
    sources={}
    for p in files:
        p=p.resolve();sources[str(p)]=digest(p)
        target=root/'source_at_launch'/p.relative_to(Path.cwd());target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    inputs=[collection/'plan.json',collection/'results.json',task/'receipt.json']
    plan=dict(project='STTW_CONTROL',experiment='dual_survival_update_engineering_v1',
        purpose='Authenticate matched task-only/physical-credit optimizer paths, not measure policy gains',
        collection=str(collection),task_credit=str(task),model=old['model'],
        input_sha256={str(p):digest(p) for p in inputs},source_sha256=sources,
        arms={'task_only':0.,'physical_credit':1.},seed=2026102409,
        collection_batches=1,shared_episodes=4,shared_legal_decisions=1029,new_physics_ticks=0,
        epochs=4,declared_optimizer_steps=8,actor_lr=1e-4,critic_lr=3e-4,
        clip=.2,entropy=.001,grad_clip=1.,fixed_denominator=2000,
        task_advantage_normalization='one frozen batch mean/std before all epochs',
        initialization='common H75 Actor/task Critic, logstd; physical Linear352 sigmoid zero weights; fresh Adam',
        task_critic='original weights, fixed original-GAE targets, no critic-only warmup in this engineering check',
        physical_critic='separate sigmoid Linear352, includes remaining horizon, frozen old baseline.5',
        tensorboard_host='127.0.0.1',stop='Exactly4 full-batch PPO epochs per arm; no rollout, retry, extension or adoption',
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',completed_optimizer_steps=0,declared_optimizer_steps=8,
        shared_collection_episodes=4,shared_legal_decisions=1029,new_physics_ticks=0))
    sys.path.insert(0,str(Path('learning/cli').resolve()))
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 双目标匹配更新工程检查',root/'status.json')
    print(str(root/'plan.json'))


def execute(root):
    state=dict(phase='running',stage='loading',pid=os.getpid(),completed_optimizer_steps=0,
        declared_optimizer_steps=8,new_physics_ticks=0,shared_collection_episodes=4,shared_legal_decisions=1029)
    def update(**kw):
        state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
    update()
    try:
        import numpy as np
        import torch
        import jax
        import urllib.request
        from torch.utils.tensorboard import SummaryWriter
        from .path_command_policy import make_torch_path_network,export_torch_actor,make_path_actor
        from .survival_learning import dual_ppo_loss
        plan=json.loads((root/'plan.json').read_text())
        assert digest(root/'plan.json')==json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']
        for p,expected in {**plan['input_sha256'],**plan['source_sha256']}.items():assert digest(p)==expected,p
        for kind in ['actor','checkpoint']:assert digest(plan['model'][kind])==plan['model'][kind+'_sha256']
        collection=Path(plan['collection']);task=Path(plan['task_credit'])
        original=json.loads((collection/'plan.json').read_text())
        cfg=json.loads(Path(original['source_declaration']).read_text())['config']
        assert digest(original['source_declaration'])==original['source_declaration_sha256']
        results=json.loads((collection/'results.json').read_text());receipt=json.loads((task/'receipt.json').read_text())
        assert receipt['optimizer_steps']==0 and receipt['critic_unchanged']
        checkpoint=torch.load(plan['model']['checkpoint'],map_location='cpu',weights_only=False)
        episode_targets=[]
        arrays={k:[] for k in ['obs','critic','action','old_logp','old_task','advantage','returns','physical']}
        for row,record in zip(results['cases'],receipt['cases']):
            index=row['case'];assert index==record['case']
            raw=collection/f'case_{index:03d}.npz';labels=task/f'case_{index:03d}_task.npz'
            assert digest(raw)==row['raw_sha256']==receipt['input_sha256'][str(raw)]
            assert digest(labels)==record['output_sha256']
            with np.load(raw) as z,np.load(labels) as t:
                valid=z['decision_valid'];np.testing.assert_array_equal(valid,t['task_valid'])
                assert not np.any(z['nonfinite']|z['lower_fault']|z['policy_fault'])
                y=0. if np.any(z['contact_failure']) else 1.
                assert y==0. or np.any(z['horizon_reached'])
                episode_targets.append(y)
                for name,a in dict(obs=z['decision_observation'],critic=t['critic_observation'],
                    action=z['decision_raw_action'],old_logp=z['decision_log_prob'],old_task=t['old_task_value'],
                    advantage=t['advantages'],returns=t['returns']).items():arrays[name].append(a[valid])
                arrays['physical'].append(np.full(int(valid.sum()),y,dtype=np.float32))
        data={k:torch.tensor(np.concatenate(a),dtype=torch.float32) for k,a in arrays.items()}
        assert len(data['obs'])==plan['shared_legal_decisions']
        assert all(torch.isfinite(x).all() for x in data.values())
        data['advantage']=(data['advantage']-data['advantage'].mean())/data['advantage'].std(unbiased=False).clamp_min(1e-8)
        rewards=[r['task_scored_return'] for r in results['cases']]
        for port in range(6011,6031):
            with socket.socket() as sock:
                try:sock.bind(('127.0.0.1',port));break
                except OSError:continue
        else:raise RuntimeError('no free TensorBoard port')
        with (root/'tensorboard.log').open('x') as stream:
            tb=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),
                '--port',str(port),'--host','127.0.0.1','--reload_interval','1'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        url=f'http://localhost:{port}'
        atomic_json(root/'tensorboard_server.json',dict(pid=tb.pid,url=url,port=port,logdir=str(root/'tensorboard')))
        update(stage='optimizer_epochs',tensorboard_url=url)
        final=[]
        for arm,beta in plan['arms'].items():
            torch.manual_seed(plan['seed'])
            actor=make_torch_path_network(351,2,cfg);critic=make_torch_path_network(352,1,cfg)
            actor.load_state_dict({k[6:]:v for k,v in checkpoint['policy'].items() if k.startswith('actor.')},strict=True)
            critic.load_state_dict({k[7:]:v for k,v in checkpoint['policy'].items() if k.startswith('critic.')},strict=True)
            physical=torch.nn.Sequential(torch.nn.Linear(352,1),torch.nn.Sigmoid())
            torch.nn.init.zeros_(physical[0].weight);torch.nn.init.zeros_(physical[0].bias)
            std=torch.nn.Parameter(checkpoint['policy']['log_std'].detach().clone())
            actor_params=list(actor.parameters())+[std];task_params=list(critic.parameters());physical_params=list(physical.parameters())
            initial=[p.detach().clone() for p in actor_params]
            optimizer=torch.optim.Adam([{'params':actor_params,'lr':plan['actor_lr']},
                {'params':task_params,'lr':plan['critic_lr']},{'params':physical_params,'lr':plan['critic_lr']}])
            writer=SummaryWriter(str(root/'tensorboard'/arm));history=[]
            writer.add_scalar('rollout/reward',float(np.mean(rewards)),0)
            writer.add_scalar('rollout/action_weighted_survival',float(data['physical'].mean()),0)
            writer.add_scalar('rollout/episode_survival',float(np.mean(episode_targets)),0)
            writer.add_scalar('rollout/legal_decisions',len(data['obs']),0)
            writer.add_text('scope','Single shared pre-update batch; reward is not a post-update evaluation.',0)
            for epoch in range(1,plan['epochs']+1):
                optimizer.zero_grad(set_to_none=True)
                distribution=torch.distributions.Normal(actor(data['obs']),std.exp())
                logp=distribution.log_prob(data['action']).sum(-1)
                if epoch==1:
                    assert float((logp.detach()-data['old_logp']).abs().max())<1e-3
                    torch.testing.assert_close(critic(data['critic']).squeeze(-1),data['old_task'],rtol=1e-5,atol=1e-4)
                losses=dual_ppo_loss(new_log_prob=logp,old_log_prob=data['old_logp'],task_advantage=data['advantage'],
                    task_values=critic(data['critic']).squeeze(-1),task_returns=data['returns'],
                    physical_values=physical(data['critic']).squeeze(-1),physical_targets=data['physical'],
                    old_physical_values=torch.full_like(logp,.5),beta=beta,
                    fixed_denominator=plan['fixed_denominator'],clip=plan['clip'])
                entropy=distribution.entropy().sum(-1).mean()
                total=losses['total_loss']-plan['entropy']*entropy
                assert torch.isfinite(total);total.backward()
                norms=[]
                for group in [actor_params,task_params,physical_params]:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group)
                    norms.append(float(torch.nn.utils.clip_grad_norm_(group,plan['grad_clip'])))
                optimizer.step()
                with torch.no_grad():std.clamp_(np.log(.05),np.log(.5))
                metrics={k:float(v.detach()) for k,v in losses.items()}
                metrics.update(epoch=epoch,entropy=float(entropy.detach()),actor_gradient_norm=norms[0],
                               task_value_gradient_norm=norms[1],physical_value_gradient_norm=norms[2])
                history.append(metrics)
                for k,v in metrics.items():writer.add_scalar('train/'+k,v,epoch)
                writer.flush()
                update(current_arm=arm,current_epoch=epoch,completed_optimizer_steps=state['completed_optimizer_steps']+1)
            writer.close();job=root/arm;job.mkdir()
            with (job/'actor.pkl').open('wb') as stream:pickle.dump(jax.device_get(export_torch_actor(actor,cfg)),stream)
            torch.save(dict(actor=actor.state_dict(),task_critic=critic.state_dict(),physical_critic=physical.state_dict(),
                log_std=std.detach(),optimizer=optimizer.state_dict(),epochs=plan['epochs'],sampling_policy_id=original['collection_policy_id']),job/'checkpoint.pt')
            jax_mean=np.asarray(make_path_actor(cfg).apply(export_torch_actor(actor,cfg),data['obs'][:16].numpy()))
            with torch.no_grad():torch_mean=actor(data['obs'][:16]).numpy()
            np.testing.assert_allclose(jax_mean,torch_mean,atol=1e-5,rtol=1e-5)
            delta=float(torch.sqrt(sum((p.detach()-old).square().sum() for p,old in zip(actor_params,initial))))
            assert delta>0
            atomic_json(job/'history.json',history)
            final.append(dict(arm=arm,beta=beta,optimizer_steps=plan['epochs'],actor_parameter_delta_norm=delta,
                actor_sha256=digest(job/'actor.pkl'),checkpoint_sha256=digest(job/'checkpoint.pt'),
                export_max_error=float(np.max(np.abs(jax_mean-torch_mean)))))
        # Verify actual reward scalars through the HTTP plugin, not event-file presence.
        for attempt in range(30):
            if tb.poll() is not None:raise RuntimeError('TensorBoard server exited')
            try:
                with urllib.request.urlopen(url+'/data/plugin/scalars/tags',timeout=2) as response:tags=json.load(response)
                if all('rollout/reward' in tags.get(arm,{}) for arm in plan['arms']):break
            except (OSError,ValueError):pass
            time.sleep(1)
        else:raise RuntimeError('current reward scalar not loaded through HTTP')
        for arm in plan['arms']:
            with urllib.request.urlopen(url+f'/data/plugin/scalars/scalars?run={arm}&tag=rollout%2Freward',timeout=3) as response:values=json.load(response)
            assert values and np.isclose(values[0][2],np.mean(rewards),rtol=1e-6)
        atomic_json(root/'tensorboard_verified.json',dict(url=url,runs=list(plan['arms']),reward_scalar_loaded=True,checked_unix=time.time()))
        for kind in ['actor','checkpoint']:assert digest(plan['model'][kind])==plan['model'][kind+'_sha256']
        assert final[0]['actor_sha256']!=final[1]['actor_sha256']
        atomic_json(root/'results.json',dict(arms=final,shared_episodes=4,shared_legal_decisions=len(data['obs']),
            new_physics_ticks=0,original_checkpoint_unchanged=True,post_update_evaluation='not_started',
            scope='Matched optimizer/export engineering only; no rollout improvement measured.'))
        update(phase='completed',stage='matched_update_and_tensorboard_verified',completed_arms=2)
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise
