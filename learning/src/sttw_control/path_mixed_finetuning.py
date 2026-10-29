"""Weights-only mixed-route continuation using the existing path environment/PPO."""
from pathlib import Path
import json
import os
import pickle
import random
import threading
import time

import jax
import jax.numpy as j
import numpy as np

from .path_command_training import PathBatch, Episode
from .preference_best import atomic_json, config_hash, digest


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    f = cfg['finetuning']; c = cfg['training_future_phase_C']
    from .path_command_scenarios import validate_mixed_finetune_distribution
    validate_mixed_finetune_distribution(cfg)
    if (c['num_envs'], c['rollout_steps'], c['updates_max'], c['episode_s'], f['warmup_updates']) != (512,128,52,10.,2):
        raise ValueError('declared mixed continuation requires 512x128, 2 warmup + 50 PPO, H10')
    if cfg['network'].get('architecture') != 'alpha_modulated_v1':
        raise ValueError('frozen H75 architecture required')
    src = Path(f['initial_source'])
    if not src.is_absolute(): src = path.parent / src
    if digest(src) != f['initial_source_sha256']: raise ValueError('initial source changed')
    source = json.loads(src.read_text())
    if source['bank_index'] != 6: raise ValueError('fresh bank6 required')
    lower=source['candidate']
    for name in ('actor','checkpoint'):
        if digest(lower[name]) != lower[name+'_sha256']:raise ValueError('lower identity changed')
    if digest(lower['config'])!=lower['config_file_sha256']:raise ValueError('lower config changed')
    for name in ('actor','checkpoint'):
        if digest(f['parent'][name]) != f['parent'][name+'_sha256']: raise ValueError('parent changed')
    if digest(source['prepared_bank']) != source['prepared_bank_sha256']: raise ValueError('bank changed')
    import torch
    from .tolerance_training import authenticate_preference_parent_reward
    parent=f['parent']; saved=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
    if saved['identity']!=parent['identity'] or saved['update']!=parent['source_update']:
        raise ValueError('parent checkpoint identity/update mismatch')
    if saved['identity']['lower_actor_sha256']!=source['candidate']['actor_sha256']:
        raise ValueError('frozen lower changed')
    authenticate_preference_parent_reward(parent,saved['identity'],cfg['path_reward'])
    if not {'optimizer','policy'} <= saved.keys():raise ValueError('complete PPO parent required')
    sampling = f['alpha_sampling']
    values = sampling['values']
    probabilities = sampling['probabilities']
    if values != [0.,.25,.5,.75,1.] or len(probabilities)!=5 or min(probabilities)<=0 or not np.isclose(sum(probabilities),1):
        raise ValueError('five positive declared alpha sampling probabilities required')
    return cfg, source


def prepare(config, output):
    cfg, source = load_config(config)
    root = Path(output); root.mkdir(parents=True,exist_ok=False)
    n=cfg['training_future_phase_C']['num_envs']; steps=cfg['training_future_phase_C']['rollout_steps']
    result=dict(state='prepared_not_run',config_sha256=config_hash(cfg),source=source,
        initialization='weights_only actor+critic+log_std; fresh Adam and physical bank reset',
        policy_transitions=52*n*steps,maximum_lower_ticks=52*n*steps*4,
        warmup_updates=2,ppo_updates=50,physics_executed=0,optimizer_updates=0,
        evaluation='none; fixed final update52 candidate, separate later evaluation')
    atomic_json(root/'PREPARATION.json',result)
    return result


class MixedPathBatch(PathBatch):
    def reset(self,eid,episode):
        from .path_command_scenarios import sample_mixed_finetune_route,episode_key
        from .path_command_reward import configured_cost_keys
        pose=self.env.physics.helpers.pose(self.snapshot.data)
        route=sample_mixed_finetune_route(self.seed,eid,episode,pose,self.config)
        f=self.config['finetuning']; sampling=f['alpha_sampling']
        values=j.asarray(sampling['values'])
        probabilities=j.asarray(sampling['probabilities'])
        key=jax.random.fold_in(episode_key(self.seed,eid,episode),991)
        alpha=values[jax.random.choice(key,5,p=probabilities)]
        state=self.env.reset(self.snapshot,route.path,route.speed,alpha)
        return Episode(state,route,eid,episode,j.zeros(6),j.zeros(len(configured_cost_keys(self.config))))


def _tensorboard(root,finetuning):
    """Attach to the existing project server; never stop or replace a service."""
    import urllib.request
    logdir=finetuning.get('tensorboard_logdir')
    pid=None
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():continue
        try:args=(item/'cmdline').read_bytes().split(b'\0');args=[a.decode() for a in args if a]
        except (OSError,UnicodeError):continue
        if not any('tensorboard' in a for a in args):continue
        if '--port' in args and args[args.index('--port')+1]=='6006':
            pid=int(item.name)
            if logdir is None and '--logdir' in args:logdir=args[args.index('--logdir')+1]
            break
    if not logdir:raise RuntimeError('existing port6006 TensorBoard logdir not found; declare tensorboard_logdir')
    target=root/'tensorboard';target.mkdir()
    link=Path(logdir).resolve()/root.name
    if link.exists() or link.is_symlink():raise FileExistsError('TensorBoard run link already exists: '+str(link))
    link.symlink_to(target.resolve(),target_is_directory=True)
    with urllib.request.urlopen('http://localhost:6006',timeout=3) as response:
        if response.status!=200:raise RuntimeError('TensorBoard HTTP unavailable')
    info=dict(url='http://localhost:6006',pid=pid,logdir=str(Path(logdir).resolve()),run=root.name+'/mixed',scalar_verified=False)
    atomic_json(root/'tensorboard_server.json',info)
    return info


def train(config, output):
    import subprocess
    import torch
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter
    from .path_command_training import make_algorithm,save_checkpoint,load_environment,_verify_reward_http
    from .tolerance_training import restore_preference_parent
    from .path_command_statistics import reduce_statistics,merge_statistics,summarize_statistics
    from .lower_command_training import handle_candidate_ppo_result
    from .direct_command_training import clean_numbers,notify
    from .runtime import configure_compilation_cache
    cfg,source=load_config(config);f=cfg['finetuning'];c=cfg['training_future_phase_C']
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    status=dict(state='initializing',stage='loading',pid=os.getpid(),completed_updates=0,declared_updates=52,policy_transitions_total=0,training_control_ticks_total=0)
    lock=threading.RLock();done=threading.Event();writer=None;algo=None
    def update(**kw):
        with lock:
            status.update(kw);status['last_update_epoch']=time.time();atomic_json(root/'status.json',status)
    def heartbeat():
        while not done.wait(4):update()
    update();worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
    try:
        torch.set_num_threads(2);seed=f['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        identity=dict(schema='sttw_path_mixed_finetuning_v1',config_sha256=config_hash(cfg),lower_actor_sha256=source['candidate']['actor_sha256'],parent_checkpoint_sha256=f['parent']['checkpoint_sha256'],seed=seed,reward=cfg['path_reward'])
        atomic_json(root/'effective_config.json',cfg)
        source_dir=root/'source_at_launch';source_dir.mkdir()
        bound_sources={}
        for name in ('path_mixed_finetuning.py','path_command_scenarios.py','path_command_training.py','path_command_policy.py','path_command_env.py','path_command_reward.py','direct_command_ppo.py','path_reference.py','path_command_statistics.py','tolerance_training.py'):
            file=Path(__file__).parent/name;data=file.read_bytes();(source_dir/name).write_bytes(data);bound_sources[name]=digest(file)
        atomic_json(root/'source_identity.json',bound_sources)
        atomic_json(root/'manifest.json',dict(config=cfg,source=source,identity=identity,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initialization='weights_only actor/critic/log_std; fresh Adam/RNG/route counters; new complete prepared bank episodes',selection='fixed final update52; no DEV or reward best',execution_optimizations=dict(ppo='DirectCommandPPO',batch=[512,128],vmap=True,cache=str(cache),statistics='device accumulation',scan='existing policy_step four lower ticks; Python rollout128'),data_role='mixed TRAIN; subsequent evaluations separately declared'))
        tb=_tensorboard(root,f);writer=SummaryWriter(str(root/'tensorboard'/'mixed'))
        started=time.monotonic();env,snapshot=load_environment(cfg,source,'terminal_contract_v2')
        batch=MixedPathBatch(env,snapshot,cfg,seed,0);n=c['num_envs'];steps=c['rollout_steps']
        sample=batch.reset(j.int32(0),j.int32(0));env.set_log_template(sample.state,sample.route.path,sample.route.speed)
        reset=jax.jit(jax.vmap(batch.reset));episodes=reset(j.arange(n,dtype=j.int32),j.zeros(n,j.int32));jax.block_until_ready(episodes)
        observe=jax.jit(jax.vmap(batch.observation))
        def td(ob):return TensorDict({'policy':torch.utils.dlpack.from_dlpack(ob[0]),'critic':torch.utils.dlpack.from_dlpack(ob[1])},batch_size=[n])
        initial=observe(episodes)
        if bool(j.any(initial[2])):raise RuntimeError('reset observation fault')
        obs=td(initial);algo=make_algorithm(obs,steps,cfg,'cuda')
        receipt=restore_preference_parent(algo,f['parent'],source);atomic_json(root/'initialization_receipt.json',receipt)
        abstract=jax.eval_shape(jax.vmap(batch.advance_one),episodes,j.zeros((n,2)))
        zero=dict(diagnostics=reduce_statistics(jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),abstract[5]['diagnostics'])),ticks=j.int32(0),reward_sum=j.float32(0),physical=j.int32(0),domain=j.int32(0),scene=j.zeros(2,j.int32),alpha=j.zeros(5,j.int32))
        def advance(ep,z,acc):
            end,ob,r,flags,final,stats=jax.vmap(batch.advance_one)(ep,z)
            diag=reduce_statistics(stats['diagnostics'])
            out=dict(diagnostics=merge_statistics(acc['diagnostics'],diag),ticks=acc['ticks']+j.sum(stats['active_ticks']),reward_sum=acc['reward_sum']+j.sum(r),physical=acc['physical']+j.sum(flags['physical_failure']),domain=acc['domain']+j.sum(flags['tracking_domain_failure']),scene=acc['scene']+j.sum(ep.route.sample_group[:,None]==j.arange(2),axis=0),alpha=acc['alpha']+j.sum(ep.state.alpha[:,None]==j.asarray([0.,.25,.5,.75,1.]),axis=0))
            return end,ob,r,flags,out,j.any(flags['engineering_fault'])|j.any(ob[2])
        update(stage='compiling');advance=jax.jit(advance).lower(episodes,j.zeros((n,2)),zero).compile()
        update(initialization_seconds=time.monotonic()-started)
        def checkpoint(i):
            cp=save_checkpoint(root/'checkpoints',algo,i,identity,rejections=rejections,route_counters=episodes.episode)
            atomic_json(root/'last_completed.json',dict(update=i,checkpoint=str(cp.resolve()),checkpoint_sha256=digest(cp)))
            return cp
        rejections=0;checkpoint(0);verified=False
        for iteration in range(1,53):
            update(state='running',stage='sampling',rollout_intervals=0);acc=zero;start=time.monotonic();previous=status['training_control_ticks_total']
            with torch.no_grad():
                for index in range(steps):
                    z=algo.act(obs)
                    if not bool(torch.isfinite(z).all()):raise RuntimeError('nonfinite action')
                    episodes,ob,reward,flags,acc,fault=advance(episodes,jax.dlpack.from_dlpack(z.detach().contiguous()),acc)
                    failed=bool(fault);status['policy_transitions_total']+=n
                    if failed:
                        with (root/'engineering_fault_states.pkl').open('wb') as stream:pickle.dump(jax.device_get(episodes),stream)
                        update(training_control_ticks_total=previous+int(acc['ticks']))
                        raise RuntimeError('engineering fault; partial rollout discarded')
                    obs=td(ob);algo.process_env_step(obs,torch.utils.dlpack.from_dlpack(reward),torch.utils.dlpack.from_dlpack(flags['done']),{})
                    if index==0 and iteration==1:
                        atomic_json(root/'first_transition_receipt.json',dict(actor_shape=list(ob[0].shape),critic_shape=list(ob[1].shape),actual_policy_transitions=n,included_in_training=True,extra_physics=0))
                    if (index+1)%16==0:update(rollout_intervals=index+1,training_control_ticks_total=previous+int(acc['ticks']))
                algo.compute_returns(obs)
            totals=jax.device_get(acc);sample_seconds=time.monotonic()-start;update(stage='optimizing');start=time.monotonic()
            if iteration<=2:
                metrics=algo.value_only_update(c['epochs']);metrics.update(surrogate=0.,entropy=0.,mean_kl=0.,accepted_epochs=c['epochs'],nonfinite_stop=False);stop=False
            else:metrics=algo.update();rejections,stop=handle_candidate_ppo_result(algo,metrics,rejections,f['optimizer_guard'])
            row=dict(update=iteration,sampling_model_update=iteration-1,training_stage='critic_warmup' if iteration<=2 else 'ppo',completed_ppo_updates=max(0,iteration-2),mean_step_reward=float(totals['reward_sum']/(n*steps)),physical_failures=int(totals['physical']),domain_failures=int(totals['domain']),scene_names=['low_speed','retention'],scene_policy_sample_counts=totals['scene'].tolist(),alpha_policy_sample_counts=totals['alpha'].tolist(),optimizer=clean_numbers(metrics),diagnostics=clean_numbers(summarize_statistics(totals['diagnostics'],cfg)),sample_seconds=sample_seconds,optimize_seconds=time.monotonic()-start,stop_reason=stop)
            row['diagnostics']['route_distribution']['sample_group_names']=['low_speed','retention','unused']
            with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            for key in ('mean_step_reward','physical_failures','domain_failures'):writer.add_scalar('train/'+key,row[key],iteration)
            for key,value in row['optimizer'].items():
                if isinstance(value,(int,float)):writer.add_scalar('ppo/'+key,value,iteration)
            writer.flush()
            if not verified:verified=_verify_reward_http(root,tb,tb['run'])
            update(completed_updates=iteration,completed_ppo_updates=max(0,iteration-2),training_control_ticks_total=previous+int(totals['ticks']))
            if iteration%5==0 or iteration==52 or stop:cp=checkpoint(iteration)
            if stop:raise RuntimeError('optimizer guard stopped stage: '+str(stop))
        update(state='completed',stage='training_complete',final_checkpoint=str(cp),stage_notified=False)
        notify('STTW训练阶段完成','低速混合续训已完成：2次critic预热及50次PPO；固定update52，尚未评价。')
        update(stage_notified=True,completion_notified=False)
        notify('STTW流水线完成','低速混合训练产物已保存；未自动运行开发评估。')
        update(completion_notified=True)
    except BaseException as exc:
        update(state='failed',stage='error',error=repr(exc))
        notify('STTW训练错误',str(exc));raise
    finally:
        done.set();worker.join(timeout=5)
        if writer is not None:writer.close()
