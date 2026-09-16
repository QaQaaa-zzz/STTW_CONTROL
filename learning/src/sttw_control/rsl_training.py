"""RSL-RL PPO on MJX, with latent Gaussian actions and identity-bound Flax export.

The upstream optimizer/GAE remain RSL implementations. Only timeout bootstrap is
adapted: use the final next-state critic before reset, never the previous value.
"""
from pathlib import Path
from dataclasses import asdict
import importlib.metadata
import json
import time
import hashlib
import numpy as np
import jax
import jax.numpy as jp
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

RSL_VERSION='3.2.0'


def torch_from_jax(value):
    return torch.utils.dlpack.from_dlpack(value)


def jax_from_torch(value):
    return jax.dlpack.from_dlpack(value.detach().contiguous())


def bootstrap_timeout(rewards,next_values,truncated,terminated,gamma):
    return rewards+gamma*next_values.reshape_as(rewards)*(truncated & ~terminated).to(rewards.dtype)


def make_algorithm(obs,steps,epochs,minibatches,device,learning_rate=3e-4,gamma=.9995,lam=.99,clip=.2,entropy=.001,std=.15,kl=.01,activation="leaky_relu",hidden_sizes=(256,128)):
    if importlib.metadata.version('rsl-rl-lib')!=RSL_VERSION:
        raise RuntimeError(f'This adapter requires rsl-rl-lib=={RSL_VERSION}')
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},2,
        actor_hidden_dims=list(hidden_sizes),critic_hidden_dims=list(hidden_sizes),activation='elu' if activation=='elu' else 'lrelu',
        init_noise_std=std,noise_std_type='log',actor_obs_normalization=False,critic_obs_normalization=False).to(device)
    linear=[m for m in policy.actor.modules() if isinstance(m,torch.nn.Linear)][-1]
    torch.nn.init.zeros_(linear.weight);torch.nn.init.zeros_(linear.bias)
    storage=RolloutStorage('rl',obs.batch_size[0],steps,obs,[2],device=device)
    return PPO(policy,storage,num_learning_epochs=epochs,num_mini_batches=minibatches,
        clip_param=clip,gamma=gamma,lam=lam,value_loss_coef=.5,entropy_coef=entropy,
        learning_rate=learning_rate,max_grad_norm=.5,use_clipped_value_loss=False,
        schedule='adaptive' if kl else 'fixed',desired_kl=kl,device=device)


def export_actor(policy):
    """Transpose Torch weights into the existing inference-only Flax schema."""
    layers=[m for m in policy.actor.modules() if isinstance(m,torch.nn.Linear)]
    if len(layers)<2 or layers[-1].out_features!=2:raise ValueError('Actor architecture differs from export contract')
    return {'params':{f'Dense_{i}':{'kernel':jp.asarray(m.weight.detach().cpu().numpy().T),
                                    'bias':jp.asarray(m.bias.detach().cpu().numpy())} for i,m in enumerate(layers)}}


def restore_cuda_rng(states):
    """torch.load(map_location=cuda) also moves RNG bytes; CUDA expects CPU bytes."""
    torch.cuda.set_rng_state_all([state.cpu() for state in states])


def train(task_path,output,c):
    from .env import RecoveryEnv,load_config
    from .network import ResidualActor,make_policy_identity,save_policy
    from .training import normalization,should_validate,should_plot
    from .validation import make_validator
    from .selection import rank_tracking_candidate,refresh_best_reward_model,record_training_reward_best
    from .tensorboard_logging import TrainingEvents
    from .training_diagnostics import plot_training,alpha_sample_sums,alpha_sample_summary
    from .runtime import configure_compilation_cache
    configure_compilation_cache()
    cfg=load_config(task_path)
    if cfg.tracking is None:raise ValueError('RSL adapter currently targets the explicit-alpha geometric tracking task')
    if c.command_selection or c.selection_incumbent:raise ValueError('RSL geometry uses geometric acceptance, not command selection')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        target=root/name;tmp=target.with_suffix(target.suffix+'.tmp')
        tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    start=time.monotonic()
    try:
        device='cuda:0' if jax.default_backend()=='gpu' else 'cpu'
        torch.set_num_threads(1);torch.manual_seed(c.seed)
        if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('JAX/PyTorch device mismatch')
        env=RecoveryEnv(cfg,backend='mjx');mean,std=normalization(cfg)
        identity=make_policy_identity(env.bundle.identity,asdict(cfg),cfg.observation.history_steps)
        declaration={'task':asdict(cfg),'training':asdict(c),'policy_identity':identity,'trainer':'rsl_rl.algorithms.PPO',
            'rsl_version':RSL_VERSION,'torch_version':torch.__version__,'device':device,
            'action_contract':'RSL stores pre-tanh Gaussian samples; physics receives tanh(sample); entropy is latent',
            'timeout_contract':'gamma*V(final_next_obs) added only on truncation; RSL time_outs shortcut disabled',
            'kl_contract':'Upstream adaptive learning rate, NOT JAX candidate rejection/rollback',
            'resume_contract':'optimizer/policy/RNG restored; fresh physics and truthful phase warmup',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
        write('declaration.json',declaration);write('status.json',{'phase':'initializing','complete':False})
        reset=jax.vmap(env.reset);step=jax.vmap(env.step)
        key=jax.random.PRNGKey(c.seed);key,rk=jax.random.split(key)
        state=jax.jit(reset)(jax.random.split(rk,c.num_envs));jax.block_until_ready(state.obs)
        warmup_count=0
        if c.warmup_steps:
            key,bk,ik=jax.random.split(key,3);bank=jax.jit(reset)(jax.random.split(bk,c.warmup_pool_size))
            targets=jp.arange(c.warmup_pool_size)*c.warmup_steps//c.warmup_pool_size
            @jax.jit
            def warmup(bank,key):
                def tick(carry,i):
                    bank,key=carry;key,rk=jax.random.split(key);nxt=step(bank,jp.zeros((c.warmup_pool_size,2)))
                    def restart(s):
                        fresh=reset(jax.random.split(rk,c.warmup_pool_size))
                        return jax.tree.map(lambda a,b:jp.where(s.done.reshape((c.warmup_pool_size,)+(1,)*(a.ndim-1)),b,a),s,fresh)
                    nxt=jax.lax.cond(jp.any(nxt.done),restart,lambda s:s,nxt)
                    bank=jax.tree.map(lambda a,b:jp.where((i<targets).reshape((c.warmup_pool_size,)+(1,)*(a.ndim-1)),b,a),bank,nxt)
                    return (bank,key),None
                return jax.lax.scan(tick,(bank,key),jp.arange(c.warmup_steps))[0]
            bank,key=warmup(bank,key);idx=jax.random.randint(ik,(c.num_envs,),0,c.warmup_pool_size)
            state=jax.tree.map(lambda x:x[idx],bank);jax.block_until_ready(state.obs);warmup_count=int(jp.sum(targets))
        write('warmup.json',{'computed_transition_budget':c.warmup_pool_size*c.warmup_steps,'active_transitions':warmup_count})
        scale=torch.tensor(std,device=device)
        def observations(value):return TensorDict({'policy':torch_from_jax(value)/scale},batch_size=[c.num_envs])
        obs=observations(state.obs)
        algo=make_algorithm(obs,c.rollout_steps,c.epochs,c.num_envs*c.rollout_steps//c.minibatch_size,device,
                            c.learning_rate,c.gamma,c.gae_lambda,c.clip,c.entropy_weight,c.initial_std,c.target_kl,c.activation,c.hidden_sizes)
        offset=0;transition_offset=0
        if c.resume_checkpoint:
            snapshot=torch.load(Path(c.resume_checkpoint)/'rsl_snapshot.pt',map_location=device,weights_only=False)
            if snapshot['identity']!=identity or snapshot['rsl_version']!=RSL_VERSION or snapshot.get('activation','leaky_relu')!=c.activation or tuple(snapshot.get('hidden_sizes',(256,128)))!=c.hidden_sizes:raise ValueError('RSL resume identity mismatch')
            algo.policy.load_state_dict(snapshot['policy']);algo.optimizer.load_state_dict(snapshot['optimizer'])
            torch.set_rng_state(snapshot['torch_rng'].cpu())
            if device.startswith('cuda'):restore_cuda_rng(snapshot['cuda_rng'])
            key=jp.asarray(snapshot['jax_rng']);offset=snapshot['update'];transition_offset=snapshot['control_transitions'];algo.learning_rate=snapshot['learning_rate']
        @jax.jit
        def advance(state,action,key):
            nxt=step(state,action);key,rk=jax.random.split(key)
            parts={k:jp.mean(v) for k,v in nxt.tracking_components.items()}
            difference=jp.abs(sum(nxt.tracking_components.values())-nxt.reward)
            error=jp.stack([jp.max(difference/(1+jp.abs(nxt.reward))),jp.max(difference)])
            def restart(s):
                fresh=reset(jax.random.split(rk,c.num_envs))
                return jax.tree.map(lambda a,b:jp.where(s.done.reshape((c.num_envs,)+(1,)*(a.ndim-1)),b,a),s,fresh)
            alpha_stats={}
            if cfg.timed_reference is not None:
                from .timed_reference import errors as reference_errors
                features=jax.vmap(reference_errors)(nxt.pose,nxt.reference_pose,state.reference_command)
                speed=jp.sum(nxt.data.qvel[:,:3]*nxt.data.xmat[:,env.bundle.chassis,:,0],axis=1)
                sampled_errors={'speed_m_s':speed-state.reference_command[:,0],
                                'lateral_m':features[:,0]}
                if cfg.timed_reference.mode=='geometry':
                    sampled_errors['heading_rad']=features[:,1]
                else:
                    sampled_errors.update(yaw_rate_rad_s=nxt.yaw_rate-state.reference_command[:,1],
                                          longitudinal_m=features[:,3])
                alpha_stats=alpha_sample_sums(state.priority_alpha,nxt.reward,nxt.terminated,sampled_errors,xp=jp)
            live=jax.lax.cond(jp.any(nxt.done),restart,lambda s:s,nxt)
            return live,key,nxt.obs,nxt.reward,nxt.done,nxt.truncated,nxt.terminated,parts,error,alpha_stats
        host=lambda x:jax.tree.map(lambda v:np.asarray(v).tolist(),x)
        baseline=None;validate=None
        if not c.training_reward_selection:
            actor=ResidualActor(hidden_sizes=c.hidden_sizes,activation=c.activation)
            validate=make_validator(env,actor,jp.asarray(std),c)
            baseline=host(validate({'actor':export_actor(algo.policy)},True));write('baseline_validation.json',baseline)
        sampling_checkpoint=Path(c.resume_checkpoint).resolve() if c.resume_checkpoint else root/'checkpoints'/f'update_{offset:04d}'
        if c.training_reward_selection and not c.resume_checkpoint:
            save_policy(sampling_checkpoint,export_actor(algo.policy),mean,std,identity,hidden_sizes=c.hidden_sizes,activation=c.activation)
            (sampling_checkpoint/'training.json').write_text(json.dumps({'update':offset,'role':'initial sampling policy'})+'\n')
        write('setup_timings.json',{'initialization_and_baseline_seconds':time.monotonic()-start})
        best=None;best_rank=None;rows=[]
        with TrainingEvents(root/'tensorboard',profile='core') as writer:
            for local in range(1,c.updates+1):
                iteration=offset+local;begin=time.monotonic();reward_sum=torch.zeros((),device=device);component_sum={};alpha_sum={};ends=torch.zeros((),device=device);max_error=0.
                with torch.no_grad():
                    for _ in range(c.rollout_steps):
                        latent=algo.act(obs)
                        state,key,final_obs,reward,done,truncated,terminated,parts,error,alpha_stats=advance(state,jax_from_torch(torch.tanh(latent)),key)
                        final=observations(final_obs);r=torch_from_jax(reward);d=torch_from_jax(done)
                        corrected=bootstrap_timeout(r,algo.policy.evaluate(final).squeeze(-1),torch_from_jax(truncated),torch_from_jax(terminated),c.gamma)
                        obs=observations(state.obs)
                        algo.process_env_step(obs,corrected,d,{})
                        reward_sum+=r.mean();ends+=d.sum()
                        for k,v in alpha_stats.items():alpha_sum[k]=alpha_sum.get(k,0)+torch_from_jax(v)
                        for k,v in parts.items():component_sum[k]=component_sum.get(k,0)+torch_from_jax(v)
                        max_error=torch.maximum(torch.as_tensor(max_error,device=device),torch_from_jax(error))
                    algo.compute_returns(obs)
                if device.startswith('cuda'):torch.cuda.synchronize()
                rollout_seconds=time.monotonic()-begin;optim_start=time.monotonic()
                # RSL update clears the storage cursor but retains arrays for audit.
                metrics=algo.update()
                with torch.no_grad():
                    algo.policy.log_std.clamp_(-4.,0.)
                    stored=algo.storage.observations.flatten(0,1)
                    old_mu=algo.storage.mu.flatten(0,1);old_sigma=algo.storage.sigma.flatten(0,1)
                    algo.policy.act(stored)
                    new_mu=algo.policy.action_mean;new_sigma=algo.policy.action_std
                    kl=torch.sum(torch.log(new_sigma/old_sigma)+(old_sigma**2+(old_mu-new_mu)**2)/(2*new_sigma**2)-.5,dim=-1).mean().item()
                if device.startswith('cuda'):torch.cuda.synchronize()
                optim_seconds=time.monotonic()-optim_start
                error=float(max_error[0])
                if not np.isfinite(error) or error>3e-5:raise RuntimeError(f'RSL reward reconstruction mismatch: {error}')
                record={'update':iteration,'stage_update':local,'mean_step_reward':reward_sum.item()/c.rollout_steps,
                    'control_transitions':transition_offset+local*c.num_envs*c.rollout_steps,'stage_control_transitions':local*c.num_envs*c.rollout_steps,
                    'episode_ends':int(ends.item()),'loss_metrics':[metrics['surrogate'],metrics['value'],metrics['entropy'],kl],
                    'learning_rate':algo.learning_rate,'rollout_seconds':rollout_seconds,'optimizer_seconds':optim_seconds,
                    'optimizer_audit':{'accepted_minibatches':c.epochs*(c.num_envs*c.rollout_steps//c.minibatch_size),'full_update_rolled_back':False},
                    'reward_components_mean_step':{k:v.item()/c.rollout_steps for k,v in component_sum.items()},'reward_components_reconstruction_max_scaled':error,'reward_components_reconstruction_max_abs':float(max_error[1])}
                if alpha_sum:
                    record['alpha_training_samples']=alpha_sample_summary({k:v.cpu().numpy() for k,v in alpha_sum.items()})
                    record['alpha_training_scope']='random training samples in three alpha intervals; not paired fixed-alpha evaluation; errors use pre-step commands and post-step physics'
                if c.training_reward_selection:
                    record['sampling_checkpoint']=str(sampling_checkpoint)
                    record['sampling_policy_update']=iteration-1
                    record['best_reward_model']=record_training_reward_best(root,sampling_checkpoint,iteration-1,iteration,record['mean_step_reward'])
                record['reward_components_sum_mean_step']=sum(record['reward_components_mean_step'].values())
                record['reward_components_units']='signed reward per sampled control transition; terminal replacement included'
                params=export_actor(algo.policy);checkpoint=root/'checkpoints'/f'update_{iteration:04d}'
                save_policy(checkpoint,params,mean,std,identity,hidden_sizes=c.hidden_sizes,activation=c.activation)
                torch.save({'policy':algo.policy.state_dict(),'optimizer':algo.optimizer.state_dict(),'identity':identity,'rsl_version':RSL_VERSION,'activation':c.activation,'hidden_sizes':c.hidden_sizes,
                    'update':iteration,'control_transitions':record['control_transitions'],'learning_rate':algo.learning_rate,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],'jax_rng':np.asarray(key)},checkpoint/'rsl_snapshot.pt')
                sampling_checkpoint=checkpoint
                metadata={'update':iteration,'trainer':'rsl_rl.algorithms.PPO','rsl_version':RSL_VERSION}
                if should_validate(c,local):
                    vs=time.monotonic();v=host(validate({'actor':params},False));record['validation']=v;metadata['validation']=v;record['validation_seconds']=time.monotonic()-vs
                    delta=np.asarray(v['episode_return'])-np.asarray(baseline['episode_return'])
                    record['paired_episode_return']={'baseline':baseline['episode_return'],'candidate':v['episode_return'],'delta':delta.tolist(),'mean_delta':float(delta.mean())}
                    rank,reason=rank_tracking_candidate(v,baseline,speed_slack=c.selection_speed_slack,nominal_slack=c.selection_nominal_slack)
                    record['selection']={'rank':rank,'reason':reason,'task_success_verified':False}
                    if rank is not None and (best_rank is None or rank<best_rank):best_rank,best=rank,str(checkpoint)
                (checkpoint/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
                if 'validation' in metadata:record['best_reward_model']=refresh_best_reward_model(root)
                record['wall_elapsed_seconds']=time.monotonic()-start
                with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                writer.write(record);print(json.dumps(record,allow_nan=False),flush=True);rows.append(record)
                status={'phase':'complete' if local==c.updates else 'training','complete':local==c.updates,'last_checkpoint':str(checkpoint),'best_checkpoint':best,'best_selection_rank':best_rank,'baseline':baseline,'selection_mode':'training_mean_step_reward' if c.training_reward_selection else 'fixed_development','control_transitions':record['control_transitions']}
                if record.get('best_reward_model'):status['best_reward_checkpoint']=record['best_reward_model']['checkpoint']
                if should_plot(c,local):plot_training(root)
                write('status.json',status);write('progress.json',{'update':iteration,'phase':status['phase'],'validation_complete':'validation' in record})
        return status
    except Exception as exc:
        write('status.json',{'phase':'error','complete':False,'error':repr(exc)})
        raise
