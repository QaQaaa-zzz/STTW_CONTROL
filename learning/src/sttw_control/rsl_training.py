"""RSL-RL PPO on MJX, with latent Gaussian actions and identity-bound Flax export.

The upstream optimizer/GAE remain RSL implementations. Only timeout bootstrap is
adapted: use the final next-state critic before reset, never the previous value.
"""
from pathlib import Path
from dataclasses import asdict
import importlib.metadata
import json
import os
import time
import copy
import math
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


def make_algorithm(obs,steps,epochs,minibatches,device,learning_rate=3e-4,gamma=.9995,lam=.99,clip=.2,entropy=.001,std=.15,kl=.01,activation="leaky_relu",hidden_sizes=(256,128),schedule="auto"):
    if importlib.metadata.version('rsl-rl-lib')!=RSL_VERSION:
        raise RuntimeError(f'This adapter requires rsl-rl-lib=={RSL_VERSION}')
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},2,
        actor_hidden_dims=list(hidden_sizes),critic_hidden_dims=list(hidden_sizes),activation='elu' if activation=='elu' else 'lrelu',
        init_noise_std=std,noise_std_type='log',actor_obs_normalization=False,critic_obs_normalization=False).to(device)
    linear=[m for m in policy.actor.modules() if isinstance(m,torch.nn.Linear)][-1]
    torch.nn.init.zeros_(linear.weight);torch.nn.init.zeros_(linear.bias)
    storage=RolloutStorage('rl',obs.batch_size[0],steps,obs,[2],device=device)
    if schedule not in ('auto','fixed','adaptive'):raise ValueError('invalid RSL schedule')
    chosen_schedule=('adaptive' if kl else 'fixed') if schedule=='auto' else schedule
    return PPO(policy,storage,num_learning_epochs=epochs,num_mini_batches=minibatches,
        clip_param=clip,gamma=gamma,lam=lam,value_loss_coef=.5,entropy_coef=entropy,
        learning_rate=learning_rate,max_grad_norm=.5,use_clipped_value_loss=False,
        schedule=chosen_schedule,desired_kl=kl,device=device)


def export_actor(policy):
    """Transpose Torch weights into the existing inference-only Flax schema."""
    layers=[m for m in policy.actor.modules() if isinstance(m,torch.nn.Linear)]
    if len(layers)<2 or layers[-1].out_features!=2:raise ValueError('Actor architecture differs from export contract')
    return {'params':{f'Dense_{i}':{'kernel':jp.asarray(m.weight.detach().cpu().numpy().T),
                                    'bias':jp.asarray(m.bias.detach().cpu().numpy())} for i,m in enumerate(layers)}}


def restore_cuda_rng(states):
    """torch.load(map_location=cuda) also moves RNG bytes; CUDA expects CPU bytes."""
    torch.cuda.set_rng_state_all([state.cpu() for state in states])



def guarded_update(algo, limit=None):
    """Audit exact Gaussian KL on ALL sampled observations; optionally rollback.

    The snapshot includes optimizer moments, not just weights. Candidate losses
    are diagnostic when rejected. This bounds a sampled mean policy change, NOT
    physical risk or a guarantee for unobserved states. RSL still computes PPO/GAE.
    """
    if limit is not None and (not math.isfinite(limit) or limit<=0):
        raise ValueError('invalid update KL limit')
    snapshot=(copy.deepcopy(algo.policy.state_dict()),copy.deepcopy(algo.optimizer.state_dict()),
              algo.learning_rate) if limit is not None else None
    metrics=algo.update()
    with torch.no_grad():
        algo.policy.log_std.clamp_(-4.,0.)
        stored=algo.storage.observations.flatten(0,1)
        old_mu=algo.storage.mu.flatten(0,1);old_sigma=algo.storage.sigma.flatten(0,1)
        # Deterministic distribution evaluation: diagnostics do not consume noise RNG.
        new_mu=algo.policy.act_inference(stored)
        new_sigma=algo.policy.log_std.exp().expand_as(new_mu)
        kl_rows=torch.sum(torch.log(new_sigma/old_sigma)+(old_sigma**2+(old_mu-new_mu)**2)/(2*new_sigma**2)-.5,dim=-1)
        kl=float(kl_rows.mean());tail=float(torch.quantile(kl_rows,.95))
        finite=math.isfinite(kl) and all(math.isfinite(float(v)) for v in metrics.values())
        finite=finite and all(bool(torch.isfinite(p).all()) for p in algo.policy.parameters())
    reject=not finite or (limit is not None and kl>limit)
    if reject and snapshot is None:
        raise RuntimeError('nonfinite RSL update; no finite checkpoint published')
    if reject:
        algo.policy.load_state_dict(snapshot[0]);algo.optimizer.load_state_dict(snapshot[1]);algo.learning_rate=snapshot[2]
    def safe(x):return float(x) if math.isfinite(float(x)) else None
    audit={'candidate_exact_kl':safe(kl),'candidate_kl_p95':safe(tail),
           'final_exact_kl':0. if reject else float(kl),
           'full_update_rolled_back':bool(reject),'kl_limit':limit,
           'learning_rate_schedule':algo.schedule,'candidate_nonfinite':not finite}
    with torch.no_grad():
        mu=algo.policy.act_inference(stored)
        mean_action=torch.tanh(mu)
        audit.update(mean_action_abs=float(mean_action.abs().mean()),
                     mean_action_saturation_fraction=float((mean_action.abs()>.95).float().mean()),
                     steer_latent_std=float(algo.policy.log_std[0].exp()),
                     rear_latent_std=float(algo.policy.log_std[1].exp()))
    return {k:safe(v) for k,v in metrics.items()},audit


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
            'kl_contract':{'schedule':c.rsl_schedule,'whole_update_exact_kl_limit':c.rsl_kl_limit,
                           'scope':'RSL PPO with optional full-batch mean Gaussian KL rollback; not safety certification'},
            'baseline_contract':{'original_ecbc_eso_base_output_scale':1.0,
                'learning_zero_residual_ablation_scale':cfg.actuator.base_output_scale,
                'prepared_state_scale':cfg.preparation_base_output_scale,
                'selection_baseline':'original ECBC+ESO at scale 1.0; zero-residual learning environment is reported separately'},
            'resume_contract':'optimizer/policy/RNG restored; fresh physics and truthful phase warmup',
            'source_revision':os.environ.get('STTW_SOURCE_REVISION'),
            'source_provenance':'launch Git revision; retain model/config/checkpoint identity, omit bulk source hashing'}
        write('declaration.json',declaration);write('status.json',{'phase':'initializing','complete':False})
        step=jax.vmap(env.step)
        key=jax.random.PRNGKey(c.seed);key,rk=jax.random.split(key)
        preparation_count=0
        if cfg.preparation_seconds:
            if c.warmup_steps:raise ValueError('task preparation and legacy phase warmup cannot be combined')
            prepare=jax.jit(jax.vmap(env.prepare_state))
            prepared=prepare(jax.random.split(rk,c.num_envs));jax.block_until_ready(prepared.obs)
            if bool(jp.any(prepared.preparation_failed)):raise RuntimeError('closed-loop preparation failed')
            preparation_count=c.num_envs*int(round(cfg.preparation_seconds/cfg.controller.dt))
            reset_from_prepared=jax.jit(jax.vmap(env.reset_from_prepared))
            def reset(keys):return reset_from_prepared(prepared,keys)
            key,rk=jax.random.split(key);state=reset(jax.random.split(rk,c.num_envs))
        else:
            reset=jax.jit(jax.vmap(env.reset))
            state=reset(jax.random.split(rk,c.num_envs))
        jax.block_until_ready(state.obs)
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
        write('warmup.json',{'legacy_phase_computed_transition_budget':c.warmup_pool_size*c.warmup_steps,
            'legacy_phase_active_transitions':warmup_count,
            'closed_loop_preparation_computed_transitions':preparation_count,
            'closed_loop_preparation_reused_for_episode_resets':bool(cfg.preparation_seconds),
            'training_control_transitions_exclude_preparation':True})
        scale=torch.tensor(std,device=device)
        def observations(value):return TensorDict({'policy':torch_from_jax(value)/scale},batch_size=[c.num_envs])
        obs=observations(state.obs)
        algo=make_algorithm(obs,c.rollout_steps,c.epochs,c.num_envs*c.rollout_steps//c.minibatch_size,device,
                            c.learning_rate,c.gamma,c.gae_lambda,c.clip,c.entropy_weight,c.initial_std,c.target_kl,c.activation,c.hidden_sizes,schedule=c.rsl_schedule)
        offset=0;transition_offset=0
        if c.resume_checkpoint:
            snapshot=torch.load(Path(c.resume_checkpoint)/'rsl_snapshot.pt',map_location=device,weights_only=False)
            if snapshot['identity']!=identity or snapshot['rsl_version']!=RSL_VERSION or snapshot.get('activation','leaky_relu')!=c.activation or tuple(snapshot.get('hidden_sizes',(256,128)))!=c.hidden_sizes:raise ValueError('RSL resume identity mismatch')
            algo.policy.load_state_dict(snapshot['policy']);algo.optimizer.load_state_dict(snapshot['optimizer'])
            torch.set_rng_state(snapshot['torch_rng'].cpu())
            if device.startswith('cuda'):restore_cuda_rng(snapshot['cuda_rng'])
            key=jp.asarray(snapshot['jax_rng']);offset=snapshot['update'];transition_offset=snapshot['control_transitions'];algo.learning_rate=snapshot['learning_rate']
            if c.rsl_schedule=='fixed':
                algo.learning_rate=c.learning_rate
                for group in algo.optimizer.param_groups:group['lr']=c.learning_rate
        from .rsl_sampling import phase_spread,EpisodeStatistics
        if c.phase_spread_initialization:
            write('status.json',{'phase':'phase_spreading','complete':False})
            phase_begin=time.monotonic()
            actor=ResidualActor(hidden_sizes=c.hidden_sizes,activation=c.activation)
            params=export_actor(algo.policy)
            horizon_steps=int(round(cfg.horizon_seconds/cfg.controller.dt))
            spread=jax.jit(lambda s,k:phase_spread(s,k,c.num_envs,horizon_steps,step,reset,
                lambda x:actor.apply(params,x.obs/jp.asarray(std))))
            state,key,phase_stats=spread(state,key);jax.block_until_ready(state.obs)
            ticks=np.asarray(state.tick)
            write('phase_spread.json',{'active_transitions':int(phase_stats['active_transitions']),
                'computed_transitions':int(phase_stats['computed_transitions']),
                'reset_count':int(phase_stats['reset_count']),
                'elapsed_seconds':time.monotonic()-phase_begin,
                'actual_tick_min':int(ticks.min()),'actual_tick_max':int(ticks.max()),
                'actual_phase_counts':np.histogram(ticks*cfg.controller.dt,bins=[0,1,3,6,cfg.horizon_seconds])[0].tolist(),
                'excluded_from_training_budget':True,'policy':'deterministic current actor; complete closed-loop state retained'})
            obs=observations(state.obs)
        episode_stats=EpisodeStatistics(torch_from_jax(state.tick==0))
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
                                'lateral_m':nxt.geometric_features[:,0] if cfg.tracking.geometric else features[:,0]}
                if cfg.timed_reference.mode=='geometry':
                    sampled_errors['heading_rad']=features[:,1]
                else:
                    sampled_errors.update(yaw_rate_rad_s=nxt.yaw_rate-state.reference_command[:,1],
                                          longitudinal_m=features[:,3])
                alpha_stats=alpha_sample_sums(state.priority_alpha,nxt.reward,nxt.terminated,sampled_errors,xp=jp,choices=getattr(cfg.priority,'training_alphas',None))
            phase=state.tick*cfg.controller.dt
            phase_id=jp.sum(phase[:,None]>=jp.array([1.,3.,6.]),axis=1)
            phase_stats={f'bin_{i}_samples':jp.sum(phase_id==i) for i in range(4)}
            phase_stats['elapsed_sum']=jp.sum(phase)
            phase_stats['physical_failures']=jp.sum(nxt.terminated)
            if nxt.tracking_raw_costs is not None:
                for name,value in nxt.tracking_raw_costs.items():phase_stats['raw_cost_'+name]=jp.sum(value)
                total=sum(nxt.tracking_raw_costs.values())
                phase_stats['soft_bound_slope_sum']=jp.sum((cfg.tracking.soft_cost_bound/(cfg.tracking.soft_cost_bound+total))**2)
            if cfg.tracking.precision_reward:
                ordinary=sum(v for k,v in nxt.tracking_components.items()
                             if k not in ('deadline','failure','recovery'))
                floor=cfg.controller.dt*cfg.tracking.precision_reward_scale*(cfg.alive_reward_rate-cfg.tracking.precision_cost_cap)
                phase_stats['cost_capped_samples']=jp.sum((~nxt.terminated)&(ordinary<=floor+1e-7))
            phase_stats['disturbed_samples']=jp.sum(jp.any(state.event[:,[2,3,5]]!=0,axis=1)&(state.tick>=state.event[:,0])&(state.tick<state.event[:,1]))
            for i in range(4):
                mask=phase_id==i
                phase_stats[f'bin_{i}_reward_sum']=jp.sum(jp.where(mask,nxt.reward,0.))
                if alpha_stats:
                    phase_stats[f'bin_{i}_speed_squared_sum']=jp.sum(jp.where(mask,sampled_errors['speed_m_s']**2,0.))
                    phase_stats[f'bin_{i}_lateral_squared_sum']=jp.sum(jp.where(mask,sampled_errors['lateral_m']**2,0.))
            live=jax.lax.cond(jp.any(nxt.done),restart,lambda s:s,nxt)
            return live,key,nxt.obs,nxt.reward,nxt.done,nxt.truncated,nxt.terminated,parts,error,alpha_stats,phase_stats
        host=lambda x:jax.tree.map(lambda v:np.asarray(v).tolist(),x)
        baseline=None;ablation=None;validate=None
        if not c.training_reward_selection:
            actor=ResidualActor(hidden_sizes=c.hidden_sizes,activation=c.activation)
            validate=make_validator(env,actor,jp.asarray(std),c)
            initial_params={'actor':export_actor(algo.policy)}
            ablation=host(validate(initial_params,True,jp.nan))
            baseline=host(validate(initial_params,True,1.))
            write('zero_residual_0p8_validation.json',ablation)
            write('baseline_validation.json',baseline)
        sampling_checkpoint=Path(c.resume_checkpoint).resolve() if c.resume_checkpoint else root/'checkpoints'/f'update_{offset:04d}'
        if c.training_reward_selection and not c.resume_checkpoint:
            save_policy(sampling_checkpoint,export_actor(algo.policy),mean,std,identity,hidden_sizes=c.hidden_sizes,activation=c.activation)
            (sampling_checkpoint/'training.json').write_text(json.dumps({'update':offset,'role':'initial sampling policy'})+'\n')
        write('setup_timings.json',{'initialization_and_baseline_seconds':time.monotonic()-start})
        best=None;best_rank=None;rows=[]
        with TrainingEvents(root/'tensorboard',profile='core') as writer:
            for local in range(1,c.updates+1):
                iteration=offset+local;begin=time.monotonic();reward_sum=torch.zeros((),device=device);component_sum={};alpha_sum={};phase_sum={};ends=torch.zeros((),device=device);max_error=0.
                with torch.no_grad():
                    for _ in range(c.rollout_steps):
                        latent=algo.act(obs)
                        state,key,final_obs,reward,done,truncated,terminated,parts,error,alpha_stats,phase_stats=advance(state,jax_from_torch(torch.tanh(latent)),key)
                        final=observations(final_obs);r=torch_from_jax(reward);d=torch_from_jax(done)
                        episode_stats.add(r,d,torch_from_jax(terminated))
                        corrected=bootstrap_timeout(r,algo.policy.evaluate(final).squeeze(-1),torch_from_jax(truncated),torch_from_jax(terminated),c.gamma)
                        obs=observations(state.obs)
                        algo.process_env_step(obs,corrected,d,{})
                        reward_sum+=r.mean();ends+=d.sum()
                        for k,v in alpha_stats.items():alpha_sum[k]=alpha_sum.get(k,0)+torch_from_jax(v)
                        for k,v in phase_stats.items():phase_sum[k]=phase_sum.get(k,0)+torch_from_jax(v)
                        for k,v in parts.items():component_sum[k]=component_sum.get(k,0)+torch_from_jax(v)
                        max_error=torch.maximum(torch.as_tensor(max_error,device=device),torch_from_jax(error))
                    algo.compute_returns(obs)
                if device.startswith('cuda'):torch.cuda.synchronize()
                rollout_seconds=time.monotonic()-begin;optim_start=time.monotonic()
                # RSL update clears the storage cursor but retains arrays for audit.
                metrics,update_audit=guarded_update(algo,c.rsl_kl_limit)
                kl=update_audit['candidate_exact_kl']
                if device.startswith('cuda'):torch.cuda.synchronize()
                optim_seconds=time.monotonic()-optim_start
                error=float(max_error[0])
                if not np.isfinite(error) or error>3e-5:raise RuntimeError(f'RSL reward reconstruction mismatch: {error}')
                record={'update':iteration,'stage_update':local,'mean_step_reward':reward_sum.item()/c.rollout_steps,
                    'control_transitions':transition_offset+local*c.num_envs*c.rollout_steps,'stage_control_transitions':local*c.num_envs*c.rollout_steps,
                    'episode_ends':int(ends.item()),'loss_metrics':[metrics['surrogate'],metrics['value'],metrics['entropy'],kl],
                    'learning_rate':algo.learning_rate,'rollout_seconds':rollout_seconds,'optimizer_seconds':optim_seconds,
                    'optimizer_audit':{**update_audit,'attempted_minibatches':c.epochs*(c.num_envs*c.rollout_steps//c.minibatch_size),
                        'accepted_minibatches':0 if update_audit['full_update_rolled_back'] else c.epochs*(c.num_envs*c.rollout_steps//c.minibatch_size)},
                    'reward_components_mean_step':{k:v.item()/c.rollout_steps for k,v in component_sum.items()},'reward_components_reconstruction_max_scaled':error,'reward_components_reconstruction_max_abs':float(max_error[1])}
                batch_count=c.num_envs*c.rollout_steps
                if cfg.tracking.objective=='soft_budget_v1':
                    record['raw_costs_mean_rate']={k[len('raw_cost_'):]:float(v)/batch_count for k,v in phase_sum.items() if k.startswith('raw_cost_')}
                    record['soft_bound_slope_mean']=float(phase_sum['soft_bound_slope_sum'])/batch_count
                if cfg.tracking.precision_reward:
                    record['precision_cost_cap_fraction']=float(phase_sum['cost_capped_samples'])/batch_count
                record['complete_training_episodes']=episode_stats.flush()
                record['phase_tracking']={}
                for i in range(4):
                    count=float(phase_sum[f'bin_{i}_samples'])
                    if count:
                        record['phase_tracking'][f'phase_{i}_mean_reward']=float(phase_sum[f'bin_{i}_reward_sum'])/count
                        for field in ('speed','lateral'):
                            k=f'bin_{i}_{field}_squared_sum'
                            if k in phase_sum:record['phase_tracking'][f'phase_{i}_{field}_rmse']=float((phase_sum[k]/count).sqrt())
                record['sample_phase']={f'fraction_{i}':float(phase_sum[f'bin_{i}_samples'])/batch_count for i in range(4)}
                record['sample_phase'].update(mean_elapsed_seconds=float(phase_sum['elapsed_sum'])/batch_count,
                    physical_failures=int(phase_sum['physical_failures']),
                    disturbed_fraction=float(phase_sum['disturbed_samples'])/batch_count)
                record['sample_phase_scope']='actual pre-step elapsed seconds: [0,1), [1,3), [3,6), [6,horizon]; not policy-matched validation'
                record['rollout_control_steps_per_second']=batch_count/rollout_seconds
                if alpha_sum:
                    record['alpha_training_samples']=alpha_sample_summary({k:v.cpu().numpy() for k,v in alpha_sum.items()},choices=getattr(cfg.priority,'training_alphas',None))
                    record['alpha_training_scope']='random training samples in three alpha intervals; not paired fixed-alpha evaluation; errors use pre-step commands and post-step physics'
                if getattr(cfg.priority,'training_alphas',None) is not None:
                    record['alpha_training_scope']='exact discrete alpha training groups; not paired fixed-scenario evaluation'
                if c.training_reward_selection:
                    record['sampling_checkpoint']=str(sampling_checkpoint)
                    record['sampling_policy_update']=iteration-1
                    if c.training_reward_best_enabled:
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
                status={'phase':'complete' if local==c.updates else 'training','complete':local==c.updates,'last_checkpoint':str(checkpoint),'best_checkpoint':best,'best_selection_rank':best_rank,'baseline':baseline,'zero_residual_0p8_ablation':ablation,'selection_mode':'training_mean_step_reward' if c.training_reward_selection else 'fixed_development','control_transitions':record['control_transitions']}
                if c.training_reward_selection and not c.training_reward_best_enabled:
                    status['selection_mode']='deferred_fixed_complete_episode_evaluation'
                if record.get('best_reward_model'):status['best_reward_checkpoint']=record['best_reward_model']['checkpoint']
                if should_plot(c,local):plot_training(root)
                write('status.json',status);write('progress.json',{'update':iteration,'phase':status['phase'],'validation_complete':'validation' in record})
        return status
    except Exception as exc:
        write('status.json',{'phase':'error','complete':False,'error':repr(exc)})
        raise
