"""Original H75 PPO factory, shared by JAX and Isaac collectors."""
def make_algorithm(obs,steps,config,device='cuda'):
    import torch
    from rsl_rl.modules import ActorCritic
    from rsl_rl.storage import RolloutStorage
    from .direct_command_ppo import DirectCommandPPO,RSL_VERSION
    import importlib.metadata
    if importlib.metadata.version('rsl-rl-lib')!=RSL_VERSION:raise RuntimeError('RSL version mismatch')
    if obs['policy'].shape[-1]!=351 or obs['critic'].shape[-1]!=352:raise ValueError('geometric schema351/352 required')
    c=config['training_future_phase_C'];n=config['network']
    if n['actor_dim']!=351 or n['critic_dim']!=352 or n['hidden']!=[128,128,64] or n['activation']!='elu':raise ValueError('path network mismatch')
    if steps*obs.batch_size[0]%c['minibatches']:raise ValueError('rollout must divide minibatches')
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['critic']},2,actor_hidden_dims=n['hidden'],critic_hidden_dims=n['hidden'],activation='elu',init_noise_std=c['std_initial'][0],noise_std_type='log',state_dependent_std=False,actor_obs_normalization=False,critic_obs_normalization=False).to(device)
    from .torch_path_policy import make_torch_path_network
    if config['network'].get('architecture')=='alpha_modulated_v1':
        policy.actor=make_torch_path_network(351,2,config).to(device)
        policy.critic=make_torch_path_network(352,1,config).to(device)
    policy.path_network_config=config
    with torch.no_grad():
        for net in (policy.actor,policy.critic):
            layers=[x for x in net.modules() if isinstance(x,torch.nn.Linear)]
            for layer in layers:torch.nn.init.orthogonal_(layer.weight,2**.5);torch.nn.init.zeros_(layer.bias)
            if net is policy.actor:torch.nn.init.zeros_(layers[-1].weight)
            else:torch.nn.init.orthogonal_(layers[-1].weight,1.)
        policy.log_std.copy_(torch.tensor(c['std_initial'],device=device).log())
    storage=RolloutStorage('rl',obs.batch_size[0],steps,obs,[2],device=device)
    # Independent endpoint run: one KL group, avoiding missing-opposite-alpha NaN.
    algorithm=DirectCommandPPO
    if config.get('alpha_conditioning',{}).get('mode')=='shared_conditional_v1':
        class ConditionalPathPPO(DirectCommandPPO):
            _full_rollout_kl=conditional_rollout_kl
        algorithm=ConditionalPathPPO
    return algorithm(policy,storage,num_learning_epochs=c['epochs'],num_mini_batches=c['minibatches'],clip_param=c['clip'],gamma=c['gamma'],lam=c['gae_lambda'],value_loss_coef=1.,entropy_coef=c['entropy'],use_clipped_value_loss=False,schedule='fixed',desired_kl=c['soft_kl'],normalize_advantage_per_mini_batch=False,device=device,actor_lr=c['actor_lr'],critic_lr=c['critic_lr'],adam_betas=(.9,.999),adam_epsilon=1e-5,weight_decay=0.,std_min=c['std_min'],std_max=c['std_max'],hard_kl=c['hard_kl'],actor_grad_limit=c['grad_clip_actor'],critic_grad_limit=c['grad_clip_critic'],kl_group_index=None)


def conditional_rollout_kl(algo):
    """Path-only KL diagnostics; the stop rule remains full-rollout mean KL."""
    import torch
    from .direct_command_ppo import _finite_tensors
    with torch.no_grad():
        observations=algo.storage.observations.flatten(0,1)
        old_mu=algo.storage.mu.flatten(0,1);old_std=algo.storage.sigma.flatten(0,1)
        alpha=observations['policy'][:,336];std=algo.policy.log_std.exp()
        if not _finite_tensors(old_mu,old_std,std,alpha) or bool((old_std<=0).any()):
            return float('nan'),{}
        totals=torch.zeros(3,device=old_mu.device,dtype=torch.float64)
        counts=torch.zeros_like(totals)
        for start in range(0,len(old_mu),16384):
            end=min(start+16384,len(old_mu));new_mu=algo.policy.act_inference(observations[start:end])
            kl=(torch.log(std/old_std[start:end])+(old_std[start:end].square()+(old_mu[start:end]-new_mu).square())/(2*std.square())-.5).sum(-1)
            if not _finite_tensors(kl):return float('nan'),{}
            totals[0]+=kl.sum();counts[0]+=len(kl)
            for i,value in enumerate((0.,1.),1):
                mask=alpha[start:end]==value
                totals[i]+=torch.where(mask,kl,0.).sum();counts[i]+=mask.sum()
        totals,counts=torch.stack((totals,counts)).cpu().tolist()
        return totals[0]/counts[0],{str(i):totals[i+1]/counts[i+1] for i in (0,1) if counts[i+1]}


