"""Bounded Gaussian PPO residuals with terminal-correct GAE and MJX rollouts.

Ratios use the stored pre-tanh sample: the bijector Jacobian cancels exactly.
Entropy is the latent Gaussian entropy surrogate. Fixed physical scaling is
saved with the Actor. Snapshots restore parameters/optimizer/RNG, not physics.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import subprocess
import time

from flax import linen as nn, serialization
import jax
import jax.numpy as jp
import numpy as np
import optax
from .env import RecoveryEnv, load_config
from .validation import make_validator
from .tracking_validation import rank_tracking_candidate
from .motion_commands import signed_reward_components,gated_action
from .selection import refresh_best_reward_model, rank_candidate, rank_command_candidate, command_improved, command_should_stop
from .network import ResidualActor, make_policy_identity, save_policy, load_policy
from .runtime import configure_compilation_cache


def gaussian_log_prob(sample,mean,log_std):
    return -.5*jp.sum(((sample-mean)*jp.exp(-log_std))**2+2*log_std+math.log(2*math.pi),axis=-1)


def gaussian_kl(old_mean,old_log_std,new_mean,new_log_std):
    """Exact KL(old || new) per observation for diagonal latent Gaussians."""
    return jp.sum(new_log_std-old_log_std+(jp.exp(2*old_log_std)+(old_mean-new_mean)**2)/(2*jp.exp(2*new_log_std))-.5,axis=-1)


def generalized_advantage(reward,value,next_value,terminated,done,gamma,gae_lambda):
    delta=reward+gamma*(1-terminated.astype(value.dtype))*next_value-value
    def step(carry,row):
        d,end=row
        carry=d+gamma*gae_lambda*(1-end.astype(value.dtype))*carry
        return carry,carry
    _,adv=jax.lax.scan(step,jp.zeros_like(value[0]),(delta,done),reverse=True)
    return adv,adv+value


class Critic(nn.Module):
    @nn.compact
    def __call__(self,x):
        for width in (256,128):
            x=nn.leaky_relu(nn.Dense(width)(x),negative_slope=.01)
        return nn.Dense(1)(x)[...,0]


@dataclass(frozen=True)
class TrainingConfig:
    num_envs: int=64
    rollout_steps: int=256
    updates: int=64
    epochs: int=4
    minibatch_size: int=2048
    resume_checkpoint: str | None=None
    selection_incumbent: str | None=None
    command_selection_scope: str="initial"
    command_validation_schedules: tuple | None=None
    plot_interval: int=1
    command_selection: bool=False
    command_patience: int=0
    command_min_updates: int=16
    command_min_delta: float=.1
    command_speed_slack: float=.05
    command_yaw_slack: float=.05
    target_kl: float | None=None
    warmup_pool_size: int=0
    warmup_steps: int=0
    learning_rate: float=.0003
    gamma: float=.9995
    gae_lambda: float=.99
    clip: float=.2
    entropy_weight: float=.001
    initial_std: float=.15
    seed: int=42
    checkpoint_interval: int=8
    validation_final_only: bool=False
    validation_updates: tuple | None=None  # stage-local indices; final always validated
    validation_post_seconds: float=10.
    validation_hold_seconds: float=.5
    validation_path_tolerance: float=.2
    validation_extra_tolerance: float=.05
    validation_heading_tolerance: float=.15
    validation_speed_tolerance: float=.2
    validation_yaw_tolerance: float=.2
    selection_speed_slack: float=.01
    selection_nominal_slack: float=.01
    validation_events: tuple | None=None
    validation_seeds: tuple=(10001,10002,10003,10004)

    def __post_init__(self):
        if self.validation_updates is not None:
            if len(set(self.validation_updates))!=len(self.validation_updates) or any(type(i) is not int or not 1<=i<=self.updates for i in self.validation_updates):
                raise ValueError('invalid explicit validation updates')
        if self.command_selection_scope not in ('initial','full_episode'):raise ValueError('invalid command selection scope')
        if type(self.plot_interval) is not int or self.plot_interval<0:raise ValueError('invalid plot interval')
        if self.command_validation_schedules is not None:
            from .motion_commands import MotionCommands
            if not self.command_validation_schedules:raise ValueError('empty command validation schedules')
            sizes={len(s) for s in self.command_validation_schedules}
            if len(sizes)!=1:raise ValueError('validation schedules need equal row counts')
            for schedule in self.command_validation_schedules:MotionCommands(fixed=schedule)
        if type(self.command_patience) is not int or self.command_patience<0 or self.command_min_updates<1 or any(not math.isfinite(x) or x<0 for x in (self.command_min_delta,self.command_speed_slack,self.command_yaw_slack)):
            raise ValueError("invalid command selection settings")
        if self.command_patience and not self.command_selection:raise ValueError("command patience requires selection")
        if self.target_kl is not None and (not math.isfinite(self.target_kl) or self.target_kl<=0):raise ValueError('invalid KL target')
        if any(type(v) is not int or v<0 for v in (self.warmup_pool_size,self.warmup_steps)) or bool(self.warmup_pool_size)!=bool(self.warmup_steps):raise ValueError('invalid warmup pool')
        for name in ('num_envs','rollout_steps','updates','epochs','minibatch_size','checkpoint_interval'):
            if not isinstance(getattr(self,name),int) or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if self.num_envs*self.rollout_steps%self.minibatch_size:
            raise ValueError('rollout must divide evenly into minibatches')
        if not (0<self.gamma<=1 and 0<self.gae_lambda<=1 and 0<self.clip<1 and self.learning_rate>0 and self.initial_std>0 and self.entropy_weight>=0):
            raise ValueError('invalid PPO parameters')
        if not (math.isfinite(self.validation_post_seconds) and math.isfinite(self.validation_hold_seconds) and self.validation_post_seconds>=self.validation_hold_seconds>0):
            raise ValueError('invalid validation observation window')
        if any(not math.isfinite(x) or x<0 for x in (self.selection_speed_slack,self.selection_nominal_slack)):
            raise ValueError('invalid selection regression allowance')
        if any(not math.isfinite(x) or x<=0 for x in (self.validation_path_tolerance,self.validation_extra_tolerance,self.validation_heading_tolerance,self.validation_speed_tolerance,self.validation_yaw_tolerance)):
            raise ValueError('invalid validation tolerances')
        if self.validation_events:
            for event in self.validation_events:
                if not math.isfinite(event['start']+event['duration']) or event['start']<0 or event['duration']<=0:
                    raise ValueError('invalid validation event')
                if any(not math.isfinite(event.get(k,0.)) for k in ('steer_rate','force','rear_torque')) or event.get('waveform','constant') not in ('constant','half_sine'):
                    raise ValueError('invalid validation event amplitude/waveform')
        if not self.validation_seeds or len(set(self.validation_seeds))!=len(self.validation_seeds):
            raise ValueError('validation seeds must be nonempty and unique')


def should_validate(config,index):
    if config.validation_updates is not None:return index==config.updates or index in config.validation_updates
    return index==config.updates or (not config.validation_final_only and (index==1 or index%config.checkpoint_interval==0))


def should_plot(config,index):
    return should_validate(config,index) or (config.plot_interval>0 and index%config.plot_interval==0)


def normalization(task):
    scale=[.2,.2,1.,3.,.4,2.,2.,30.,30.,.4,3.,3.,3.,30.,10.]
    if task.observation.include_path:
        scale += [1.,1.,1.]
    if task.observation.include_motion:scale += [2.,2.]
    if task.observation.include_tracking:scale += [1.]*5
    if task.observation.include_priority:scale += [1.] + ([1.] if task.observation.include_attitude_risk else [])
    std=np.array(scale*task.observation.history_steps+[1.]*task.observation.history_steps,np.float32)
    return np.zeros_like(std),std


def restore_training_snapshot(path,template,identity):
    path=Path(path)
    if json.loads((path/'identity.json').read_text())['identity']!=identity:
        raise ValueError('resume policy identity mismatch')
    raw=(path/'training.msgpack').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=json.loads((path/'training.json').read_text())['training_sha256']:
        raise ValueError('resume snapshot hash mismatch')
    restored=serialization.from_bytes(template,raw)
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(restored)):
        raise ValueError('nonfinite resume snapshot')
    return restored


def train(task_path,output,config=TrainingConfig()):
    from .tensorboard_logging import TrainingEvents
    with TrainingEvents(Path(output)/'tensorboard', profile='core') as events:
        return _train(task_path,output,config,events)


def _train(task_path,output,config,events):
    run_start=time.monotonic();setup_timings={}
    cache_dir=configure_compilation_cache()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    env=RecoveryEnv(load_config(task_path),backend='mjx')
    c=config
    from .tensorboard_logging import validation_labels
    event_labels=validation_labels({'task':asdict(env.config),'training':asdict(c)})
    identity=make_policy_identity(env.bundle.identity,asdict(env.config),env.config.observation.history_steps)
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('learning/src/sttw_control').glob('*.py'))}
    declaration={'task':asdict(env.config),'training':asdict(c),'policy_identity':identity,'source_sha256':source,
                 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 'jax_compilation_cache_dir':cache_dir,
                 'budget_control_transitions':c.num_envs*c.rollout_steps*c.updates,
                 'planned_optimizer_steps_per_update':c.epochs*c.num_envs*c.rollout_steps//c.minibatch_size,
                 'warmup_compute_transition_budget':c.warmup_pool_size*c.warmup_steps,
                 'scheduled_seconds_per_env':c.rollout_steps*c.updates*env.config.controller.dt,
                 'scheduled_episode_equivalents_per_env':c.rollout_steps*c.updates/env.horizon,
                 'coverage_note':'Scheduled time is divided across resets; it does not guarantee completed recovery episodes.',
                 'validation_role':'development selection, not held-out recovery evidence',
                 'validation_peak_window':'event onset through observation end',
                 'snapshot_resume':'optimizer/parameters/RNG only; no exact physics continuation'}
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    mean,std=normalization(env.config)
    scale=jp.asarray(std)
    actor,critic=ResidualActor(),Critic()
    key=jax.random.PRNGKey(c.seed)
    key,ak,vk,rk=jax.random.split(key,4)
    ap=actor.init(ak,jp.zeros(env.observation_size))
    # Zero deterministic residual at initialization; hidden layers retain diversity.
    ap['params']['Dense_2']=jax.tree.map(jp.zeros_like,ap['params']['Dense_2'])
    params={'actor':ap,'critic':critic.init(vk,jp.zeros(env.observation_size)),
            'log_std':jp.full((2,),math.log(c.initial_std))}
    optimizer=optax.chain(optax.clip_by_global_norm(.5),optax.adam(c.learning_rate))
    opt_state=optimizer.init(params)
    update_offset=0; resume_key=None
    template={'params':params,'optimizer':opt_state,'rng':key,'update':0}
    if c.resume_checkpoint:
        parent=Path(c.resume_checkpoint).resolve()
        load_policy(parent,expected=identity)
        meta=json.loads((parent/'identity.json').read_text())
        if not np.array_equal(np.asarray(meta['mean']),mean) or not np.array_equal(np.asarray(meta['std']),std):
            raise ValueError('resume normalization mismatch')
        parent_config=json.loads((parent.parents[1]/'declaration.json').read_text())['training']
        for field in ('num_envs','rollout_steps'):
            if parent_config[field]!=getattr(c,field):raise ValueError('resume batch geometry changed')
        restored=restore_training_snapshot(parent,template,identity)
        params,opt_state,resume_key=restored['params'],restored['optimizer'],restored['rng']
        saved_actor=serialization.from_bytes(params['actor'],(parent/'actor.msgpack').read_bytes())
        if not all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree.leaves(saved_actor),jax.tree.leaves(params['actor']))):
            raise ValueError('snapshot actor differs from policy export')
        update_offset=int(restored['update'])
        declaration['continuation']={'checkpoint':str(parent),'training_sha256':json.loads((parent/'training.json').read_text())['training_sha256'],'update_offset':update_offset,'scope':'parameters, optimizer and PPO RNG restored; physics, ESO, histories reset and zero-residual warmup repeated'}
        (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    reset=jax.vmap(env.reset)
    step=jax.vmap(env.step)
    reset_start=time.monotonic()
    print('Compiling MJX batched reset',flush=True)
    state=jax.jit(reset)(jax.random.split(rk,c.num_envs))
    jax.block_until_ready(state.obs)
    setup_timings['reset_seconds']=time.monotonic()-reset_start
    warmup_start=time.monotonic()

    warmup_record={'computed_transition_budget':c.warmup_pool_size*c.warmup_steps,'active_transitions':0}
    if c.warmup_steps:
        print('Compiling physical phase-pool warmup',flush=True)
        key,bk,ik=jax.random.split(key,3)
        bank=jax.jit(reset)(jax.random.split(bk,c.warmup_pool_size))
        targets=jp.arange(c.warmup_pool_size)*c.warmup_steps//c.warmup_pool_size
        @jax.jit
        def warmup(bank,key):
            def tick(carry,i):
                bank,key=carry;key,rk=jax.random.split(key)
                nxt=step(bank,jp.zeros((c.warmup_pool_size,2)))
                def restart(nxt):
                    fresh=reset(jax.random.split(rk,c.warmup_pool_size))
                    return jax.tree.map(lambda a,b:jp.where(nxt.done.reshape((c.warmup_pool_size,)+(1,)*(a.ndim-1)),b,a),nxt,fresh)
                nxt=jax.lax.cond(jp.any(nxt.done),restart,lambda x:x,nxt)
                bank=jax.tree.map(lambda a,b:jp.where((i<targets).reshape((c.warmup_pool_size,)+(1,)*(a.ndim-1)),b,a),bank,nxt)
                return (bank,key),None
            return jax.lax.scan(tick,(bank,key),jp.arange(c.warmup_steps))[0]
        bank,key=warmup(bank,key);jax.block_until_ready(bank.obs)
        indices=jax.random.randint(ik,(c.num_envs,),0,c.warmup_pool_size)
        state=jax.tree.map(lambda x:x[indices],bank)
        warmup_record.update(active_transitions=int(jp.sum(targets)),pool_size=c.warmup_pool_size,tick_quantiles=np.quantile(np.asarray(bank.tick),[0,.25,.5,.75,1]).tolist(),scope='Real zero-residual trajectories; bank states sampled with replacement, not independent initial histories')
        print('Warmup: '+json.dumps(warmup_record),flush=True)
    if resume_key is not None:key=resume_key
    setup_timings['warmup_seconds']=time.monotonic()-warmup_start
    (output/'warmup.json').write_text(json.dumps(warmup_record,indent=2)+'\n')

    def outputs(p,obs):
        x=obs/scale
        return actor.apply(p['actor'],x,return_logits=True),critic.apply(p['critic'],x)

    @jax.jit
    def rollout(state,key,p):
        def tick(carry,_):
            state,key=carry
            key,noise_key,reset_key=jax.random.split(key,3)
            mu,value=outputs(p,state.obs)
            logstd=jp.clip(p['log_std'],-4.,0.)
            z=mu+jp.exp(logstd)*jax.random.normal(noise_key,mu.shape)
            nxt=step(state,jp.tanh(z))
            _,nv=outputs(p,nxt.obs)
            row=(state.obs,z,gaussian_log_prob(z,mu,logstd),value,nxt.reward,nv,nxt.terminated,nxt.done,jp.stack([(state.event[:,2]!=0)&(state.tick>=state.event[:,0])&(state.tick<state.event[:,1]),(state.event[:,3]!=0)&(state.tick>=state.event[:,0])&(state.tick<state.event[:,1]),(state.event[:,5]!=0)&(state.tick>=state.event[:,0])&(state.tick<state.event[:,1])],axis=-1))
            if env.config.motion_commands is not None:
                cfg=env.config
                raw=jax.vmap(env.requested)(state.tick,state.command_schedule)
                speed=jp.sum(nxt.data.qvel[:,:3]*nxt.data.xmat[:,env.bundle.chassis,:,0],axis=-1)
                parts=jax.vmap(lambda roll,rate,ev,ey,action,alpha,failed:signed_reward_components(roll,rate,ev,ey,action,alpha,cfg.motion_commands,cfg.controller.dt,cfg.alive_reward_rate,cfg.failure_penalty,failed))(nxt.measurement[:,0],nxt.measurement[:,1],speed-raw[:,0],nxt.yaw_rate-raw[:,1],gated_action(jp.tanh(z),state.priority_alpha,cfg.motion_commands),state.priority_alpha,nxt.terminated)
                # Aggregate on-device before scan storage: no per-environment log transfer.
                audit={k:jp.mean(v) for k,v in parts.items()}
                audit['reconstruction_max_abs']=jp.max(jp.abs(sum(parts.values())-nxt.reward))
                audit['reconstruction_max_scaled']=jp.max(jp.abs(sum(parts.values())-nxt.reward)/(1+jp.abs(nxt.reward)))
                row=row+(audit,)
            if env.config.tracking_reward is not None:
                parts=nxt.reward_parts
                audit={k:jp.mean(v) for k,v in parts.items()}
                audit['reconstruction_max_abs']=jp.max(jp.abs(sum(parts.values())-nxt.reward))
                audit['reconstruction_max_scaled']=jp.max(jp.abs(sum(parts.values())-nxt.reward)/(1+jp.abs(nxt.reward)))
                row=row+(audit,)
            def restart(s):
                fresh=reset(jax.random.split(reset_key,c.num_envs))
                return jax.tree.map(lambda a,b:jp.where(s.done.reshape((c.num_envs,)+(1,)*(a.ndim-1)),b,a),s,fresh)
            nxt=jax.lax.cond(jp.any(nxt.done),restart,lambda s:s,nxt)
            return (nxt,key),row
        return jax.lax.scan(tick,(state,key),None,length=c.rollout_steps)

    def loss(p,batch):
        obs,z,oldlog,adv,target=batch
        mu,value=outputs(p,obs)
        logstd=jp.clip(p['log_std'],-4.,0.)
        logprob=gaussian_log_prob(z,mu,logstd)
        ratio=jp.exp(logprob-oldlog)
        policy=-jp.mean(jp.minimum(ratio*adv,jp.clip(ratio,1-c.clip,1+c.clip)*adv))
        value_loss=.5*jp.mean((value-target)**2)
        entropy=jp.sum(logstd+.5*math.log(2*math.pi*math.e))
        total=policy+value_loss-c.entropy_weight*entropy
        return total,jp.array([policy,value_loss,entropy,jp.mean((ratio-1)-(logprob-oldlog))])

    @jax.jit
    def update(p,opt_state,key,rows):
        obs,z,logprob,value,reward,nv,terminated,done=rows[:8]
        adv,target=generalized_advantage(reward,value,nv,terminated,done,c.gamma,c.gae_lambda)
        adv=(adv-jp.mean(adv))/(jp.std(adv)+1e-8)
        flat=jax.tree.map(lambda x:x.reshape((-1,)+x.shape[2:]),(obs,z,logprob,adv,target))
        count=c.num_envs*c.rollout_steps
        if c.target_kl is not None:
            old_p=p;old_opt=opt_state
            old_std=jp.clip(old_p['log_std'],-4.,0.)
            def kl_on(candidate,observations):
                old_mu,_=outputs(old_p,observations);new_mu,_=outputs(candidate,observations)
                return jp.mean(gaussian_kl(old_mu,old_std,new_mu,jp.clip(candidate['log_std'],-4.,0.)))
            def constrained_epoch(carry,_):
                p,opt_state,key,stopped=carry;key,pk=jax.random.split(key)
                indices=jax.random.permutation(pk,count).reshape((-1,c.minibatch_size))
                def minibatch(carry,idx):
                    p,opt_state,stopped=carry
                    def attempt(_):
                        batch=jax.tree.map(lambda x:x[idx],flat)
                        (_,metrics),grad=jax.value_and_grad(loss,has_aux=True)(p,batch)
                        delta,new_opt=optimizer.update(grad,opt_state,p);candidate=optax.apply_updates(p,delta)
                        kl=kl_on(candidate,batch[0]);finite=jp.isfinite(kl)&jp.all(jp.isfinite(metrics))
                        accept=finite&(kl<=c.target_kl)
                        selected=jax.lax.cond(accept,lambda _:(candidate,new_opt),lambda _:(p,opt_state),None)
                        return (*selected,~accept),jp.concatenate([metrics,jp.array([1.,accept.astype(jp.float32),jp.nan_to_num(kl,nan=1e30,posinf=1e30)])])
                    return jax.lax.cond(stopped,lambda _:((p,opt_state,stopped),jp.zeros(7)),attempt,None)
                (p,opt_state,stopped),metrics=jax.lax.scan(minibatch,(p,opt_state,stopped),indices)
                return (p,opt_state,key,stopped),jp.sum(metrics,axis=0)
            (candidate,new_opt,key,stopped),metrics=jax.lax.scan(constrained_epoch,(p,opt_state,key,jp.bool_(False)),None,length=c.epochs)
            # Check ALL collected observations in bounded chunks; rollback params AND optimizer.
            def final_chunk(_,obs):return None,kl_on(candidate,obs)
            _,kls=jax.lax.scan(final_chunk,None,flat[0].reshape((-1,c.minibatch_size,flat[0].shape[-1])))
            final_kl=jp.mean(kls);accept=jp.isfinite(final_kl)&(final_kl<=c.target_kl)
            p,opt_state=jax.lax.cond(accept,lambda _:(candidate,new_opt),lambda _:(old_p,old_opt),None)
            totals=jp.sum(metrics,axis=0);means=totals[:4]/jp.maximum(totals[4],1)
            audit=jp.array([totals[4],totals[5],jp.where(accept,final_kl,0.),~accept])
            return p,opt_state,key,jp.concatenate([means,audit])
        def epoch(carry,_):
            p,opt_state,key=carry
            key,permkey=jax.random.split(key)
            indices=jax.random.permutation(permkey,count).reshape((-1,c.minibatch_size))
            def minibatch(carry,idx):
                p,opt_state=carry
                (_,metrics),grad=jax.value_and_grad(loss,has_aux=True)(p,jax.tree.map(lambda x:x[idx],flat))
                updates,opt_state=optimizer.update(grad,opt_state,p)
                return (optax.apply_updates(p,updates),opt_state),metrics
            (p,opt_state),metrics=jax.lax.scan(minibatch,(p,opt_state),indices)
            return (p,opt_state,key),jp.mean(metrics,axis=0)
        (p,opt_state,key),metrics=jax.lax.scan(epoch,(p,opt_state,key),None,length=c.epochs)
        return p,opt_state,key,jp.mean(metrics,axis=0)

    validate=make_validator(env,actor,scale,c)

    def host_metrics(raw):
        # Preserve invalid validation evidence as JSON null; ranking rejects it.
        return {k:np.where(np.isfinite(np.asarray(v)),np.asarray(v),None).tolist() for k,v in raw.items()}

    def checkpoint(index,p,opt_state,key,validation):
        path=output/'checkpoints'/f'update_{index:04d}'
        save_policy(path,p['actor'],mean,std,identity)
        snapshot=serialization.to_bytes({'params':p,'optimizer':opt_state,'rng':key,'update':index})
        (path/'training.msgpack').write_bytes(snapshot)
        (path/'training.json').write_text(json.dumps({'update':index,'validation':validation,'training_sha256':hashlib.sha256(snapshot).hexdigest()},indent=2)+'\n')
        return str(path)

    print(f'Compiling {len(c.validation_seeds)}-seed deterministic baseline validation',flush=True)
    baseline_start=time.monotonic()
    baseline=host_metrics(validate(params,True))
    setup_timings['baseline_validation_seconds']=time.monotonic()-baseline_start
    (output/'baseline_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    print('Baseline validation: '+json.dumps(baseline),flush=True)
    best_score=None; best=None; best_radial=None; stale=0
    stage_best=None;stage_best_score=None
    initial_start=time.monotonic()
    if c.command_selection and env.config.motion_commands is not None:
        initial=host_metrics(validate(params,False)) if c.resume_checkpoint else baseline
        (output/'initial_policy_validation.json').write_text(json.dumps(initial,indent=2)+'\n')
        for source in dict.fromkeys((c.selection_incumbent,c.resume_checkpoint)):
            if source is None:continue
            prior=restore_training_snapshot(source,template,identity)
            measured=initial if source==c.resume_checkpoint else host_metrics(validate(prior['params'],False))
            score,reason=rank_command_candidate(measured,baseline,speed_slack=c.command_speed_slack,yaw_slack=c.command_yaw_slack,scope=c.command_selection_scope)
            if command_improved(score,best_score,c.command_min_delta):best_score,best=score,str(Path(source).resolve())
            with (output/'incumbents.jsonl').open('a') as f:f.write(json.dumps({'checkpoint':str(source),'rank':score,'validation':measured})+'\n')
    setup_timings['initial_and_incumbent_seconds']=time.monotonic()-initial_start
    setup_timings['total_initialization_seconds']=time.monotonic()-run_start
    setup_timings['fresh_initial_reuses_zero_residual_baseline']=not bool(c.resume_checkpoint)
    (output/'setup_timings.json').write_text(json.dumps(setup_timings,indent=2)+'\n')
    print('Setup timings: '+json.dumps(setup_timings),flush=True)
    start=time.monotonic()
    for index in range(1,c.updates+1):
        stop=False
        rollout_compile_seconds=optimizer_compile_seconds=0.
        if index==1:
            compile_start=time.monotonic()
            rollout_executable=rollout.lower(state,key,params).compile()
            rollout_compile_seconds=time.monotonic()-compile_start
        rollout_start=time.monotonic()
        (state,key),rows=rollout_executable(state,key,params)
        jax.block_until_ready(rows[4])
        rollout_seconds=time.monotonic()-rollout_start
        if index==1:
            compile_start=time.monotonic()
            update_executable=update.lower(params,opt_state,key,rows).compile()
            optimizer_compile_seconds=time.monotonic()-compile_start
        update_start=time.monotonic()
        params,opt_state,key,metrics=update_executable(params,opt_state,key,rows)
        host=np.asarray(metrics)
        if not np.isfinite(host).all() or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
            raise RuntimeError('Nonfinite PPO update; stopping without promotion')
        record={'update':index+update_offset,'stage_update':index,'stage_control_transitions':index*c.num_envs*c.rollout_steps,'control_transitions':(index+update_offset)*c.num_envs*c.rollout_steps,
                'elapsed_seconds':time.monotonic()-start,'loss_metrics':host[:4].tolist(),
                'rollout_seconds':rollout_seconds,'optimizer_seconds':time.monotonic()-update_start,
                'rollout_control_steps_per_second':c.num_envs*c.rollout_steps/rollout_seconds,
                'rollout_compile_seconds':rollout_compile_seconds,
                'optimizer_compile_seconds':optimizer_compile_seconds,
                'device_memory_stats':jax.devices()[0].memory_stats(),
                'validation_seconds':0.,'checkpoint_seconds':0.,
                'steer_disturbed_transitions':int(jp.sum(rows[8][...,0])),
                'force_disturbed_transitions':int(jp.sum(rows[8][...,1])),'rear_disturbed_transitions':int(jp.sum(rows[8][...,2])),
                'mean_step_reward':float(jp.mean(rows[4])),'episode_ends':int(jp.sum(rows[7]))}
        if c.target_kl is not None:
            record['optimizer_audit']={'attempted_minibatches':int(host[4]),'accepted_minibatches':int(host[5]),'final_exact_kl':float(host[6]),'full_update_rolled_back':bool(host[7]),'target_kl':c.target_kl}
        if env.config.motion_commands is not None or env.config.tracking_reward is not None:
            component_means={k:float(jp.mean(v)) for k,v in rows[9].items() if not k.startswith('reconstruction_')}
            error=float(jp.max(rows[9]['reconstruction_max_abs']))
            scaled_error=float(jp.max(rows[9]['reconstruction_max_scaled']))
            record['reward_components_reconstruction_max_scaled']=scaled_error
            record['reward_components_mean_step']=component_means
            record['reward_components_sum_mean_step']=sum(component_means.values())
            record['reward_components_reconstruction_max_abs']=error
            record['reward_components_units']='signed reward per sampled control transition; terminal reward replaces other terms'
            if not all(math.isfinite(v) for v in component_means.values()) or not math.isfinite(error) or not math.isfinite(scaled_error) or scaled_error>3e-5:
                raise RuntimeError(f'Reward component reconstruction mismatch: {error}')
        checkpoint_start=time.monotonic()
        path=checkpoint(index+update_offset,params,opt_state,key,None)
        record['checkpoint_seconds']=time.monotonic()-checkpoint_start
        (output/'progress.json').write_text(json.dumps({'update':index+update_offset,'stage_update':index,'control_transitions':record['control_transitions'],'last_checkpoint':path,'phase':'validating' if should_validate(c,index) else 'training','validation_complete':False},indent=2)+'\n')
        if should_validate(c,index):
            validation_start=time.monotonic()
            validation=host_metrics(validate(params,False))
            record['validation_seconds']=time.monotonic()-validation_start
            record['validation']=validation
            if env.config.motion_commands is not None or env.config.tracking_reward is not None:
                delta=np.asarray(validation['episode_return'],float)-np.asarray(baseline['episode_return'],float)
                record['paired_episode_return']={'baseline':baseline['episode_return'],'candidate':validation['episode_return'],'delta':delta.tolist(),'mean_delta':float(np.mean(delta)),'scope':'same declared horizon, commands, alpha and initial seed; terminal replacement and actual early termination retained'}
            metadata_path=Path(path)/'training.json'
            metadata=json.loads(metadata_path.read_text());metadata['validation']=validation
            metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
            if env.config.motion_commands is not None:
                record['best_reward_model']=refresh_best_reward_model(output)
            if env.config.tracking_reward is not None:
                score,reason=rank_tracking_candidate(validation,baseline,speed_slack=c.selection_speed_slack,path_slack=c.selection_nominal_slack)
            else:
                score,reason=(None,"command task: predeclared final endpoint; no path recovery ranking") if env.config.motion_commands is not None else rank_candidate(validation,baseline,speed_slack=c.selection_speed_slack,nominal_slack=c.selection_nominal_slack)
            command_mode=env.config.motion_commands is not None and c.command_selection
            if command_mode:score,reason=rank_command_candidate(validation,baseline,speed_slack=c.command_speed_slack,yaw_slack=c.command_yaw_slack,scope=c.command_selection_scope)
            record['selection']={'rank':score,'reason':reason,'task_success_verified':False}
            if env.config.tracking_reward is not None:
                record['selection']['development_gates_passed']=bool(score is not None and not any(score[:4]))
            if command_mode and c.command_selection_scope=='full_episode':record['selection']['development_gates_passed']=bool(score is not None and all(v==0 for v in score[:-1]))
            improved=command_improved(score,best_score,c.command_min_delta) if command_mode else score is not None and (best_score is None or score<best_score)
            if improved:
                best_score,best=score,path
                best_radial=None if command_mode else float(np.mean(validation['radial_rmse']))
            if command_mode:
                if command_improved(score,stage_best_score,c.command_min_delta):stage_best_score,stage_best=score,path
                stale=0 if improved else stale+1
                stop=command_should_stop(index,stale,c.command_min_updates,c.command_patience)
                record['command_selection']={'stale_evaluations':stale,'early_stop':stop,'best_checkpoint':best}
            status={'last_checkpoint':path,'best_checkpoint':best,'stage_best_checkpoint':stage_best,'stage_best_rank':stage_best_score,'stage_control_transitions':record['stage_control_transitions'],'best_radial_rmse':best_radial,'best_selection_rank':best_score,
                    'baseline':baseline,'control_transitions':record['control_transitions'],'complete':index==c.updates or stop,'stop_reason':'development_patience' if stop else ('budget' if index==c.updates else None)}
            if record.get('best_reward_model'):
                status['best_reward_checkpoint']=record['best_reward_model']['checkpoint']
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        record['wall_elapsed_seconds']=time.monotonic()-run_start
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        events.write(record,event_labels)
        print(json.dumps(record,allow_nan=False),flush=True)
        (output/'progress.json').write_text(json.dumps({'update':index+update_offset,'stage_update':index,'control_transitions':record['control_transitions'],'last_checkpoint':path,'phase':'complete' if index==c.updates or stop else 'training','validation_complete':'validation' in record},indent=2)+'\n')
        from .training_diagnostics import plot_training
        if should_plot(c,index):
            plot_start=time.monotonic();plot_training(output)
            with (output/'plot_timings.jsonl').open('a') as f:f.write(json.dumps({'update':record['update'],'seconds':time.monotonic()-plot_start})+'\n')
        if stop:break
    return status
