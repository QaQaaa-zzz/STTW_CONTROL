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
from .selection import rank_candidate
from .network import ResidualActor, make_policy_identity, save_policy
from .runtime import configure_compilation_cache


def gaussian_log_prob(sample,mean,log_std):
    return -.5*jp.sum(((sample-mean)*jp.exp(-log_std))**2+2*log_std+math.log(2*math.pi),axis=-1)


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
    learning_rate: float=.0003
    gamma: float=.9995
    gae_lambda: float=.99
    clip: float=.2
    entropy_weight: float=.001
    initial_std: float=.15
    seed: int=42
    checkpoint_interval: int=8
    validation_post_seconds: float=10.
    validation_hold_seconds: float=.5
    validation_path_tolerance: float=.2
    validation_extra_tolerance: float=.05
    validation_heading_tolerance: float=.15
    validation_speed_tolerance: float=.2
    selection_speed_slack: float=.01
    selection_nominal_slack: float=.01
    validation_events: tuple | None=None
    validation_seeds: tuple=(10001,10002,10003,10004)

    def __post_init__(self):
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
        if any(not math.isfinite(x) or x<=0 for x in (self.validation_path_tolerance,self.validation_extra_tolerance,self.validation_heading_tolerance,self.validation_speed_tolerance)):
            raise ValueError('invalid validation tolerances')
        if self.validation_events:
            for event in self.validation_events:
                if not math.isfinite(event['start']+event['duration']) or event['start']<0 or event['duration']<=0:
                    raise ValueError('invalid validation event')
                if any(not math.isfinite(event.get(k,0.)) for k in ('steer_rate','force')) or event.get('waveform','constant') not in ('constant','half_sine'):
                    raise ValueError('invalid validation event amplitude/waveform')
        if not self.validation_seeds or len(set(self.validation_seeds))!=len(self.validation_seeds):
            raise ValueError('validation seeds must be nonempty and unique')


def normalization(task):
    scale=[.2,.2,1.,3.,.4,2.,2.,30.,30.,.4,3.,3.,3.,30.,10.]
    if task.observation.include_path:
        scale += [1.,1.,1.]
    if task.observation.include_priority:scale += [1.] + ([1.] if task.observation.include_attitude_risk else [])
    std=np.array(scale*task.observation.history_steps+[1.]*task.observation.history_steps,np.float32)
    return np.zeros_like(std),std


def train(task_path,output,config=TrainingConfig()):
    run_start=time.monotonic()
    cache_dir=configure_compilation_cache()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    env=RecoveryEnv(load_config(task_path),backend='mjx')
    c=config
    identity=make_policy_identity(env.bundle.identity,asdict(env.config),env.config.observation.history_steps)
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('learning/src/sttw_control').glob('*.py'))}
    declaration={'task':asdict(env.config),'training':asdict(c),'policy_identity':identity,'source_sha256':source,
                 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 'jax_compilation_cache_dir':cache_dir,
                 'budget_control_transitions':c.num_envs*c.rollout_steps*c.updates,
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
    reset=jax.vmap(env.reset)
    step=jax.vmap(env.step)
    print('Compiling MJX batched reset',flush=True)
    state=jax.jit(reset)(jax.random.split(rk,c.num_envs))
    jax.block_until_ready(state.obs)

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
            row=(state.obs,z,gaussian_log_prob(z,mu,logstd),value,nxt.reward,nv,nxt.terminated,nxt.done,jp.stack([(state.event[:,2]!=0)&(state.tick>=state.event[:,0])&(state.tick<state.event[:,1]),(state.event[:,3]!=0)&(state.tick>=state.event[:,0])&(state.tick<state.event[:,1])],axis=-1))
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
    baseline=host_metrics(validate(params,True))
    (output/'baseline_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    print('Baseline validation: '+json.dumps(baseline),flush=True)
    best_score=None; best=None; best_radial=None
    start=time.monotonic()
    for index in range(1,c.updates+1):
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
        record={'update':index,'control_transitions':index*c.num_envs*c.rollout_steps,
                'elapsed_seconds':time.monotonic()-start,'loss_metrics':host.tolist(),
                'rollout_seconds':rollout_seconds,'optimizer_seconds':time.monotonic()-update_start,
                'rollout_control_steps_per_second':c.num_envs*c.rollout_steps/rollout_seconds,
                'rollout_compile_seconds':rollout_compile_seconds,
                'optimizer_compile_seconds':optimizer_compile_seconds,
                'device_memory_stats':jax.devices()[0].memory_stats(),
                'validation_seconds':0.,'checkpoint_seconds':0.,
                'steer_disturbed_transitions':int(jp.sum(rows[8][...,0])),
                'force_disturbed_transitions':int(jp.sum(rows[8][...,1])),
                'mean_step_reward':float(jp.mean(rows[4])),'episode_ends':int(jp.sum(rows[7]))}
        if index==1 or index%c.checkpoint_interval==0 or index==c.updates:
            validation_start=time.monotonic()
            validation=host_metrics(validate(params,False))
            record['validation_seconds']=time.monotonic()-validation_start
            checkpoint_start=time.monotonic()
            record['validation']=validation
            path=checkpoint(index,params,opt_state,key,validation)
            record['checkpoint_seconds']=time.monotonic()-checkpoint_start
            score,reason=rank_candidate(validation,baseline,speed_slack=c.selection_speed_slack,nominal_slack=c.selection_nominal_slack)
            record['selection']={'rank':score,'reason':reason}
            if score is not None and (best_score is None or score<best_score):
                best_score,best=score,path
                best_radial=float(np.mean(validation['radial_rmse']))
            status={'last_checkpoint':path,'best_checkpoint':best,'best_radial_rmse':best_radial,'best_selection_rank':best_score,
                    'baseline':baseline,'control_transitions':record['control_transitions'],'complete':index==c.updates}
            (output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        record['wall_elapsed_seconds']=time.monotonic()-run_start
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    return status
