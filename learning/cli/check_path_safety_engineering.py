#!/usr/bin/env python3
"""One discarded 8x16 GPU rollout/update; never a model qualification run."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json,time,sys,pickle,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
p=argparse.ArgumentParser();p.add_argument('--config',default='learning/configs/path_safety_candidate_run.json');p.add_argument('--parent-checkpoint',required=True);p.add_argument('--parent-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute:raise PermissionError('engineering physics requires explicit --execute')
import jax
import jax.numpy as j
import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from sttw_control.path_command_training import load_plan,load_environment,PathBatch,make_algorithm,_start_tensorboard,_verify_reward_http
from sttw_control.path_command_fork import load_weights_only
from sttw_control.preference_best import atomic_json,digest
from sttw_control.runtime import configure_compilation_cache
from sttw_control.direct_command_training import clean_numbers,notify
plan,cfg,source=load_plan(a.config);out=Path(a.output).resolve()
if out.exists():raise FileExistsError(out)
if digest(a.parent_checkpoint)!=a.parent_sha256:raise ValueError('engineering parent checkpoint SHA mismatch')
out.mkdir(parents=True);start=time.perf_counter();status=dict(state='initializing',pid=os.getpid(),declared_policy_steps=16,declared_envs=8,declared_policy_transitions=128,declared_lower_ticks=512,completed_policy_steps=0,policy_transitions=0,lower_ticks=0,optimizer_updates=0,discarded=True)
def update(**kw):
 status.update(kw,last_update_epoch=time.time());atomic_json(out/'status.json',status)
update(stage='loading')
writer=None
try:
 if jax.default_backend()!='gpu' or not torch.cuda.is_available():raise RuntimeError('GPU engineering check requires JAX and Torch CUDA')
 torch.set_num_threads(2);seed=911101;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
 cache=configure_compilation_cache();tb=_start_tensorboard(out);writer=SummaryWriter(str(out/'tensorboard'/'engineering'))
 env,snapshot=load_environment(cfg,source,'terminal_contract_v2');batch=PathBatch(env,snapshot,cfg,seed,0)
 sample=batch.reset(j.int32(0),j.int32(0));env.set_log_template(sample.state,sample.route.path,sample.route.speed)
 # First two environments are identical complete closed-loop states for a
 # deterministic paired step within, rather than in addition to, the budget.
 ids=j.array([0,0,102,103,204,205,306,307],j.int32)
 reset=jax.jit(jax.vmap(batch.reset));episodes=reset(ids,j.zeros(8,j.int32));jax.block_until_ready(episodes)
 host_initial=jax.device_get(episodes);restored=pickle.loads(pickle.dumps(host_initial))
 serialization_equal=all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(host_initial),jax.tree.leaves(restored)))
 if not serialization_equal:raise RuntimeError('complete closed-loop serialization mismatch')
 episodes=jax.tree.map(j.asarray,restored)
 observe=jax.jit(jax.vmap(batch.observation));oa,oc,fault=observe(episodes)
 def to_torch(x,y):return TensorDict({'policy':torch.utils.dlpack.from_dlpack(x),'critic':torch.utils.dlpack.from_dlpack(y)},batch_size=[8])
 obs=to_torch(oa,oc);algo=make_algorithm(obs,16,cfg,'cuda');parent=torch.load(a.parent_checkpoint,map_location='cpu',weights_only=False);initialization=load_weights_only(algo,parent)
 advance=jax.jit(jax.vmap(batch.advance_one)).lower(episodes,j.zeros((8,2))).compile()
 compile_s=time.perf_counter()-start;update(state='running',stage='sampling',compile_seconds=compile_s)
 rewards=[];sample_start=time.perf_counter();deterministic=False
 with torch.no_grad():
  for step in range(16):
   z=algo.act(obs)
   if step==0:
    z[1].copy_(z[0]);algo.transition.actions=z.clone();algo.transition.actions_log_prob=algo.policy.get_actions_log_prob(z).detach()
   episodes,nobs,reward,flags,final,stats=advance(episodes,jax.dlpack.from_dlpack(z.detach().contiguous()));jax.block_until_ready(episodes)
   if bool(j.any(flags['engineering_fault']|nobs[2])):raise RuntimeError('engineering fault; discard rollout')
   if step==0:
    deterministic=all(np.array_equal(np.asarray(x[0]),np.asarray(x[1])) for x in jax.tree.leaves(episodes))
    if not deterministic:raise RuntimeError('complete-state deterministic paired continuation mismatch')
   ticks=int(j.sum(stats['active_ticks']));status['lower_ticks']+=ticks;status['policy_transitions']+=8
   obs=to_torch(nobs[0],nobs[1]);algo.process_env_step(obs,torch.utils.dlpack.from_dlpack(reward),torch.utils.dlpack.from_dlpack(flags['done']),{})
   rewards.append(np.asarray(reward));writer.add_scalar('train/mean_step_reward',float(np.mean(rewards[-1])),step+1);writer.flush()
   update(completed_policy_steps=step+1)
  algo.compute_returns(obs)
 sample_s=time.perf_counter()-sample_start;update(stage='optimizing');opt_start=time.perf_counter();metrics=algo.update();opt_s=time.perf_counter()-opt_start
 if metrics.get('nonfinite_stop'):raise RuntimeError('nonfinite PPO update')
 status['optimizer_updates']=1;notify('STTW path engineering stage ended','8x16 rollout/update completed; learner discarded')
 verified=_verify_reward_http(out,tb,'engineering')
 if not verified:raise RuntimeError('current engineering reward scalar not loaded over HTTP')
 receipt=dict(config_sha256=__import__('sttw_control.preference_best',fromlist=['config_hash']).config_hash(cfg),parent_checkpoint=str(Path(a.parent_checkpoint).resolve()),parent_sha256=a.parent_sha256,parent_role='archived engineering weights, not frozen formal Coverage parents',initialization=initialization,actual_batch_dimensions=[8,16],compilation_cache=str(cache),physics='unchanged existing PathBatch.advance_one',optimizer='existing DirectCommandPPO',deterministic_complete_state_pair=deterministic,complete_state_serialization_equal=serialization_equal,metrics=clean_numbers(metrics),sampling_seconds=sample_s,optimize_seconds=opt_s,compile_seconds=compile_s,tensorboard=tb,scalar_verified=verified,formal_training='not_run',learner_retained=False)
 atomic_json(out/'receipt.json',receipt)
 if status['lower_ticks']!=512 or status['policy_transitions']!=128:raise RuntimeError('engineering check did not complete declared budget')
 update(state='completed',stage='discarded',total_seconds=time.perf_counter()-start)
 notify('STTW path engineering pipeline completed',str(out));print(json.dumps({'status':status,'tensorboard_url':tb['url']},ensure_ascii=False),flush=True)
except BaseException as e:
 update(state='error',stage='discarded',error=repr(e));notify('STTW path engineering error',str(e));raise
finally:
 if writer:writer.close()
