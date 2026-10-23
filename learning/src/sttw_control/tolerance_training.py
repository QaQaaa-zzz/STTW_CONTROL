"""Explicit matched model-reserve experiment; old exact-resume contract unchanged."""
from copy import deepcopy
from pathlib import Path
import json
from .preference_best import config_hash,digest,atomic_json
from .path_command_fork import IDENTITY_KEYS,_audit_saved_parent
from .tolerance_shaping import validate_tolerance_config

SPEC=dict(weight=10.,gamma=.815560907125473,effort_budget=.030839085298575167,duration=.1,target_force=6.)

def experiment_config(parent,mode,reward_variant="historical"):
 if mode not in ('baseline','static','dynamic'):raise ValueError('unknown experiment arm')
 c=deepcopy(parent)
 if reward_variant=='persistent_v3':c['path_reward']=json.loads((Path(__file__).parents[2]/'configs/STTW_Tolerance_V3_Reward.json').read_text())
 elif reward_variant!='historical':raise ValueError('unknown reward variant')
 c['training_future_phase_C'].update(updates_initial=50,updates_max=50,validation_updates=[],save_every=5)
 if mode!='baseline':
  r=c['path_reward'];r['tolerance']=dict(method=mode,**SPEC);r['component_caps']['tolerance']=10.;r['failure_cost_rate_bound']+=10.
 return c

def validate_training_config(config,parent,mode,reward_variant="historical"):
 if config_hash(config)!=config_hash(experiment_config(parent,mode,reward_variant)):raise ValueError('configuration differs outside declared reward and schedule fork')
 validate_tolerance_config(config)

def restore_weights_only(algo,record):
 import torch
 if algo.optimizer.state:raise ValueError('weights-only fork requires fresh optimizer')
 identity={k:record[k] for k in IDENTITY_KEYS}
 for k in ('parent_source_update','parent_checkpoint_sha256'):
  if k in record:identity[k]=record[k]
 saved=_audit_saved_parent(record,identity)
 algo.policy.load_state_dict(saved['policy'],strict=True)
 for k,v in saved['policy'].items():
  if not torch.equal(algo.policy.state_dict()[k].detach().cpu(),v.detach().cpu()):raise ValueError('copied policy differs from parent')
 if algo.optimizer.state:raise ValueError('optimizer unexpectedly changed')
 return dict(parent=record,original_identity=identity,restored=['actor','critic','log_std'],optimizer='fresh empty Adam state',rng='new declared training seed; parent RNG not restored',new_update=0)

def pulse_training_config(parent):
 c=deepcopy(parent)
 if c['path_reward'].get('tolerance'):raise ValueError('pulse training requires unshaped baseline reward')
 c['training_future_phase_C'].update(updates_initial=25,updates_max=25,validation_updates=[],save_every=5)
 return c

def restore_pulse_parent(algo,record,source,alpha):
 import torch
 if algo.optimizer.state:raise ValueError('fresh optimizer required')
 if digest(record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('pulse parent checksum mismatch')
 saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
 if saved['identity']!=record['identity'] or saved['update']!=50:raise ValueError('pulse parent identity mismatch')
 identity=saved['identity']
 if identity['arm']!='baseline' or identity['alpha']!=alpha or identity['lower_actor_sha256']!=source['candidate']['actor_sha256']:raise ValueError('wrong pulse parent')
 algo.policy.load_state_dict(saved['policy'],strict=True)
 for k,v in saved['policy'].items():
  if not torch.equal(algo.policy.state_dict()[k].detach().cpu(),v.detach().cpu()):raise ValueError('pulse weights copy differs')
 return dict(parent=record,restored=['actor','critic','log_std'],optimizer='fresh empty Adam',new_update=0)

def run(plan_path,output):
 import os,time,random,pickle,subprocess
 import numpy as np
 import torch
 import jax
 import jax.numpy as j
 from tensordict import TensorDict
 from torch.utils.tensorboard import SummaryWriter
 from .path_command_training import PathBatch,load_environment,make_algorithm,save_checkpoint,_verify_reward_http
 from .path_command_policy import SCHEMA
 from .path_command_statistics import reduce_statistics,merge_statistics,summarize_statistics
 from .path_command_reward import configured_cost_keys
 from .direct_command_training import clean_numbers
 from .lower_command_training import handle_candidate_ppo_result
 from .runtime import configure_compilation_cache
 plan=json.loads(Path(plan_path).read_text());root=Path(output);root.mkdir(parents=True,exist_ok=True)
 if (root/'manifest.json').exists():raise FileExistsError('immutable training attempt exists')
 pulse_mode=plan.get('experiment')=='actual_pulse_curriculum_v1';updates=25 if pulse_mode else 50
 status=dict(state='initializing',phase='running',pid=os.getpid(),arm=plan['arm'],alpha=plan['alpha'],completed_updates=0,declared_updates=updates,policy_transitions_total=0,training_control_ticks_total=0,declared_policy_transitions=updates*512*128)
 def update(**kw):
  status.update(kw);status['checked_unix']=time.time();atomic_json(root/'status.json',status)
 update(stage='loading');writer=None
 try:
  parent=json.loads(Path(plan['parent_manifest']).read_text());cfg=json.loads(Path(plan['config']).read_text());source=parent['source']
  if pulse_mode:
   record=plan['parent_record']
   if record['identity']!=parent['identity'] or record['identity']['config_sha256']!=config_hash(parent['config']):raise ValueError('parent manifest identity differs')
   if plan['arm'] not in ('uniform','guided') or plan['pulse_seed']!=910402:raise ValueError('invalid pulse protocol')
   if config_hash(cfg)!=config_hash(pulse_training_config(parent['config'])):raise ValueError('pulse config differs from baseline task')
  else:
   pkg=json.loads(Path(plan['package']).read_text());record=pkg['model_selection'][f"alpha{plan['alpha']}"]
   validate_training_config(cfg,parent['config'],plan['arm'],plan.get('reward_variant','historical'))
   if source['candidate']['actor_sha256']!=record['lower_actor_sha256']:raise ValueError('lower identity mismatch')
  if plan['updates']!=updates or plan['seed']!=(910401 if pulse_mode else 910301) or plan['alpha'] not in (0,1):raise ValueError('experiment plan differs from declared protocol')
  optimizer_guard=json.loads(Path(parent['plan']['parent_manifest']).read_text())['plan'] if pulse_mode else parent['plan']
  if not all(k in optimizer_guard for k in ('actor_lr_floor','hard_reject_consecutive_stop')):raise ValueError('missing optimizer guard contract')
  torch.set_num_threads(2);seed=plan['seed']+plan['alpha'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
  cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache');c=cfg['training_future_phase_C'];n=c['num_envs'];steps=c['rollout_steps']
  if (n,steps)!=(512,128):raise ValueError('declared batch mismatch')
  env,snapshot=load_environment(cfg,source,'terminal_contract_v2');batch=PathBatch(env,snapshot,cfg,plan['seed'],plan['alpha'])
  if pulse_mode:
   from .tolerance_pulse import PulseBatch
   batch=PulseBatch(batch,guided=plan['arm']=='guided',pulse_seed=plan['pulse_seed'],reserve_config=experiment_config(cfg,'dynamic'))
  sample=batch.reset(j.int32(0),j.int32(0));env.set_log_template(sample.state,sample.route.path,sample.route.speed)
  reset=jax.jit(jax.vmap(batch.reset));episodes=reset(j.arange(n,dtype=j.int32),j.zeros(n,j.int32));jax.block_until_ready(episodes)
  def to_torch(obs):return TensorDict({'policy':torch.utils.dlpack.from_dlpack(obs[0]),'critic':torch.utils.dlpack.from_dlpack(obs[1])},batch_size=[n])
  initial=jax.jit(jax.vmap(batch.observation))(episodes)
  if bool(j.any(initial[2])):raise RuntimeError('initial observation fault')
  obs=to_torch(initial);algo=make_algorithm(obs,steps,cfg,'cuda');receipt=restore_pulse_parent(algo,record,source,plan['alpha']) if pulse_mode else restore_weights_only(algo,record);atomic_json(root/'fork_receipt.json',receipt)
  identity=dict(schema=SCHEMA,alpha=plan['alpha'],lower_actor_sha256=source['candidate']['actor_sha256'],config_sha256=config_hash(cfg),seed=plan['seed'],parent_checkpoint_sha256=record['checkpoint_sha256'],parent_source_update=50 if pulse_mode else 200,experiment='actual_pulse_curriculum_v1' if pulse_mode else 'model_reserve_shaping_v1',arm=plan['arm'],initialization='weights_only_fresh_optimizer')
  atomic_json(root/'manifest.json',dict(plan=plan,config=cfg,source=source,parent_manifest_sha256=digest(plan['parent_manifest']),identity=identity,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution_optimizations=dict(batch=n,rollout_steps=steps,cache=str(cache),ppo='DirectCommandPPO',physics='existing PathBatch vmap'),selection=f'fixed final update{updates}; no intermediate selection',evaluation='separate matched development panel after campaign'))
  def advance_batch(ep,z):
   end,ob,r,fl,final,stats=jax.vmap(batch.advance_one)(ep,z);stats['diagnostics']=reduce_statistics(stats['diagnostics']);return end,ob,r,fl,stats
  update(stage='compiling');advance=jax.jit(advance_batch).lower(episodes,j.zeros((n,2))).compile()
  tb=json.loads(Path(plan['tensorboard_info']).read_text());writer=SummaryWriter(str(Path(tb['logdir'])/plan['run_name']));verified=False;rejections=0
  cp=save_checkpoint(root/'checkpoints',algo,0,identity,route_counters=episodes.episode);atomic_json(root/'last_completed.json',dict(update=0,checkpoint=str(cp.resolve()),checkpoint_sha256=digest(cp)))
  for iteration in range(1,updates+1):
   update(state='running',stage='sampling',completed_updates=iteration-1,rollout_intervals=0);start=time.monotonic();rewards=[];diag=None;physical=domain=0;pulse_summary=dict(fired=0,fallback=0,completed=0,enabled_completed=0,force_histogram=[0]*8,max_completed_impulse_error_Ns=0.)
   with torch.no_grad():
    for index in range(steps):
     z=algo.act(obs)
     if not bool(torch.isfinite(z).all()):raise RuntimeError('nonfinite action')
     episodes,next_obs,reward,flags,stats=advance(episodes,jax.dlpack.from_dlpack(z.detach().contiguous()));jax.block_until_ready(episodes)
     status['policy_transitions_total']+=n;status['training_control_ticks_total']+=int(j.sum(stats['active_ticks']))
     if bool(j.any(flags['engineering_fault'])) or bool(j.any(next_obs[2])):
      with (root/'engineering_fault_states.pkl').open('wb') as f:pickle.dump(jax.device_get(episodes),f)
      raise RuntimeError('engineering fault; incomplete rollout discarded')
     obs=to_torch(next_obs);algo.process_env_step(obs,torch.utils.dlpack.from_dlpack(reward),torch.utils.dlpack.from_dlpack(flags['done']),{})
     rewards.append(np.asarray(reward));physical+=int(j.sum(flags['physical_failure']));domain+=int(j.sum(flags['tracking_domain_failure']));x=jax.device_get(stats['diagnostics']);diag=x if diag is None else merge_statistics(diag,x)
     if pulse_mode:
      p=jax.device_get(stats['pulse']);fired=np.asarray(p['fired_now']);pulse_summary['fired']+=int(fired.sum());pulse_summary['fallback']+=int(np.sum(p['fallback_now']));pulse_summary['completed']+=int(np.sum(p['completed']));pulse_summary['enabled_completed']+=int(np.sum(p['completed']&p['enabled']))
      for fi,force in enumerate([-8,-6,-4,-2,2,4,6,8]):pulse_summary['force_histogram'][fi]+=int(np.sum(fired&(p['force']==force)))
      pulse_summary['max_completed_impulse_error_Ns']=max(pulse_summary['max_completed_impulse_error_Ns'],float(np.max(np.abs(p['completed_impulse']-p['completed_expected_impulse']))))
     if (index+1)%16==0:update(rollout_intervals=index+1)
    algo.compute_returns(obs)
   sample_s=time.monotonic()-start;update(stage='optimizing');start=time.monotonic();metrics=algo.update();opt_s=time.monotonic()-start
   rejections,stop=handle_candidate_ppo_result(algo,metrics,rejections,optimizer_guard);reward=float(np.mean(rewards));diagnostic=summarize_statistics(diag,cfg)
   row=dict(update=iteration,sampling_model_update=iteration-1,mean_step_reward=reward,physical_failures=physical,domain_failures=domain,optimizer=clean_numbers(metrics),sample_seconds=sample_s,optimize_seconds=opt_s,policy_transitions_total=status['policy_transitions_total'],training_control_ticks_total=status['training_control_ticks_total'],diagnostics=diagnostic)
   if pulse_mode:row['pulse']=pulse_summary
   with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   writer.add_scalar('train/mean_step_reward',reward,iteration);writer.add_scalar('train/physical_failures',physical,iteration);writer.add_scalar('train/domain_failures',domain,iteration)
   for key in ['surrogate','value','entropy','mean_kl','accepted_epochs']:writer.add_scalar('ppo/'+key,float(metrics[key]),iteration)
   for k,value in zip(configured_cost_keys(cfg),diag['cost_effective_sum']):writer.add_scalar('cost_rate/'+k,float(value/max(float(diag['count']),1)),iteration)
   if pulse_mode:
    for key in ['fired','fallback','completed','enabled_completed','max_completed_impulse_error_Ns']:writer.add_scalar('pulse/'+key,pulse_summary[key],iteration)
   writer.flush()
   if iteration%5==0 or stop:
    cp=save_checkpoint(root/'checkpoints',algo,iteration,identity,rejections=rejections,route_counters=episodes.episode);atomic_json(root/'last_completed.json',dict(update=iteration,checkpoint=str(cp.resolve()),checkpoint_sha256=digest(cp)))
   if not verified and (iteration==1 or iteration%5==0):verified=_verify_reward_http(root,tb,plan['run_name'])
   update(completed_updates=iteration,stage='optimized',last_reward=reward,tensorboard_reward_verified=verified)
   if stop:raise RuntimeError('optimizer stop: '+stop)
  update(state='completed',phase='completed',stage='training_complete',evaluation_status='not_started')
 except BaseException as e:update(state='error',phase='error',error=f'{type(e).__name__}: {e}');raise
 finally:
  if writer is not None:writer.close()
