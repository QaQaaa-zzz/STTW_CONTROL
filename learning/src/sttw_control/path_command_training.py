"""351/352 geometric task bindings to the existing RSL PPO implementation."""
from pathlib import Path
import json,os,pickle,random
import numpy as np
import jax
import jax.numpy as j
from flax import struct
from .path_command_policy import SCHEMA
from .preference_best import atomic_json,config_hash,digest


def termination_flags(tick,physical_failure,fault,domain_exit,horizon,config,terminal_contract='terminal_contract_v2'):
    physical_failure=j.asarray(physical_failure,dtype=j.bool_)
    domain=j.asarray(domain_exit,dtype=j.bool_)
    engineering=j.asarray(fault,dtype=j.bool_)|((terminal_contract!='terminal_contract_v2')&domain)
    tracking=domain&~engineering
    finite_end=(tick>=round(horizon/config['timing']['lower_dt_s']))&~physical_failure&~engineering&~tracking
    done=physical_failure|engineering|tracking|finite_end
    end_code=j.where(engineering,4,j.where(physical_failure,1,j.where(tracking,2,j.where(finite_end,3,0))))
    return dict(physical_failure=physical_failure,tracking_domain_failure=tracking,engineering_fault=engineering,domain_exit=domain,finite_task_end=finite_end,done=done,bootstrap=~done,end_code=end_code)


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
    with torch.no_grad():
        for net in (policy.actor,policy.critic):
            layers=[x for x in net.modules() if isinstance(x,torch.nn.Linear)]
            for layer in layers:torch.nn.init.orthogonal_(layer.weight,2**.5);torch.nn.init.zeros_(layer.bias)
            if net is policy.actor:torch.nn.init.zeros_(layers[-1].weight)
            else:torch.nn.init.orthogonal_(layers[-1].weight,1.)
        policy.log_std.copy_(torch.tensor(c['std_initial'],device=device).log())
    storage=RolloutStorage('rl',obs.batch_size[0],steps,obs,[2],device=device)
    # Independent endpoint run: one KL group, avoiding missing-opposite-alpha NaN.
    return DirectCommandPPO(policy,storage,num_learning_epochs=c['epochs'],num_mini_batches=c['minibatches'],clip_param=c['clip'],gamma=c['gamma'],lam=c['gae_lambda'],value_loss_coef=1.,entropy_coef=c['entropy'],use_clipped_value_loss=False,schedule='fixed',desired_kl=c['soft_kl'],normalize_advantage_per_mini_batch=False,device=device,actor_lr=c['actor_lr'],critic_lr=c['critic_lr'],adam_betas=(.9,.999),adam_epsilon=1e-5,weight_decay=0.,std_min=c['std_min'],std_max=c['std_max'],hard_kl=c['hard_kl'],actor_grad_limit=c['grad_clip_actor'],critic_grad_limit=c['grad_clip_critic'],kl_group_index=None)


def export_actor(policy):
    import torch
    layers=[x for x in policy.actor.modules() if isinstance(x,torch.nn.Linear)]
    if [(x.in_features,x.out_features) for x in layers]!=[(351,128),(128,128),(128,64),(64,2)]:raise ValueError('path Actor architecture mismatch')
    return {'params':{f'Dense_{i}':{'kernel':j.asarray(x.weight.detach().cpu().numpy().T),'bias':j.asarray(x.bias.detach().cpu().numpy())} for i,x in enumerate(layers)}}


def save_checkpoint(root,algo,update,identity,*,rejections=0,route_counters=None):
    import torch
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/f'update_{update:04d}.pt'
    if path.exists():raise FileExistsError(path)
    saved=dict(schema=SCHEMA,identity=identity,update=update,route_counters=None if route_counters is None else np.asarray(route_counters),policy=algo.policy.state_dict(),optimizer=algo.optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,numpy_rng=np.random.get_state(),python_rng=random.getstate(),consecutive_rejections=rejections,terminal_contract='terminal_contract_v2',environment_continuation='learner exact; physical/controller/history/filter restart from original prepared bank, not frame-continuous')
    temp=path.with_suffix('.tmp');torch.save(saved,temp);os.replace(temp,path)
    with (root/f'actor_{update:04d}.pkl').open('wb') as f:pickle.dump(jax.device_get(export_actor(algo.policy)),f)
    return path


def restore_checkpoint(path,algo,identity):
    import torch
    saved=torch.load(path,map_location=algo.device,weights_only=False)
    if saved.get('schema')!=SCHEMA or saved.get('identity')!=identity:raise ValueError('checkpoint task/lower/config identity mismatch')
    algo.policy.load_state_dict(saved['policy']);algo.optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    if saved['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    return saved

def resume_source(parent,alpha,identity):
    """Resolve an immutable saved boundary, never a guessed checkpoint filename."""
    out=Path(parent)/f'alpha{alpha}'
    receipt=json.loads((out/'last_completed.json').read_text())
    if receipt['schema']!=SCHEMA or digest(receipt['checkpoint'])!=receipt['checkpoint_sha256']:
        raise ValueError('resume boundary identity mismatch')
    best=json.loads((out/'best_model.json').read_text()) if (out/'best_model.json').exists() else None
    if best:
        if any(best.get(k)!=v for k,v in identity.items()):raise ValueError('parent best task identity mismatch')
        for key in ['checkpoint','actor']:
            if digest(best[key])!=best[key+'_sha256']:raise ValueError('parent best file changed')
        if best['source_update']>receipt['update']:raise ValueError('best newer than resume boundary')
    return receipt,best

def reset_finished_episode(ended,flags,reset):
    return jax.lax.cond(flags['done']&~flags['engineering_fault'],lambda _:reset(ended.env_id,ended.episode+1),lambda _:ended,None)

@struct.dataclass
class Episode:
    state: object
    route: object
    env_id: object
    episode: object
    totals: object
    component_totals: object

class PathBatch:
    """Auto-reset only valid terminal episodes; faults stop the owning trainer."""
    def __init__(self,env,snapshot,config,seed,alpha):
        self.env=env;self.snapshot=snapshot;self.config=config;self.seed=seed;self.alpha=alpha
        self.horizon=config['training_future_phase_C']['episode_s']
    def reset(self,eid,episode):
        from .path_command_scenarios import sample_route,episode_key
        pose=self.env.physics.helpers.pose(self.snapshot.data)
        route=sample_route(episode_key(self.seed,eid,episode),pose,self.config)
        state=self.env.reset(self.snapshot,route.path,route.speed,j.asarray(self.alpha))
        return Episode(state,route,eid,episode,j.zeros(6),j.zeros(10))
    def observation(self,ep):return self.env.observation(ep.state,ep.route.path,ep.route.speed,self.horizon)
    def advance_one(self,ep,z):
        end,logs=self.env.policy_step(ep.state,z,ep.route.path,ep.route.speed,self.horizon)
        final=self.env.observation(end,ep.route.path,ep.route.speed,self.horizon)
        flags=termination_flags(end.tick,end.physical.failed,end.fault|final[2],end.domain_exit,self.horizon,self.config,self.env.terminal_contract)
        active=logs['active_tick'];ev=logs['actual_forward_speed']-ep.route.speed
        sums=j.array([j.sum(active),j.sum(j.where(active,ev**2,0)),j.sum(j.where(active,logs['path_cross_track']**2,0)),j.sum(j.where(active,logs['path_heading_error']**2,0)),j.sum(logs['scored_tick_reward']),j.max(logs['peak_roll'])])
        totals=ep.totals+sums;totals=totals.at[5].set(j.maximum(ep.totals[5],sums[5]))
        from .path_command_statistics import step_statistics
        diagnostics=step_statistics(logs,ep.route,self.config)
        cost_totals=ep.component_totals+diagnostics['cost_effective_sum']*self.config['timing']['lower_dt_s']
        ended=ep.replace(state=end,totals=totals,component_totals=cost_totals)
        nxt=reset_finished_episode(ended,flags,self.reset)
        obs=self.observation(nxt)
        return nxt,obs,j.sum(logs['scored_tick_reward']),flags,final,dict(diagnostics=diagnostics,completed_component_costs=j.where(flags['done'],cost_totals,0.),rollout=sums,completed_totals=j.where(flags['done'],totals,0.),family=ep.route.family,episode=ep.episode,active_ticks=j.sum(active),component_caps={k:j.sum(v&active) for k,v in logs['component_capped'].items()})


def load_plan(path):
    path=Path(path).resolve();plan=json.loads(path.read_text());cfgpath=(path.parent/plan['design_config']).resolve();sourcepath=(path.parent/plan['integration_manifest']).resolve()
    cfg=json.loads(cfgpath.read_text());source=json.loads(sourcepath.read_text());c=cfg['training_future_phase_C']
    if plan['schema']!='sttw_path_training_plan_v1' or cfg['network']['actor_dim']!=351:raise ValueError('wrong task schema')
    if plan['alphas']!=[0,1] or not 1<=plan['updates']<=c['updates_initial'] or plan['automatic_extension']:raise ValueError('unapproved endpoint/budget extension')
    if source['candidate']['source_update']!=300:raise ValueError('frozen300 required')
    # These are explicit resolved details; reject silent changes to code constants.
    expected={'ordinary_fraction':.3,'ordinary_straight_fraction':.5,'gentle_angle_deg':30,'gentle_radius_m':4,'single_turn_fraction':.5,'S_fraction':.2,'S_inter_bend_straight_m':3,'fixed_grid_length_m':100,'fixed_grid_points':5001}
    if any(plan['sampler_resolution'].get(k)!=v for k,v in expected.items()):raise ValueError('sampler resolution differs from implementation')
    if cfg['path']['spacing_m']!=.02 or c['family_fraction']!={'straight_or_gentle':.3,'rounded_left_or_right_60_to_100_degrees':.5,'smooth_S_path':.2}:raise ValueError('sampler contract changed')
    if plan.get('terminal_contract','terminal_contract_v1') not in ['terminal_contract_v1','terminal_contract_v2']:raise ValueError('unknown terminal contract')
    if plan['wall_budget_s_per_endpoint'] is not None:raise ValueError('no default wall limit; explicit user contract required')
    return plan,cfg,source


def prepare(path,output,check_interface=False):
    plan,cfg,source=load_plan(path);out=Path(output)
    if out.exists():raise FileExistsError('new preparation directory required')
    out.mkdir(parents=True);c=cfg['training_future_phase_C'];updates=plan['updates'];candidate=source['candidate']
    from .lower_command import SCALES
    if digest(candidate['config'])!=candidate['config_file_sha256']:raise ValueError('lower config changed')
    np.testing.assert_array_equal(np.asarray(SCALES),np.asarray(candidate['normalizer']['scales'],np.float32))
    for key in ['actor','checkpoint']:
        if digest(candidate[key])!=candidate[key+'_sha256']:raise ValueError('frozen lower identity mismatch')
    if digest(source['prepared_bank'])!=source['prepared_bank_sha256']:raise ValueError('prepared bank identity mismatch')
    validation_updates=[x for x in c['validation_updates'] if x<=updates]
    policy_per_endpoint=c['num_envs']*c['rollout_steps']*updates
    report=dict(schema=SCHEMA,state='prepared_not_run',plan=plan,config=cfg,source_manifest=source,training_policy_transitions_per_endpoint=policy_per_endpoint,training_policy_transitions_total=2*policy_per_endpoint,training_lower_ticks_total=8*policy_per_endpoint,validation_updates=validation_updates,maximum_validation_lower_ticks=(len(validation_updates)*2+1+2)*6*round(c['episode_s']/.005),additional_training_executed=0,additional_physics_executed=0,mainline_gate='baseline perfection not required',requires_explicit_execute=True,physical_training_smoke='not_run')
    if check_interface:
        env,snapshot=load_environment(cfg,source,plan.get('terminal_contract','terminal_contract_v1'));batch=PathBatch(env,snapshot,cfg,plan['seed'],0)
        ep=batch.reset(j.int32(0),j.int32(0));env.set_log_template(ep.state,ep.route.path,ep.route.speed)
        shapes=jax.eval_shape(batch.advance_one,ep,j.zeros(2))
        report['abstract_single_step_shapes']={'actor':list(shapes[1][0].shape),'critic':list(shapes[1][1].shape),'reward':list(shapes[2].shape),'done':list(shapes[3]['done'].shape)}
        # Batched tracing catches shape/reset semantics without evaluating physics.
        eps=jax.vmap(batch.reset)(j.arange(2,dtype=j.int32),j.zeros(2,j.int32))
        multi=jax.eval_shape(jax.vmap(batch.advance_one),eps,j.zeros((2,2)))
        report['abstract_batch_shapes']={'actor':list(multi[1][0].shape),'critic':list(multi[1][1].shape)}
    atomic_json(out/'PREPARATION.json',report);return report


def load_environment(cfg,source,terminal_contract="terminal_contract_v1"):
    from .path_command_env import PathCommandEnv
    from .frozen_local_lower import FrozenLocalLower
    from dataclasses import asdict
    candidate=source['candidate']
    from .lower_command import SCALES
    if digest(candidate['config'])!=candidate['config_file_sha256']:raise ValueError('lower config changed')
    np.testing.assert_array_equal(np.asarray(SCALES),np.asarray(candidate['normalizer']['scales'],np.float32))
    for key in ['actor','checkpoint']:
        if digest(candidate[key])!=candidate[key+'_sha256']:raise ValueError('lower SHA changed')
    if digest(source['prepared_bank'])!=source['prepared_bank_sha256']:raise ValueError('bank SHA changed')
    with Path(candidate['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
    spec=json.loads((Path(source['upper_run'])/'alpha0/frozen_config_stage52.json').read_text())
    env=PathCommandEnv(cfg,spec,FrozenLocalLower(params),terminal_contract)
    for key,live in [('model',env.physics.bundle.identity),('controller',asdict(env.cc)),('actuator',asdict(env.ac))]:
        if config_hash(live)!=config_hash(source[key]):raise ValueError('physical identity mismatch '+key)
    with Path(source['prepared_bank']).open('rb') as f:bank=pickle.load(f)
    return env,jax.tree.map(lambda x:j.asarray(x[source['bank_index']]),bank)

class PanelEvaluator:
    """Same fixed path protocol for every candidate; no nominal-stream matching."""
    def __init__(self,env,snapshot,config,check_budget=lambda:None,status=lambda **kw:None,account=lambda ticks:None):
        self.env=env;self.snapshot=snapshot;self.config=config;self.check_budget=check_budget;self.status=status;self.compiled={};self.account=account
    def evaluate(self,params,alpha,output,cached_names=None):
        from .path_command_selection import cases
        from .path_command_policy import PathCommandActor
        from .geometric_path import build_path
        from .direct_command_training import flatten_logs
        out=Path(output)
        if out.exists() and cached_names is None:raise FileExistsError('validation output immutable')
        out.mkdir(parents=True,exist_ok=cached_names is not None);traces={};h=self.config['training_future_phase_C']['episode_s'];pose=self.env.physics.helpers.pose(self.snapshot.data)
        for name,route,speed in cases(self.config):
            if cached_names and name in cached_names:
                traces[name]=dict(np.load(out/f'{name}.npz'));continue
            self.check_budget();path=build_path(route,np.asarray(pose),self.config);v=j.asarray(speed,j.float32)
            s=self.env.reset(self.snapshot,path,v,j.asarray(alpha,j.float32));self.env.set_log_template(s,path,v)
            if route not in self.compiled:
                def step(state,path,v,params):
                    obs,_,fault=self.env.observation(state,path,v,h)
                    state=state.replace(fault=state.fault|fault)
                    z=PathCommandActor().apply(params,obs)
                    return self.env.policy_step(state,z,path,v,h)
                self.compiled[route]=jax.jit(step).lower(s,path,v,params).compile()
            execute=self.compiled[route];chunks=[]
            for index in range(round(h/.02)):
                if index%50==0:self.check_budget();self.status(state='running',stage='validation',case=name,completed_intervals=index)
                s,log=execute(s,path,v,params);jax.block_until_ready(s);chunks.append(jax.device_get(log));self.account(int(j.sum(log['active_tick'])))
                if bool(s.fault)|bool(s.domain_exit)|bool(s.physical.failed):break
            flat=flatten_logs(jax.tree.map(lambda *xs:np.concatenate(xs),*chunks));valid=flat['active_tick'];d={k:x[valid] for k,x in flat.items()}
            d['goal_progress']=np.full(len(d['time']),float(path.goal))
            np.savez_compressed(out/f'{name}.npz',**d)
            np.savez_compressed(out/f'{name}_path.npz',s=np.asarray(path.s),xy=np.asarray(path.xy),heading=np.asarray(path.heading),curvature=np.asarray(path.curvature),goal=np.asarray(path.goal),turn_end=np.asarray(path.turn_end))
            if bool(s.fault):raise RuntimeError('fixed validation engineering fault; no best selection')
            traces[name]=d
        return traces


def _start_tensorboard(root):
    import socket,subprocess,sys,time
    root=Path(root);logdir=root/'tensorboard';logdir.mkdir(parents=True,exist_ok=True)
    port=None
    for candidate in range(6006,6027):
        with socket.socket() as probe:
            try:probe.bind(('127.0.0.1',candidate));port=candidate;break
            except OSError:continue
    if port is None:raise RuntimeError('no free TensorBoard port; existing servers untouched')
    with (root/'tensorboard_server.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(logdir.resolve()),'--host','127.0.0.1','--port',str(port),'--reload_interval','1'],stdout=log,stderr=log,start_new_session=True)
    info=dict(pid=process.pid,url=f'http://localhost:{port}',logdir=str(logdir.resolve()),scalar_verified=False)
    atomic_json(root/'tensorboard_server.json',info);return info


def _verify_reward_http(root,info,run):
    import urllib.request,urllib.parse,time
    error=None
    if not info.get('url'):
        atomic_json(Path(root)/f'{run}_reward_http_warning.json',dict(warning='TensorBoard server unavailable; local logging continues',scalar_verified=False));return False
    for _ in range(40):
        try:
            query=urllib.parse.urlencode({'run':run,'tag':'train/mean_step_reward'})
            with urllib.request.urlopen(info['url']+'/data/plugin/scalars/scalars?'+query,timeout=2) as response:rows=json.load(response)
            if rows and np.isfinite(rows[-1][2]):
                receipt=dict(info,scalar_verified=True,run=run,tag='train/mean_step_reward',latest=rows[-1]);atomic_json(Path(root)/f'{run}_reward_http_verified.json',receipt);print('TensorBoard '+info['url']+' run='+run+' reward verified',flush=True);return True
        except Exception as e:error=str(e)
        time.sleep(.5)
    atomic_json(Path(root)/f'{run}_reward_http_warning.json',dict(warning=str(error),scalar_verified=False))
    print('WARNING: reward HTTP not verified; local training continues: '+str(error),flush=True)
    return False


def rescore_saved_candidates(parent,out,alpha,cfg,identity):
    """Only saved same-protocol DEV; no extra physical panel for selection v2."""
    import torch
    from .path_command_selection import cases,summarize_task,persist_task_best
    task_pointer=Path(parent)/f'alpha{alpha}'/'best_task_model.json'
    if task_pointer.exists():
        incumbent=json.loads(task_pointer.read_text())
        if any(incumbent.get(k)!=v for k,v in identity.items()):raise ValueError('parent task best identity mismatch')
        for key in ['actor','checkpoint']:
            if digest(incumbent[key])!=incumbent[key+'_sha256']:raise ValueError('parent task best SHA mismatch')
        persist_task_best(out,incumbent['checkpoint'],incumbent['actor'],incumbent['metrics'],dict(identity,source_update=incumbent['source_update']))
    for directory in sorted((Path(parent)/f'alpha{alpha}'/'validation').glob('update_*')):
        if not (directory/'metrics.json').exists():continue
        update=int(directory.name.split('_')[-1]);checkpoint=directory.parents[1]/'checkpoints'/f'update_{update:04d}.pt';actor=checkpoint.with_name(f'actor_{update:04d}.pkl')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('identity')!=identity or saved['update']!=update:raise ValueError('saved candidate identity mismatch')
        traces={name:dict(np.load(directory/f'{name}.npz')) for name,_,_ in cases(cfg)}
        summary=summarize_task(traces,alpha,cfg)
        atomic_json(Path(out)/'rescored_parent'/f'update_{update:04d}.json',summary)
        persist_task_best(out,checkpoint,actor,summary,dict(identity,source_update=update))

def train(path,output,*,execute=False,resume=None):
    """Explicit-only bounded runner. Merely importing or preparing never trains."""
    if not execute:raise PermissionError('use preparation by default; training needs explicit execution')
    import time,torch,subprocess
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter
    from .lower_command_training import handle_candidate_ppo_result
    from .direct_command_training import clean_numbers,notify
    from .path_command_selection import summarize,persist_best,summarize_task,persist_task_best
    from .path_command_policy import PathCommandActor
    plan,cfg,source=load_plan(path);c=cfg['training_future_phase_C'];root=Path(output)
    if root.exists():raise FileExistsError('new run directory required, including resumed stage')
    root.mkdir(parents=True);torch.set_num_threads(2)
    from .runtime import configure_compilation_cache
    compilation_cache=configure_compilation_cache()
    env,snapshot=load_environment(cfg,source,plan.get('terminal_contract','terminal_contract_v1'));n=c['num_envs'];steps=c['rollout_steps'];updates=plan['updates']
    candidate=source['candidate'];protected_paths=[Path(candidate[k]) for k in ['actor','checkpoint','config']]
    protected_paths.extend([Path(candidate['original_protocol_best']['checkpoint']),Path(candidate['config']).parent/'training/best_model.json'])
    for upper in candidate['upper'].values():protected_paths.extend([Path(upper['actor']),Path(upper['checkpoint']),Path(upper['actor']).parents[2]/'best_model.json'])
    protected={str(p):digest(p) for p in protected_paths if p.exists()}
    manifest=dict(schema=SCHEMA,plan=plan,config=cfg,source=source,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protected=protected,physical_continuation='reset to original complete prepared bank, not resumed simulator state',terminal_contract=plan.get('terminal_contract','terminal_contract_v1'),data_role='random TRAIN; fixed six-route DEV; no independent holdout',resume_parent=None if resume is None else str(Path(resume).resolve()))
    manifest['execution_optimizations']=dict(evaluator='vmap_scan50',compilation_cache=str(compilation_cache),ppo='existing_direct_command_ppo',actual_batch_dimensions=[n,steps],experimental_summary_reset_guard=False)
    atomic_json(root/'manifest.json',manifest)
    try:tb=_start_tensorboard(root)
    except (OSError,RuntimeError) as e:
        tb=dict(url=None,scalar_verified=False,warning=str(e));atomic_json(root/'tensorboard_server_warning.json',tb)
    status={'state':'initializing','declared_updates_per_endpoint':updates,'policy_transitions_total':0,'training_control_ticks_total':0,'validation_control_ticks_total':0}
    def update_status(**kw):
        status.update(kw)
        if 'alpha' in status and 'completed_updates' in status:status.setdefault('endpoint_updates',{})[f"alpha{status['alpha']}"]=status['completed_updates']
        status['last_update_epoch']=time.time();atomic_json(root/'status.json',status)
    def account_validation(ticks):status['validation_control_ticks_total']+=ticks
    shared_eval_cache={}
    baseline_done=False;best_panels={};writer=None;algo=None;out=None;episodes=None;learner_ready=False
    if resume:
        parent_status=json.loads((Path(resume)/'status.json').read_text())
        status['parent_consumed_policy_transitions']=parent_status.get('policy_transitions_total',0)+parent_status.get('parent_consumed_policy_transitions',0)
        status['parent_reported_updates']=parent_status.get('completed_updates',0)
    update_status(stage='loading')
    try:
        for alpha in plan['alphas']:
            algo=None;learner_ready=False;started=time.monotonic();update_status(stage='compiling',alpha=alpha,completed_updates=0);out=root/f'alpha{alpha}';out.mkdir();writer=SummaryWriter(str(root/'tensorboard'/f'alpha{alpha}'))
            def check_budget():
                status['endpoint_elapsed_s']=time.monotonic()-started
            def to_torch(a,cr):return TensorDict({'policy':torch.utils.dlpack.from_dlpack(a),'critic':torch.utils.dlpack.from_dlpack(cr)},batch_size=[n])
            seed=plan['seed']+alpha;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            batch=PathBatch(env,snapshot,cfg,plan['seed'],alpha)
            sample=batch.reset(j.int32(0),j.int32(0));env.set_log_template(sample.state,sample.route.path,sample.route.speed)
            reset=jax.jit(jax.vmap(batch.reset));episodes=reset(j.arange(n,dtype=j.int32),j.zeros(n,j.int32));jax.block_until_ready(episodes)
            observe=jax.jit(jax.vmap(batch.observation))
            a,cr,fault=observe(episodes)
            if bool(j.any(fault)):raise RuntimeError('reset observation fault')
            status['compile_and_setup_seconds']=time.monotonic()-started
            obs=to_torch(a,cr);algo=make_algorithm(obs,steps,cfg,'cuda')
            identity=dict(schema=SCHEMA,alpha=alpha,lower_actor_sha256=source['candidate']['actor_sha256'],config_sha256=config_hash(cfg),sampler_sha256=config_hash(plan['sampler_resolution']),best_sha256=config_hash(plan['best_resolution']),seed=plan['seed'])
            start_update=0;rejections=0
            if resume and (Path(resume)/f'alpha{alpha}'/'last_completed.json').exists():
                receipt,parent_best=resume_source(resume,alpha,identity)
                saved=restore_checkpoint(receipt['checkpoint'],algo,identity);
                if parent_best:persist_best(out,parent_best['checkpoint'],parent_best['actor'],parent_best['metrics'],{**identity,'source_update':parent_best['source_update']})
                start_update=saved['update'];rejections=saved['consecutive_rejections']
                if start_update>updates:raise ValueError('resume boundary exceeds declared target')
                if saved.get('route_counters') is None or np.shape(saved['route_counters'])!=(n,):raise ValueError('resume requires per-environment route counters')
                route_counters=np.asarray(saved['route_counters'])
                fault_file=Path(resume)/f'alpha{alpha}'/'engineering_fault_states.pkl'
                if fault_file.exists():
                    with fault_file.open('rb') as f:failed_episodes=pickle.load(f)
                    route_counters=np.maximum(route_counters,np.asarray(failed_episodes.episode))
                episodes=reset(j.arange(n,dtype=j.int32),j.asarray(route_counters)+1)
                atomic_json(out/'resume_receipt.json',dict(parent_checkpoint=receipt,parent_terminal_contract=saved.get('terminal_contract','terminal_contract_v1'),terminal_contract=plan.get('terminal_contract'),restored_update=start_update,parent_reported_updates=status.get('parent_reported_updates'),physical_state='reset to prepared bank',routes='next unused after saved and fault snapshot counters'))
                a,cr,fault=observe(episodes)
                if bool(j.any(fault)):raise RuntimeError('resume observation fault')
                obs=to_torch(a,cr)
            advance=None if start_update>=updates else jax.jit(jax.vmap(batch.advance_one)).lower(episodes,j.zeros((n,2))).compile()
            status['compile_and_setup_seconds']=time.monotonic()-started
            initial_checkpoint=save_checkpoint(out/'checkpoints',algo,start_update,identity,rejections=rejections,route_counters=episodes.episode)
            atomic_json(out/'last_completed.json',dict(update=start_update,checkpoint=str(initial_checkpoint.resolve()),checkpoint_sha256=digest(initial_checkpoint),schema=SCHEMA))
            update_status(completed_updates=start_update)
            learner_ready=True
            if resume:rescore_saved_candidates(resume,out,alpha,cfg,identity)
            from .path_command_evaluation import BatchedPanelEvaluator
            evaluator=BatchedPanelEvaluator(env,snapshot,cfg,check_budget,update_status,account_validation)
            evaluator.compiled=shared_eval_cache
            if not baseline_done:
                cached=set()
                if resume:
                    from .path_command_cache import reuse_zero_upper
                    from .geometric_path import build_path
                    pose=np.asarray(env.physics.helpers.pose(snapshot.data))
                    paths={route:build_path(route,pose,cfg) for route in cfg['stage_C_evaluation']['route_ids']}
                    cached=reuse_zero_upper(resume,root,cfg,source,paths)
                zero=PathCommandActor().init(jax.random.PRNGKey(0),j.zeros(351));evaluator.evaluate(zero,0,root/'zero_upper',cached_names=cached if resume else None);baseline_done=True
                status['reused_baseline_cases']=len(cached)
            accepted=0;http_verified=False;stop=None;update=start_update;window_stats=None;window_episodes=0;window_return=0.;window_costs=np.zeros(10);window_physical=0;window_domain=0
            from .path_command_statistics import summarize_statistics,COST_KEYS
            for update in range(start_update+1,updates+1):
                check_budget();tick_start=time.monotonic();update_status(state='running',stage='sampling',case=None,completed_intervals=None,alpha=alpha,completed_updates=update-1,declared_updates=updates)
                rewards=[];rollout_stats=[];completed_episodes=[];caps={};episode_ends=0;physical_failures=0;domain_failures=0;cost_totals_update=np.zeros(10)
                with torch.no_grad():
                    for _ in range(steps):
                        check_budget();z=algo.act(obs)
                        if not bool(torch.isfinite(z).all()):raise RuntimeError('nonfinite upper latent')
                        episodes,next_obs,reward,flags,final,stats=advance(episodes,jax.dlpack.from_dlpack(z.detach().contiguous()))
                        jax.block_until_ready(episodes)
                        status['training_control_ticks_total']+=int(j.sum(stats['active_ticks']));status['policy_transitions_total']+=n
                        if bool(j.any(flags['engineering_fault'])) or bool(j.any(next_obs[2])):
                            with (out/'engineering_fault_states.pkl').open('wb') as f:pickle.dump(jax.device_get(episodes),f)
                            status['engineering_fault_count']=int(j.sum(flags['engineering_fault']|next_obs[2]))
                            raise RuntimeError('nonfinite/observation/lower engineering fault; discard rollout, no optimizer update')
                        obs=to_torch(next_obs[0],next_obs[1]);algo.process_env_step(obs,torch.utils.dlpack.from_dlpack(reward),torch.utils.dlpack.from_dlpack(flags['done']),{})
                        rewards.append(np.asarray(reward));rollout_stats.append(np.asarray(stats['rollout']));completed_episodes.append(np.asarray(stats['completed_totals']))
                        episode_ends+=int(j.sum(flags['done']));physical_failures+=int(j.sum(flags['physical_failure']));domain_failures+=int(j.sum(flags['tracking_domain_failure']))
                        cost_totals_update+=np.asarray(j.sum(stats['completed_component_costs'],axis=0))
                        diagnostic=jax.tree.map(lambda x:np.asarray(j.sum(x,axis=0)),stats['diagnostics'])
                        window_stats=diagnostic if window_stats is None else jax.tree.map(lambda x,y:x+y,window_stats,diagnostic)
                        for key,count in stats['component_caps'].items():caps[key]=caps.get(key,0)+int(j.sum(count))
                    algo.compute_returns(obs)
                sample_seconds=time.monotonic()-tick_start;opt_start=time.monotonic();update_status(stage='optimizing');metrics=algo.update();opt_seconds=time.monotonic()-opt_start;check_budget()
                update_status(stage='optimized',completed_updates=update)
                rejections,stop=handle_candidate_ppo_result(algo,metrics,rejections,plan);accepted+=int(metrics['accepted_epochs']>0)
                totals=np.sum(np.stack(rollout_stats),axis=(0,1));done_totals=np.sum(np.stack(completed_episodes),axis=(0,1));count=max(totals[0],1)
                record=dict(update=update,sampling_model_update=update-1,accepted_batches_this_stage=accepted,rejected_streak=rejections,optimizer=clean_numbers(metrics),mean_step_reward=float(np.mean(rewards)),reward_unit='policy interval .02s sum of four lower rewards including failure replacement',rollout_speed_rmse=float(np.sqrt(totals[1]/count)),rollout_path_rmse=float(np.sqrt(totals[2]/count)),rollout_heading_rmse=float(np.sqrt(totals[3]/count)),complete_episode_count=episode_ends,complete_episode_totals=done_totals.tolist(),physical_failures=physical_failures,tracking_domain_failures=domain_failures,engineering_faults=0,complete_episode_return_mean=None if not episode_ends else float(done_totals[4]/episode_ends),complete_episode_cost_integral_means=None if not episode_ends else dict(zip(COST_KEYS,(cost_totals_update/episode_ends).tolist())),log_std=algo.policy.log_std.detach().cpu().tolist(),learning_rates=[g['lr'] for g in algo.optimizer.param_groups],terminal_contract=env.terminal_contract,physical_steps=status['training_control_ticks_total'],policy_steps=status['policy_transitions_total'],component_cap_counts=caps,sample_seconds=sample_seconds,optimize_seconds=opt_seconds,stop_reason=stop)
                window_episodes+=episode_ends;window_return+=done_totals[4];window_costs+=cost_totals_update;window_physical+=physical_failures;window_domain+=domain_failures
                if update%25==0 or update==updates:
                    diagnostic=summarize_statistics(window_stats,cfg)
                    diagnostic.update(update=update,complete_episode_count=window_episodes,complete_episode_return_mean=None if not window_episodes else float(window_return/window_episodes),complete_episode_cost_integral_means=None if not window_episodes else dict(zip(COST_KEYS,(window_costs/window_episodes).tolist())),physical_failure_count=window_physical,tracking_domain_failure_count=window_domain,engineering_fault_count=0,physical_failure_rate=None if not window_episodes else window_physical/window_episodes,tracking_domain_failure_rate=None if not window_episodes else window_domain/window_episodes)
                    atomic_json(out/'statistics'/f'update_{update:04d}.json',diagnostic)
                    window_stats=None;window_episodes=0;window_return=0.;window_costs=np.zeros(10);window_physical=0;window_domain=0
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                writer.add_scalar('train/mean_step_reward',record['mean_step_reward'],update)
                for key in ['rollout_speed_rmse','rollout_path_rmse','rollout_heading_rmse','complete_episode_count','physical_failures','tracking_domain_failures']:writer.add_scalar('train/'+key,record[key],update)
                for key in ['surrogate','value','entropy','mean_kl','accepted_epochs']:writer.add_scalar('ppo/'+key,float(metrics[key]),update)
                for channel,value in enumerate(algo.policy.log_std.detach().cpu().tolist()):writer.add_scalar(f'ppo/std_{channel}',float(np.exp(value)),update)
                for group,g in enumerate(algo.optimizer.param_groups):writer.add_scalar(f'ppo/lr_{group}',g['lr'],update)
                writer.flush()
                if update%c['save_every']==0 or update==updates or stop:
                    checkpoint=save_checkpoint(out/'checkpoints',algo,update,identity,rejections=rejections,route_counters=episodes.episode)
                    atomic_json(out/'last_completed.json',dict(update=update,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest(checkpoint),schema=SCHEMA))
                if not http_verified and (update==start_update+1 or update%25==0):http_verified=_verify_reward_http(root,tb,f'alpha{alpha}')
                update_status(stage='optimized',completed_updates=update,accepted_batches=accepted,last_reward=record['mean_step_reward'])
                if update in c['validation_updates'] and not stop:
                    dev_started=time.monotonic();params=export_actor(algo.policy);traces=evaluator.evaluate(params,alpha,out/'validation'/f'update_{update:04d}');summary=summarize(traces,alpha,cfg)
                    atomic_json(out/'validation'/f'update_{update:04d}'/'metrics.json',summary)
                    persist_best(out,checkpoint,out/'checkpoints'/f'actor_{update:04d}.pkl',summary,dict(identity,source_update=update))
                    status['dev_seconds']=status.get('dev_seconds',0)+time.monotonic()-dev_started
                    task_summary=summarize_task(traces,alpha,cfg)
                    atomic_json(out/'validation'/f'update_{update:04d}'/'task_metrics.json',task_summary)
                    persist_task_best(out,checkpoint,out/'checkpoints'/f'actor_{update:04d}.pkl',task_summary,dict(identity,source_update=update))
                print(f'alpha{alpha} update{update}/{updates} reward={record["mean_step_reward"]:.6g} accepted={accepted}',flush=True)
                if stop:break
            writer.close();notify('STTW path training stage ended',f'alpha{alpha}: {update}/{updates}; {stop or "budget reached"}')
            if stop:raise RuntimeError(stop)
            if not (out/'best_task_model.json').exists():raise RuntimeError('no fixed validation candidate; no last-as-best fallback')
            best=json.loads((out/'best_task_model.json').read_text())
            if digest(best['actor'])!=best['actor_sha256'] or digest(best['checkpoint'])!=best['checkpoint_sha256']:raise ValueError('best task weights identity changed')
            with Path(best['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
            cached_final=set()
            if resume:
                from .path_command_cache import reuse_final_panel, inherit_numerical_history
                cached_final=reuse_final_panel(resume,out,alpha,cfg,source,best,paths)
                inherit_numerical_history(resume,out,alpha)
            update_status(reused_final_cases=len(cached_final))
            best_panels[alpha]=evaluator.evaluate(params,alpha,out/'final_best',cached_names=cached_final if cached_final else None);atomic_json(out/'final_best'/'metrics.json',summarize_task(best_panels[alpha],alpha,cfg));atomic_json(out/'final_best'/'source.json',best)
        for p,sha in protected.items():
            if digest(p)!=sha:raise RuntimeError('protected source changed')
        from .path_command_reporting import report_final_panel
        report_final_panel(root,cfg)
        from .path_command_reporting import convergence_decision
        paired=json.loads((root/'report'/'paired_metrics.json').read_text());decisions={}
        qualified=all(json.loads((root/f'alpha{a}'/'best_task_model.json').read_text())['qualified_task'] for a in plan['alphas'])
        for a in plan['alphas']:
            endpoint=root/f'alpha{a}'
            history=[dict(update=int(p.parent.name.split('_')[-1]),summary=json.loads(p.read_text())) for p in sorted((endpoint/'validation').glob('update_*/task_metrics.json'))]
            inherited=[dict(update=int(p.parent.name.split('_')[-1]),summary=json.loads(p.read_text())) for p in sorted((endpoint/'inherited_history'/'validation').glob('update_*/task_metrics.json'))]
            history=sorted({row['update']:row for row in inherited+history}.values(),key=lambda row:row['update'])
            stats_root=endpoint/'statistics'
            if not (stats_root/'update_0200.json').exists():stats_root=endpoint/'inherited_history'/'statistics'
            previous=stats_root/'update_0175.json';last=stats_root/'update_0200.json';trend={}
            if previous.exists() and last.exists():
                key='lateral_rmse_m' if a==0 else 'speed_rmse_m_s'
                before=json.loads(previous.read_text())['windows']['strong_demand'][key];after=json.loads(last.read_text())['windows']['strong_demand'][key]
                if before is not None and after is not None:trend=dict(matched_condition_relative_improvement=(before-after)/max(abs(before),1e-12),window='strong_demand',metric=key,updates=[175,200])
            decisions[f'alpha{a}']=convergence_decision(history,trend,paired_qualified=qualified)
        atomic_json(root/'report'/'next_step.json',dict(endpoint_decisions=decisions,paired_task_qualified=qualified,automatic_extension=False,paired_metrics='paired_metrics.json',note='Both jointly qualified endpoints need not be forced to diverge; inspect paired metrics for preference claim.'))
        update_status(state='completed',stage='reported',protected_unchanged=True);notify('STTW path pipeline completed',str(root))
    except BaseException as e:
        if learner_ready and algo is not None and status.get('stage')!='optimizing':
            completed=int(status.get('completed_updates',0));checkpoint=out/'checkpoints'/f'update_{completed:04d}.pt'
            if not checkpoint.exists():checkpoint=save_checkpoint(out/'checkpoints',algo,completed,identity,rejections=rejections,route_counters=episodes.episode)
            atomic_json(out/'last_completed.json',dict(update=completed,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest(checkpoint),schema=SCHEMA))
        if writer is not None:writer.close()
        update_status(state='error',error_type=type(e).__name__,message=str(e));notify('STTW path error',str(e));raise
