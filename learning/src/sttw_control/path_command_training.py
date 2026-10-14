"""351/352 geometric task bindings to the existing RSL PPO implementation."""
from pathlib import Path
import json,os,pickle,random
import numpy as np
import jax
import jax.numpy as j
from flax import struct
from .path_command_policy import SCHEMA
from .preference_best import atomic_json,config_hash,digest
from .path_command_conditioning import is_conditional,is_strong,execution_dimensions


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
    if is_strong(config):
        from .path_command_policy import make_torch_path_network
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
    return DirectCommandPPO(policy,storage,num_learning_epochs=c['epochs'],num_mini_batches=c['minibatches'],clip_param=c['clip'],gamma=c['gamma'],lam=c['gae_lambda'],value_loss_coef=1.,entropy_coef=c['entropy'],use_clipped_value_loss=False,schedule='fixed',desired_kl=c['soft_kl'],normalize_advantage_per_mini_batch=False,device=device,actor_lr=c['actor_lr'],critic_lr=c['critic_lr'],adam_betas=(.9,.999),adam_epsilon=1e-5,weight_decay=0.,std_min=c['std_min'],std_max=c['std_max'],hard_kl=c['hard_kl'],actor_grad_limit=c['grad_clip_actor'],critic_grad_limit=c['grad_clip_critic'],kl_group_index=None)


def export_actor(policy):
    from .path_command_policy import export_torch_actor
    return export_torch_actor(policy.actor,getattr(policy,'path_network_config',None))


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
    out=Path(parent)/('shared' if alpha is None else f'alpha{alpha}')
    receipt=json.loads((out/'last_completed.json').read_text())
    if receipt['schema']!=SCHEMA or digest(receipt['checkpoint'])!=receipt['checkpoint_sha256']:
        raise ValueError('resume boundary identity mismatch')
    pointer=out/('best_conditional_model.json' if alpha is None else 'best_model.json')
    best=json.loads(pointer.read_text()) if pointer.exists() else None
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
    def reset(self,eid,episode,absolute_update=1):
        from .path_command_scenarios import sample_training_route
        pose=self.env.physics.helpers.pose(self.snapshot.data)
        route=sample_training_route(self.seed,eid,episode,pose,self.config)
        if self.alpha is None:
            from .path_command_conditioning import sample_episode_alpha
            alpha=sample_episode_alpha(self.config['alpha_conditioning']['seed'],eid,episode,self.config,absolute_update)
        else:alpha=j.asarray(self.alpha)
        state=self.env.reset(self.snapshot,route.path,route.speed,alpha)
        return Episode(state,route,eid,episode,j.zeros(6),j.zeros(10))
    def observation(self,ep):return self.env.observation(ep.state,ep.route.path,ep.route.speed,self.horizon)
    def advance_one(self,ep,z,absolute_update=1):
        end,logs=self.env.policy_step(ep.state,z,ep.route.path,ep.route.speed,self.horizon)
        final=self.env.observation(end,ep.route.path,ep.route.speed,self.horizon)
        flags=termination_flags(end.tick,end.physical.failed,end.fault|final[2],end.domain_exit,self.horizon,self.config,self.env.terminal_contract)
        active=logs['active_tick'];ev=logs['actual_forward_speed']-ep.route.speed
        sums=j.array([j.sum(active),j.sum(j.where(active,ev**2,0)),j.sum(j.where(active,logs['path_cross_track']**2,0)),j.sum(j.where(active,logs['path_heading_error']**2,0)),j.sum(logs['scored_tick_reward']),j.max(logs['peak_roll'])])
        totals=ep.totals+sums;totals=totals.at[5].set(j.maximum(ep.totals[5],sums[5]))
        from .path_command_statistics import step_statistics,merge_statistics,reset_statistics
        diagnostics=step_statistics(logs,ep.route,self.config)
        cost_totals=ep.component_totals+diagnostics['cost_effective_sum']*self.config['timing']['lower_dt_s']
        ended=ep.replace(state=end,totals=totals,component_totals=cost_totals)
        nxt=reset_finished_episode(ended,flags,lambda eid,episode:self.reset(eid,episode,absolute_update))
        diagnostics=merge_statistics(diagnostics,reset_statistics(nxt.route,flags['done']&~flags['engineering_fault']))
        obs=self.observation(nxt)
        return nxt,obs,j.sum(logs['scored_tick_reward']),flags,final,dict(diagnostics=diagnostics,completed_component_costs=j.where(flags['done'],cost_totals,0.),rollout=sums,completed_totals=j.where(flags['done'],totals,0.),family=ep.route.family,episode=ep.episode,active_ticks=j.sum(active),component_caps={k:j.sum(v&active) for k,v in logs['component_capped'].items()})


def load_plan(path):
    path=Path(path).resolve();plan=json.loads(path.read_text());cfgpath=(path.parent/plan['design_config']).resolve();sourcepath=(path.parent/plan['integration_manifest']).resolve()
    cfg=json.loads(cfgpath.read_text());source=json.loads(sourcepath.read_text());c=cfg['training_future_phase_C']
    conditional=is_conditional(plan)
    if conditional:
        from .path_command_conditioning import validate_conditional_plan
        validate_conditional_plan(plan,cfg)
    elif plan['schema']!='sttw_path_training_plan_v1' or cfg['network']['actor_dim']!=351:raise ValueError('wrong task schema')
    if not conditional and (plan['alphas']!=[0,1] or not 1<=plan['updates']<=c['updates_initial'] or plan['automatic_extension']):raise ValueError('unapproved endpoint/budget extension')
    if source['candidate']['source_update']!=300:raise ValueError('frozen300 required')
    # These are explicit resolved details; reject silent changes to code constants.
    expected={'ordinary_fraction':.3,'ordinary_straight_fraction':.5,'gentle_angle_deg':30,'gentle_radius_m':4,'single_turn_fraction':.5,'S_fraction':.2,'S_inter_bend_straight_m':3,'fixed_grid_length_m':100,'fixed_grid_points':5001}
    if any(plan['sampler_resolution'].get(k)!=v for k,v in expected.items()):raise ValueError('sampler resolution differs from implementation')
    if cfg['path']['spacing_m']!=.02 or c['family_fraction']!={'straight_or_gentle':.3,'rounded_left_or_right_60_to_100_degrees':.5,'smooth_S_path':.2}:raise ValueError('sampler contract changed')
    if plan.get('terminal_contract','terminal_contract_v1') not in ['terminal_contract_v1','terminal_contract_v2']:raise ValueError('unknown terminal contract')
    if plan['wall_budget_s_per_endpoint'] is not None:raise ValueError('no default wall limit; explicit user contract required')
    from .path_command_scenarios import validate_training_sampler
    validate_training_sampler(cfg)
    version=plan.get('sampler_version','legacy')
    reward_fork=plan.get('initialization',{}).get('mode')=='weights_only_reward_fork'
    if conditional:
        plan['initialization']['checkpoint']=str((path.parent/plan['initialization']['checkpoint']).resolve())
        plan['baseline_parent']=str((path.parent/plan['baseline_parent']).resolve())
    elif reward_fork:
        from .path_command_fork import reward_validation_schedule
        reward_validation_schedule(plan,cfg)
        if version!='demand_exposure_v2' or plan['seed']!=911101:raise ValueError('reward fork pilot contract changed')
        if plan['sampler_resolution'].get('training_sampler')!=cfg.get('training_sampler'):raise ValueError('reward fork sampler disconnected')
        if plan.get('conditional_extension'):raise ValueError('reward pilot cannot extend')
        if plan.get('terminal_contract')!='terminal_contract_v2' or plan.get('http_failure_policy')!='error':raise ValueError('reward pilot terminal/TensorBoard contract changed')
        if plan.get('coverage_monitor')!={'target':.05,'check_every':10,'consecutive_low_windows':2}:raise ValueError('reward pilot coverage monitor changed')
        if cfg.get('task_selection_schema')!='path_safety_task_v1' or plan.get('task_selection_schema')!='path_safety_task_v1':raise ValueError('reward fork safety selection required')
        init=plan['initialization'];init['parent_run']=str((path.parent/init['parent_run']).resolve())
        receipt_path=(path.parent/init['identities_file']).resolve();init['identities_file']=str(receipt_path)
        if receipt_path.exists():
            frozen=json.loads(receipt_path.read_text())
            if frozen['parent_run']!=init['parent_run']:raise ValueError('frozen parent run mismatch')
            init['expected_parents']=frozen['expected_parents']
    elif version=='demand_exposure_v2':
        if plan['sampler_resolution'].get('training_sampler')!=cfg.get('training_sampler'):raise ValueError('sampler config not connected to declared resolution')
        if plan.get('conditional_extension')!={'enabled':True,'block':50,'hard_max_updates':200,'default_updates':100}:raise ValueError('coverage extension contract changed')
        if plan.get('coverage_monitor')!={'target':.05,'check_every':10,'consecutive_low_windows':2}:raise ValueError('coverage exposure contract changed')
        for key in ['parent_run','identities']:plan['fork'][key]=str((path.parent/plan['fork'][key]).resolve())
        from .path_command_fork import validate_fork_contract
        validate_fork_contract(plan,cfg,source)
    elif version!='legacy' or cfg.get('training_sampler') is not None:raise ValueError('unknown or disconnected sampler version')
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
    from .path_command_fork import reward_validation_schedule
    schedule=reward_validation_schedule(plan,cfg) if plan.get('initialization',{}).get('mode')=='weights_only_reward_fork' else c['validation_updates']
    validation_updates=[x for x in schedule if x<=updates]
    policy_per_endpoint=c['num_envs']*c['rollout_steps']*updates
    report=dict(schema=SCHEMA,state='prepared_not_run',plan=plan,config=cfg,source_manifest=source,training_policy_transitions_per_endpoint=policy_per_endpoint,training_policy_transitions_total=2*policy_per_endpoint,training_lower_ticks_total=8*policy_per_endpoint,validation_updates=validation_updates,maximum_validation_lower_ticks=(len(validation_updates)*2+1+2)*6*round(c['episode_s']/.005),additional_training_executed=0,additional_physics_executed=0,mainline_gate='baseline perfection not required',requires_explicit_execute=True,physical_training_smoke='not_run')
    if plan.get('sampler_version')=='demand_exposure_v2':
        report.update(maximum_validation_lower_ticks=len(validation_updates)*2*6*round(c['episode_s']/.005),conditional_hard_max_updates_per_endpoint=200,baseline_physics=0,final_best_physics=0)
    if plan.get('initialization',{}).get('mode')=='weights_only_reward_fork':
        report.update(conditional_hard_max_updates_per_endpoint=plan['updates'],parent_identities_frozen=bool(plan['initialization'].get('expected_parents')),formal_stage='not_run',maximum_validation_lower_ticks=len(validation_updates)*48000,baseline_physics=0,final_best_physics=0)
    if is_conditional(plan):
        n,steps=execution_dimensions(plan,cfg);engineering=plan.get('engineering_check',False)
        if digest(plan['initialization']['checkpoint'])!=plan['initialization']['checkpoint_sha256']:raise ValueError('parent checkpoint SHA mismatch')
        report.update(policy_mode=plan['policy_mode'],training_policy_transitions_total=n*steps*updates,
            training_policy_transitions_per_endpoint=None,training_lower_ticks_total=4*n*steps*updates,
            validation_updates=[] if engineering else validation_updates,maximum_validation_lower_ticks=0 if engineering else len(validation_updates)*len(plan['alphas'])*6*4000,
            engineering_check=engineering,discard_engineering_update=engineering,actual_batch_dimensions=[n,steps],learner_count=1,
            baseline_physics=0,final_best_physics=0,conditional_hard_max_updates_per_endpoint=None)
    if check_interface:
        env,snapshot=load_environment(cfg,source,plan.get('terminal_contract','terminal_contract_v1'));batch=PathBatch(env,snapshot,cfg,plan['seed'],None if is_conditional(plan) else 0)
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
        from .path_command_policy import make_path_actor
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
            cache_key=(config_hash(self.config),self.env.terminal_contract,route)
            if cache_key not in self.compiled:
                def step(state,path,v,params):
                    obs,_,fault=self.env.observation(state,path,v,h)
                    state=state.replace(fault=state.fault|fault)
                    z=make_path_actor(self.config).apply(params,obs)
                    return self.env.policy_step(state,z,path,v,h)
                self.compiled[cache_key]=jax.jit(step).lower(s,path,v,params).compile()
            execute=self.compiled[cache_key];chunks=[]
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
    from .path_command_selection import cases,summarize_task,persist_task_best,task_pointer_name
    task_pointer=Path(parent)/f'alpha{alpha}'/task_pointer_name(cfg)
    if task_pointer.exists():
        incumbent=json.loads(task_pointer.read_text())
        if any(incumbent.get(k)!=v for k,v in identity.items()):raise ValueError('parent task best identity mismatch')
        for key in ['actor','checkpoint']:
            if digest(incumbent[key])!=incumbent[key+'_sha256']:raise ValueError('parent task best SHA mismatch')
        persist_task_best(out,incumbent['checkpoint'],incumbent['actor'],incumbent['metrics'],dict(identity,source_update=incumbent['source_update']))
    for directory in sorted((Path(parent)/f'alpha{alpha}'/'validation').glob('update_*')):
        if not (directory/'metrics.json').exists():continue
        update=int(directory.name.split('_')[-1])
        source_file=directory/'source.json'
        if source_file.exists():
            record=json.loads(source_file.read_text())
            if record.get('source_update')!=update or any(record.get(k)!=v for k,v in identity.items()):raise ValueError('saved candidate receipt identity mismatch')
            checkpoint=Path(record['checkpoint']);actor=Path(record['actor'])
            for key,file in (('checkpoint',checkpoint),('actor',actor)):
                if digest(file)!=record[key+'_sha256']:raise ValueError('saved candidate receipt SHA mismatch')
        else:
            if cfg.get('task_selection_schema')=='path_safety_task_v1':raise ValueError('safety candidate requires immutable source receipt')
            checkpoint=directory.parents[1]/'checkpoints'/f'update_{update:04d}.pt';actor=checkpoint.with_name(f'actor_{update:04d}.pkl')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('schema')!=SCHEMA or saved.get('identity')!=identity or saved['update']!=update:raise ValueError('saved candidate identity mismatch')
        traces={name:dict(np.load(directory/f'{name}.npz')) for name,_,_ in cases(cfg)}
        summary=summarize_task(traces,alpha,cfg)
        atomic_json(Path(out)/'rescored_parent'/f'update_{update:04d}.json',summary)
        persist_task_best(out,checkpoint,actor,summary,dict(identity,source_update=update))

class CoveragePause(RuntimeError):
    pass


def train(path,output,*,execute=False,resume=None):
    """Explicit-only bounded runner. Merely importing or preparing never trains."""
    if not execute:raise PermissionError('use preparation by default; training needs explicit execution')
    import time,torch,subprocess
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter
    from .lower_command_training import handle_candidate_ppo_result
    from .direct_command_training import clean_numbers,notify
    from .path_command_selection import summarize,persist_best,summarize_task,persist_task_best,task_pointer_name
    from .path_command_policy import make_path_actor
    plan,cfg,source=load_plan(path);c=cfg['training_future_phase_C'];root=Path(output)
    conditional=is_conditional(plan);engineering=conditional and plan.get('engineering_check',False)
    reward_fork=plan.get('initialization',{}).get('mode')=='weights_only_reward_fork'
    coverage_mode=plan.get('sampler_version')=='demand_exposure_v2' and not reward_fork and not conditional
    if conditional and resume and not (Path(resume)/'shared/last_completed.json').exists():raise ValueError('shared resume boundary missing')
    fork_receipt=None
    if coverage_mode:
        from .path_command_fork import validate_fork_contract,restore_fork,extension_decision
        fork_receipt=validate_fork_contract(plan,cfg,source)
    if reward_fork:
        from .path_command_fork import validate_reward_fork,restore_reward_fork,reward_validation_schedule,validate_reward_resume
        fork_receipt=validate_reward_fork(plan,cfg,source)
        validate_reward_resume(plan,cfg,source,resume)
    validation_schedule=[] if engineering else (reward_validation_schedule(plan,cfg) if reward_fork else c['validation_updates'])
    if root.exists():raise FileExistsError('new run directory required, including resumed stage')
    root.mkdir(parents=True);torch.set_num_threads(2)
    from .runtime import configure_compilation_cache
    compilation_cache=configure_compilation_cache()
    initial_status=dict(state='initializing',stage='loading_frozen_physics_and_bank',pid=os.getpid(),endpoint_updates={'alpha0':0,'alpha1':0},declared_updates_per_endpoint=plan['updates'],last_update_epoch=time.time())
    if conditional:initial_status.update(policy_mode=plan['policy_mode'],endpoint_updates={},shared_updates=0,declared_updates=plan['updates'],engineering_check=engineering)
    atomic_json(root/'status.json',initial_status)
    env,snapshot=load_environment(cfg,source,plan.get('terminal_contract','terminal_contract_v1'));n,steps=execution_dimensions(plan,cfg);updates=plan['updates']
    candidate=source['candidate'];protected_paths=[Path(candidate[k]) for k in ['actor','checkpoint','config']]
    protected_paths.extend([Path(candidate['original_protocol_best']['checkpoint']),Path(candidate['config']).parent/'training/best_model.json'])
    for upper in candidate['upper'].values():protected_paths.extend([Path(upper['actor']),Path(upper['checkpoint']),Path(upper['actor']).parents[2]/'best_model.json'])
    if conditional:protected_paths.append(Path(plan['initialization']['checkpoint']))
    protected={str(p):digest(p) for p in protected_paths if p.exists()}
    manifest=dict(schema=SCHEMA,plan=plan,config=cfg,source=source,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),protected=protected,physical_continuation='reset to original complete prepared bank, not resumed simulator state',terminal_contract=plan.get('terminal_contract','terminal_contract_v1'),data_role='random TRAIN; fixed six-route DEV; no independent holdout',resume_parent=None if resume is None else str(Path(resume).resolve()))
    manifest['execution_optimizations']=dict(evaluator='vmap_scan50',compilation_cache=str(compilation_cache),ppo='existing_direct_command_ppo',actual_batch_dimensions=[n,steps],experimental_summary_reset_guard=False)
    if coverage_mode or reward_fork:manifest['fork']=fork_receipt
    atomic_json(root/'manifest.json',manifest)
    try:tb=_start_tensorboard(root)
    except (OSError,RuntimeError) as e:
        tb=dict(url=None,scalar_verified=False,warning=str(e));atomic_json(root/'tensorboard_server_warning.json',tb)
        if plan.get('http_failure_policy')=='error':raise
    status={'state':'initializing','pid':os.getpid(),'endpoint_updates':{'alpha0':0,'alpha1':0},'declared_updates_per_endpoint':updates,'policy_transitions_total':0,'training_control_ticks_total':0,'validation_control_ticks_total':0}
    if conditional:status.update(policy_mode=plan['policy_mode'],endpoint_updates={},declared_updates=updates,engineering_check=engineering,discard_engineering_update=engineering)
    def update_status(**kw):
        status.update(kw)
        if conditional and 'completed_updates' in status:status['shared_updates']=status['completed_updates']
        elif 'alpha' in status and 'completed_updates' in status:status.setdefault('endpoint_updates',{})[f"alpha{status['alpha']}"]=status['completed_updates']
        status['last_update_epoch']=time.time();atomic_json(root/'status.json',status)
    def account_validation(ticks):status['validation_control_ticks_total']+=ticks
    shared_eval_cache={}
    baseline_done=False;best_panels={};writer=None;algo=None;out=None;episodes=None;learner_ready=False
    if resume:
        parent_status=json.loads((Path(resume)/'status.json').read_text())
        status['parent_consumed_policy_transitions']=parent_status.get('policy_transitions_total',0)+parent_status.get('parent_consumed_policy_transitions',0)
        if conditional:status['parent_consumed_validation_control_ticks']=parent_status.get('validation_control_ticks_total',0)+parent_status.get('parent_consumed_validation_control_ticks',0)
        status['parent_reported_updates']=parent_status.get('completed_updates',0)
    update_status(stage='loading')
    try:
        for alpha in ([None] if conditional else plan['alphas']):
            learner_name='shared' if conditional else f'alpha{alpha}'
            endpoint_target=updates
            endpoint_resume=bool(resume and (Path(resume)/learner_name/'last_completed.json').exists())
            if coverage_mode and endpoint_resume:
                for decision_file in sorted((Path(resume)/f'alpha{alpha}'/'decisions').glob('update_*.json')):
                    decision=json.loads(decision_file.read_text())
                    if decision['allow'] and decision['previous_target']==endpoint_target and decision['next_target']==endpoint_target+50:
                        endpoint_target=decision['next_target']
                if endpoint_target>200:raise ValueError('resume target exceeds authorized coverage maximum')
            algo=None;learner_ready=False;started=time.monotonic();update_status(stage='compiling',alpha=alpha,completed_updates=0);out=root/learner_name;out.mkdir();writer=SummaryWriter(str(root/'tensorboard'/learner_name))
            def check_budget():
                status['endpoint_elapsed_s']=time.monotonic()-started
            def to_torch(a,cr):return TensorDict({'policy':torch.utils.dlpack.from_dlpack(a),'critic':torch.utils.dlpack.from_dlpack(cr)},batch_size=[n])
            seed=plan['seed']+(0 if conditional else alpha);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            batch=PathBatch(env,snapshot,cfg,plan['seed'],alpha)
            sample=batch.reset(j.int32(0),j.int32(0));env.set_log_template(sample.state,sample.route.path,sample.route.speed)
            strong=is_strong(cfg)
            reset=jax.jit(jax.vmap(batch.reset,in_axes=(0,0,None))) if strong else jax.jit(jax.vmap(batch.reset))
            def reset_batch(counters,absolute_update):
                args=(j.arange(n,dtype=j.int32),counters)
                return reset(*args,j.int32(absolute_update)) if strong else reset(*args)
            episodes=reset_batch(j.zeros(n,j.int32),1);jax.block_until_ready(episodes)
            observe=jax.jit(jax.vmap(batch.observation))
            a,cr,fault=observe(episodes)
            if bool(j.any(fault)):raise RuntimeError('reset observation fault')
            status['compile_and_setup_seconds']=time.monotonic()-started
            obs=to_torch(a,cr);algo=make_algorithm(obs,steps,cfg,'cuda')
            identity=dict(schema=SCHEMA,alpha=alpha,lower_actor_sha256=source['candidate']['actor_sha256'],config_sha256=config_hash(cfg),sampler_sha256=config_hash(plan['sampler_resolution']),best_sha256=config_hash(plan['best_resolution']),seed=plan['seed'])
            if conditional:
                from .path_command_conditioning import conditional_identity
                identity=conditional_identity(plan,cfg,source)
            if coverage_mode or reward_fork:
                identity.update(parent_source_update=fork_receipt['identities'][f'alpha{alpha}']['source_update'],parent_checkpoint_sha256=fork_receipt['identities'][f'alpha{alpha}']['checkpoint_sha256'])
            start_update=0;rejections=0
            if coverage_mode and not endpoint_resume:
                saved_parent=restore_fork(algo,fork_receipt,alpha)
                rejections=saved_parent['consecutive_rejections']
                atomic_json(out/'fork_receipt.json',dict(parent=fork_receipt['identities'][f'alpha{alpha}'],restored=['actor','critic','log_std','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng'],new_update=0,route_seed=cfg['training_sampler']['route_seed'],route_counters='new sampler version starts at episode0',physical_state='original prepared bank reset; not frame continuation'))
            if reward_fork and not endpoint_resume:
                initialization=restore_reward_fork(algo,fork_receipt,alpha)
                atomic_json(out/'fork_receipt.json',dict(parent=fork_receipt['identities'][f'alpha{alpha}'],initialization=initialization,physical_state='complete prepared bank reset',route_counters='new episode0'))
            if conditional and not endpoint_resume:
                from .path_command_conditioning import initialize_conditional_actor
                init=plan['initialization'];initialization=initialize_conditional_actor(algo,init['checkpoint'],init['checkpoint_sha256'],expected_lower_sha=source['candidate']['actor_sha256'],config=cfg)
                atomic_json(out/'fork_receipt.json',dict(initialization=initialization,physical_state='complete prepared bank reset',route_counters='new episode0'))
            if endpoint_resume:
                receipt,parent_best=resume_source(resume,alpha,identity)
                saved=restore_checkpoint(receipt['checkpoint'],algo,identity);
                if conditional and saved['update']!=receipt['update']:raise ValueError('shared resume update receipt mismatch')
                if parent_best:
                    if conditional:
                        from .path_command_selection import persist_conditional_best
                        persist_conditional_best(out,parent_best['checkpoint'],parent_best['actor'],parent_best['metrics'],{**identity,'source_update':parent_best['source_update'],'validation_dir':parent_best['validation_dir']})
                    else:persist_best(out,parent_best['checkpoint'],parent_best['actor'],parent_best['metrics'],{**identity,'source_update':parent_best['source_update']})
                if strong:
                    from .path_command_selection import persist_preference_best
                    preference_file=Path(resume)/'shared/best_preference_model.json'
                    if preference_file.exists():
                        incumbent=json.loads(preference_file.read_text())
                        if any(incumbent.get(k)!=v for k,v in identity.items()) or incumbent['source_update']>saved['update']:raise ValueError('resume preference best identity mismatch')
                        for key in ('checkpoint','actor'):
                            if digest(incumbent[key])!=incumbent[key+'_sha256']:raise ValueError('resume preference best SHA mismatch')
                        persist_preference_best(out,incumbent['checkpoint'],incumbent['actor'],incumbent['metrics'],{**identity,'source_update':incumbent['source_update'],'validation_dir':incumbent['validation_dir']})
                start_update=saved['update'];rejections=saved['consecutive_rejections']
                if start_update>endpoint_target:raise ValueError('resume boundary exceeds declared target')
                if saved.get('route_counters') is None or np.shape(saved['route_counters'])!=(n,):raise ValueError('resume requires per-environment route counters')
                route_counters=np.asarray(saved['route_counters'])
                fault_file=Path(resume)/learner_name/'engineering_fault_states.pkl'
                if fault_file.exists():
                    with fault_file.open('rb') as f:failed_episodes=pickle.load(f)
                    route_counters=np.maximum(route_counters,np.asarray(failed_episodes.episode))
                episodes=reset_batch(j.asarray(route_counters)+1,start_update+1)
                atomic_json(out/'resume_receipt.json',dict(parent_checkpoint=receipt,parent_terminal_contract=saved.get('terminal_contract','terminal_contract_v1'),terminal_contract=plan.get('terminal_contract'),restored_update=start_update,parent_reported_updates=status.get('parent_reported_updates'),physical_state='reset to prepared bank',routes='next unused after saved and fault snapshot counters'))
                a,cr,fault=observe(episodes)
                if bool(j.any(fault)):raise RuntimeError('resume observation fault')
                obs=to_torch(a,cr)
            from .path_command_statistics import reduce_statistics,merge_statistics,reset_statistics
            def advance_batch(episodes,z,absolute_update=1):
                if strong:nxt,obs_batch,reward,flags,final,stats=jax.vmap(batch.advance_one,in_axes=(0,0,None))(episodes,z,absolute_update)
                else:nxt,obs_batch,reward,flags,final,stats=jax.vmap(batch.advance_one)(episodes,z)
                stats['diagnostics']=reduce_statistics(stats['diagnostics'],axis=0)
                return nxt,obs_batch,reward,flags,final,stats
            merge_diagnostics=jax.jit(merge_statistics)
            advance_args=(episodes,j.zeros((n,2)),j.int32(start_update+1)) if strong else (episodes,j.zeros((n,2)))
            advance=None if start_update>=endpoint_target else jax.jit(advance_batch).lower(*advance_args).compile()
            status['compile_and_setup_seconds']=time.monotonic()-started
            initial_checkpoint=save_checkpoint(out/'checkpoints',algo,start_update,identity,rejections=rejections,route_counters=episodes.episode)
            atomic_json(out/'last_completed.json',dict(update=start_update,checkpoint=str(initial_checkpoint.resolve()),checkpoint_sha256=digest(initial_checkpoint),schema=SCHEMA))
            update_status(completed_updates=start_update)
            learner_ready=True
            if endpoint_resume and not conditional:
                rescore_saved_candidates(resume,out,alpha,cfg,identity)
                if reward_fork:
                    from .path_command_cache import inherit_reward_validation
                    inherit_reward_validation(resume,out,alpha,cfg,identity)
                if coverage_mode or reward_fork:
                    import shutil
                    for folder in ('validation','statistics','coverage','decisions'):
                        prior=Path(resume)/f'alpha{alpha}'/folder
                        if prior.exists() and not (reward_fork and folder=='validation'):shutil.copytree(prior,out/folder)
            parent_traces=None
            if coverage_mode:
                from .path_command_cache import reuse_fork_parent
                parent_traces=reuse_fork_parent(fork_receipt,out,alpha,cfg,identity,initial_checkpoint,register=not endpoint_resume)
            if reward_fork and not endpoint_resume:
                from .path_command_cache import reuse_reward_parent
                from .geometric_path import build_path
                pose=np.asarray(env.physics.helpers.pose(snapshot.data))
                expected_paths={r:build_path(r,pose,cfg) for r in cfg['stage_C_evaluation']['route_ids']}
                parent_traces=reuse_reward_parent(fork_receipt,out,alpha,cfg,identity,initial_checkpoint,expected_paths)
            from .path_command_evaluation import BatchedPanelEvaluator
            evaluator=BatchedPanelEvaluator(env,snapshot,cfg,check_budget,update_status,account_validation)
            evaluator.compiled=shared_eval_cache
            if not baseline_done and not engineering:
                cached=set()
                if conditional:
                    from .path_command_cache import reuse_conditional_baseline
                    from .geometric_path import build_path
                    pose=np.asarray(env.physics.helpers.pose(snapshot.data))
                    paths={route:build_path(route,pose,cfg) for route in cfg['stage_C_evaluation']['route_ids']}
                    cached=reuse_conditional_baseline(plan['baseline_parent'],root,cfg,source,paths)
                    if len(cached)!=6:raise ValueError('conditional baseline requires six reusable cases')
                if (resume or coverage_mode or reward_fork) and not conditional:
                    from .path_command_cache import reuse_zero_upper
                    from .geometric_path import build_path
                    pose=np.asarray(env.physics.helpers.pose(snapshot.data))
                    paths={route:build_path(route,pose,cfg) for route in cfg['stage_C_evaluation']['route_ids']}
                    cache_parent=resume if resume else fork_receipt['parent_run']
                    cache_cfg=cfg if resume else fork_receipt['parent_manifest']['config']
                    if reward_fork:
                        from .path_command_cache import reuse_reward_baseline
                        cached=reuse_reward_baseline(cache_parent,root,cfg,source,paths)
                    else:cached=reuse_zero_upper(cache_parent,root,cache_cfg,source,paths)
                    if (coverage_mode or reward_fork) and len(cached)!=6:raise ValueError('coverage fork requires all six existing baseline traces')
                zero=make_path_actor(cfg).init(jax.random.PRNGKey(0),j.zeros(351));evaluator.evaluate(zero,0,root/'zero_upper',cached_names=cached if (resume or coverage_mode or reward_fork or conditional) else None);baseline_done=True
                status['reused_baseline_cases']=len(cached)
            def evaluate_shared(update,checkpoint):
                from .path_command_evaluation import evaluate_conditional
                from .path_command_selection import summarize_conditional,persist_conditional_best
                destination=out/'validation'/f'update_{update:04d}';panels=evaluate_conditional(evaluator,export_actor(algo.policy),plan['alphas'],destination)
                summary=summarize_conditional(panels,cfg)
                actor=out/'checkpoints'/f'actor_{update:04d}.pkl'
                provenance=dict(identity,source_update=update,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest(checkpoint),actor=str(actor.resolve()),actor_sha256=digest(actor),validation_dir=str(destination.resolve()))
                atomic_json(destination/'source.json',provenance);atomic_json(destination/'task_metrics.json',summary)
                persist_conditional_best(out,checkpoint,actor,summary,{**identity,'source_update':update,'validation_dir':str(destination.resolve())})
                if strong:
                    from .path_command_selection import summarize_preference,persist_preference_best,preference_score_key
                    preference=summarize_preference(panels,cfg)
                    preference['score_tuple']=preference_score_key(preference,update)
                    atomic_json(destination/'preference_metrics.json',preference)
                    persist_preference_best(out,checkpoint,actor,preference,{**identity,'source_update':update,'validation_dir':str(destination.resolve())})
                return summary
            if conditional and not engineering and start_update in validation_schedule:
                dev_started=time.monotonic()
                if endpoint_resume:
                    from .path_command_conditioning import complete_resume_validation
                    status['resume_validation']=complete_resume_validation(resume,out,start_update,initial_checkpoint,identity,cfg,export_actor(algo.policy),evaluate_shared,lambda:status['validation_control_ticks_total'])
                else:evaluate_shared(start_update,initial_checkpoint)
                status['dev_seconds']=time.monotonic()-dev_started
            accepted=0;http_verified=False;stop=None;update=start_update;window_stats=None;window_episodes=0;window_return=0.;window_costs=np.zeros(10);window_physical=0;window_domain=0
            from .path_command_statistics import summarize_statistics,COST_KEYS
            coverage_stats=None;coverage_low=0;coverage_fraction=None;window_start=start_update+1;coverage_start=start_update+1
            if (coverage_mode or conditional) and endpoint_resume:
                previous_coverage=(Path(resume)/learner_name if conditional else out)/'coverage'/f'update_{start_update:04d}.json'
                if previous_coverage.exists():
                    prior=json.loads(previous_coverage.read_text());coverage_low=prior['consecutive_low_windows'];coverage_fraction=prior['windows']['strong_demand']['sample_fraction']
            initial_resets=reduce_statistics(jax.vmap(reset_statistics)(episodes.route),axis=0)
            atomic_json(out/'initial_reset_counts.json',{k:np.asarray(v).tolist() for k,v in initial_resets.items()})
            while update<endpoint_target:
                update+=1
                check_budget();tick_start=time.monotonic();update_status(state='running',stage='sampling',case=None,completed_intervals=None,alpha=alpha,completed_updates=update-1,declared_updates=endpoint_target)
                alpha_counts=np.zeros(3,dtype=np.int64);alpha_demand_counts=np.zeros(6)
                rewards=[];rollout_stats=[];completed_episodes=[];caps={};episode_ends=0;physical_failures=0;domain_failures=0;cost_totals_update=np.zeros(10)
                with torch.no_grad():
                    for _ in range(steps):
                        check_budget()
                        if strong:
                            sampled=obs['policy'][:,336].detach().cpu().numpy()
                            alpha_counts+= [np.sum(sampled==0),np.sum(sampled==1),np.sum((sampled>0)&(sampled<1))]
                        z=algo.act(obs)
                        if not bool(torch.isfinite(z).all()):raise RuntimeError('nonfinite upper latent')
                        advance_args=(episodes,jax.dlpack.from_dlpack(z.detach().contiguous()))
                        if strong:advance_args+= (j.int32(update),)
                        episodes,next_obs,reward,flags,final,stats=advance(*advance_args)
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
                        diagnostic=stats['diagnostics']
                        if strong and 'alpha_strong_count' in diagnostic:alpha_demand_counts+=np.asarray(diagnostic['alpha_strong_count'])
                        window_stats=diagnostic if window_stats is None else merge_diagnostics(window_stats,diagnostic)
                        if coverage_mode or reward_fork or conditional:coverage_stats=diagnostic if coverage_stats is None else merge_diagnostics(coverage_stats,diagnostic)
                        for key,count in stats['component_caps'].items():caps[key]=caps.get(key,0)+int(j.sum(count))
                    algo.compute_returns(obs)
                sample_seconds=time.monotonic()-tick_start;opt_start=time.monotonic();update_status(stage='optimizing');metrics=algo.update();opt_seconds=time.monotonic()-opt_start;check_budget()
                update_status(stage='optimized',completed_updates=update)
                rejections,stop=handle_candidate_ppo_result(algo,metrics,rejections,plan);accepted+=int(metrics['accepted_epochs']>0)
                totals=np.sum(np.stack(rollout_stats),axis=(0,1));done_totals=np.sum(np.stack(completed_episodes),axis=(0,1));count=max(totals[0],1)
                record=dict(update=update,sampling_model_update=update-1,accepted_batches_this_stage=accepted,rejected_streak=rejections,optimizer=clean_numbers(metrics),mean_step_reward=float(np.mean(rewards)),reward_unit='policy interval .02s sum of four lower rewards including failure replacement',rollout_speed_rmse=float(np.sqrt(totals[1]/count)),rollout_path_rmse=float(np.sqrt(totals[2]/count)),rollout_heading_rmse=float(np.sqrt(totals[3]/count)),complete_episode_count=episode_ends,complete_episode_totals=done_totals.tolist(),physical_failures=physical_failures,tracking_domain_failures=domain_failures,engineering_faults=0,complete_episode_return_mean=None if not episode_ends else float(done_totals[4]/episode_ends),complete_episode_cost_integral_means=None if not episode_ends else dict(zip(COST_KEYS,(cost_totals_update/episode_ends).tolist())),log_std=algo.policy.log_std.detach().cpu().tolist(),learning_rates=[g['lr'] for g in algo.optimizer.param_groups],terminal_contract=env.terminal_contract,physical_steps=status['training_control_ticks_total'],policy_steps=status['policy_transitions_total'],component_cap_counts=caps,sample_seconds=sample_seconds,optimize_seconds=opt_seconds,stop_reason=stop)
                if strong:
                    from .path_command_conditioning import curriculum_probabilities
                    record.update(curriculum_stage=1 if update<=cfg['alpha_conditioning']['curriculum'][0]['updates_inclusive'][1] else 2,absolute_update=update,endpoint_strong_demand_lower_samples={'alpha0':float(alpha_demand_counts[0]),'alpha1':float(alpha_demand_counts[-1])},reset_alpha_probabilities=np.asarray(curriculum_probabilities(cfg,update)).tolist(),actual_alpha_policy_samples=dict(zip(('alpha0','alpha1','interior'),alpha_counts.tolist())),actual_alpha_policy_proportions=(alpha_counts/max(alpha_counts.sum(),1)).tolist())
                    for label in ('alpha0','alpha1'):writer.add_scalar('curriculum/'+label+'_strong_demand_lower_samples',record['endpoint_strong_demand_lower_samples'][label],update)
                    for label,count in zip(('alpha0','alpha1','interior'),alpha_counts):writer.add_scalar('curriculum/'+label+'_samples',int(count),update)
                    writer.add_scalar('curriculum/stage',record['curriculum_stage'],update)
                if coverage_mode:record.update(new_update=update,lineage_update=identity['parent_source_update']+update,parent_source_update=identity['parent_source_update'])
                window_episodes+=episode_ends;window_return+=done_totals[4];window_costs+=cost_totals_update;window_physical+=physical_failures;window_domain+=domain_failures
                if update%25==0 or update==endpoint_target:
                    diagnostic=summarize_statistics(window_stats,cfg)
                    diagnostic.update(update=update,complete_episode_count=window_episodes,complete_episode_return_mean=None if not window_episodes else float(window_return/window_episodes),complete_episode_cost_integral_means=None if not window_episodes else dict(zip(COST_KEYS,(window_costs/window_episodes).tolist())),physical_failure_count=window_physical,tracking_domain_failure_count=window_domain,engineering_fault_count=0,physical_failure_rate=None if not window_episodes else window_physical/window_episodes,tracking_domain_failure_rate=None if not window_episodes else window_domain/window_episodes)
                    diagnostic.update(window_updates=update-window_start+1,engineering_failure_rate=0.,domain_failure_rate=window_domain/max(diagnostic['active_lower_samples'],1))
                    atomic_json(out/'statistics'/f'update_{update:04d}.json',diagnostic)
                    for group_name,group in diagnostic.get('alpha_groups',{}).items():
                        for metric in ('strong_demand_samples','samples','reward_mean','speed_rmse_m_s','lateral_rmse_m','heading_rmse_rad','strong_demand_fraction'):
                            if group.get(metric) is not None:writer.add_scalar(f'alpha/{group_name}/{metric}',group[metric],update)
                    window_start=update+1
                    window_stats=None;window_episodes=0;window_return=0.;window_costs=np.zeros(10);window_physical=0;window_domain=0
                coverage_pause=False
                if (coverage_mode or reward_fork or conditional) and update%10==0:
                    coverage_summary=summarize_statistics(coverage_stats,cfg);coverage_stats=None
                    coverage_fraction=coverage_summary['windows']['strong_demand']['sample_fraction']
                    complete_window=update-coverage_start+1==10
                    if complete_window:coverage_low=coverage_low+1 if coverage_fraction<plan['coverage_monitor']['target'] else 0
                    coverage_summary.update(new_update=update,consecutive_low_windows=coverage_low,window_updates=update-coverage_start+1)
                    coverage_start=update+1
                    atomic_json(out/'coverage'/f'update_{update:04d}.json',coverage_summary)
                    writer.add_scalar('coverage/strong_active_fraction',coverage_fraction,update)
                    coverage_pause=complete_window and coverage_low>=plan['coverage_monitor']['consecutive_low_windows']
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                writer.add_scalar('train/mean_step_reward',record['mean_step_reward'],update)
                for key in ['rollout_speed_rmse','rollout_path_rmse','rollout_heading_rmse','complete_episode_count','physical_failures','tracking_domain_failures']:writer.add_scalar('train/'+key,record[key],update)
                for key in ['surrogate','value','entropy','mean_kl','accepted_epochs']:writer.add_scalar('ppo/'+key,float(metrics[key]),update)
                for channel,value in enumerate(algo.policy.log_std.detach().cpu().tolist()):writer.add_scalar(f'ppo/std_{channel}',float(np.exp(value)),update)
                for group,g in enumerate(algo.optimizer.param_groups):writer.add_scalar(f'ppo/lr_{group}',g['lr'],update)
                writer.flush()
                if update%c['save_every']==0 or update==endpoint_target or stop:
                    checkpoint=save_checkpoint(out/'checkpoints',algo,update,identity,rejections=rejections,route_counters=episodes.episode)
                    atomic_json(out/'last_completed.json',dict(update=update,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest(checkpoint),schema=SCHEMA))
                if not http_verified and (update==start_update+1 or update%25==0):http_verified=_verify_reward_http(root,tb,learner_name)
                if plan.get('http_failure_policy')=='error' and not http_verified:raise RuntimeError('current TensorBoard reward scalar not verified')
                update_status(stage='optimized',completed_updates=update,accepted_batches=accepted,last_reward=record['mean_step_reward'])
                if coverage_pause:
                    atomic_json(out/'coverage_diagnosis.json',dict(reason='two consecutive complete ten-update windows below exposure target',sampler=cfg['training_sampler'],actual=coverage_summary,domain_failures_last_rollout=domain_failures,policy='save and pause; no silent sampler/reward/lower change'))
                    raise CoveragePause('strong demand exposure below target in two windows; saved boundary and diagnosis')
                if conditional and update in validation_schedule and not stop:
                    dev_started=time.monotonic();evaluate_shared(update,checkpoint);status['dev_seconds']=status.get('dev_seconds',0)+time.monotonic()-dev_started
                if not conditional and update in validation_schedule and not stop:
                    dev_started=time.monotonic();params=export_actor(algo.policy);traces=evaluator.evaluate(params,alpha,out/'validation'/f'update_{update:04d}');summary=summarize(traces,alpha,cfg)
                    atomic_json(out/'validation'/f'update_{update:04d}'/'metrics.json',summary)
                    persist_best(out,checkpoint,out/'checkpoints'/f'actor_{update:04d}.pkl',summary,dict(identity,source_update=update))
                    status['dev_seconds']=status.get('dev_seconds',0)+time.monotonic()-dev_started
                    task_summary=summarize_task(traces,alpha,cfg)
                    atomic_json(out/'validation'/f'update_{update:04d}'/'task_metrics.json',task_summary)
                    atomic_json(out/'validation'/f'update_{update:04d}'/'source.json',dict(identity,source_update=update,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest(checkpoint),actor=str((out/'checkpoints'/f'actor_{update:04d}.pkl').resolve()),actor_sha256=digest(out/'checkpoints'/f'actor_{update:04d}.pkl')))
                    persist_task_best(out,checkpoint,out/'checkpoints'/f'actor_{update:04d}.pkl',task_summary,dict(identity,source_update=update))
                if coverage_mode and update==endpoint_target and update<plan['conditional_extension']['hard_max_updates'] and not stop:
                    previous=json.loads((out/'statistics'/f'update_{update-25:04d}.json').read_text());latest=json.loads((out/'statistics'/f'update_{update:04d}.json').read_text())
                    decision=extension_decision(fork_receipt['identities'][f'alpha{alpha}']['metrics'],task_summary,parent_traces,traces,previous,latest,coverage_fraction,alpha,cfg)
                    decision.update(new_update=update,previous_target=endpoint_target)
                    if decision['allow']:endpoint_target+=plan['conditional_extension']['block']
                    decision['next_target']=endpoint_target;atomic_json(out/'decisions'/f'update_{update:04d}.json',decision)
                    update_status(declared_updates=endpoint_target)
                print(f'{learner_name} update{update}/{endpoint_target} reward={record["mean_step_reward"]:.6g} accepted={accepted}',flush=True)
                if stop:break
            atomic_json(out/'training_status.json',dict(state='completed' if not stop else 'error',completed_updates=update,declared_updates=endpoint_target,stop_reason=stop))
            writer.close();notify('STTW path training stage ended',f'{learner_name}: {update}/{endpoint_target}; {stop or "budget reached"}')
            if stop:raise RuntimeError(stop)
            if engineering:
                atomic_json(out/'engineering_receipt.json',dict(state='completed',discarded_update=True,training_policy_transitions=status['policy_transitions_total'],training_lower_ticks=status['training_control_ticks_total'],validation_lower_ticks=0,eligible_as_parent=False))
                continue
            if conditional:
                from .path_command_evaluation import reuse_conditional_best
                best=json.loads((out/('best_preference_model.json' if strong else 'best_conditional_model.json')).read_text());best_panels['shared']=reuse_conditional_best(out,best,cfg)
                if strong:
                    from .path_command_dynamic import run_dynamic_preference_checks
                    run_dynamic_preference_checks(root,cfg,best,evaluator=evaluator,resume_parent=resume)
                continue
            if not (out/task_pointer_name(cfg)).exists():raise RuntimeError('no fixed validation candidate; no last-as-best fallback')
            best=json.loads((out/task_pointer_name(cfg)).read_text())
            if digest(best['actor'])!=best['actor_sha256'] or digest(best['checkpoint'])!=best['checkpoint_sha256']:raise ValueError('best task weights identity changed')
            with Path(best['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
            cached_final=set()
            if coverage_mode or reward_fork:
                from .path_command_cache import reuse_selected_validation
                cached_final=reuse_selected_validation(out,best,cfg)
            elif resume:
                from .path_command_cache import reuse_final_panel, inherit_numerical_history
                cached_final=reuse_final_panel(resume,out,alpha,cfg,source,best,paths)
                inherit_numerical_history(resume,out,alpha)
            update_status(reused_final_cases=len(cached_final))
            best_panels[alpha]=evaluator.evaluate(params,alpha,out/'final_best',cached_names=cached_final if cached_final else None);atomic_json(out/'final_best'/'metrics.json',summarize_task(best_panels[alpha],alpha,cfg));atomic_json(out/'final_best'/'source.json',best)
        for p,sha in protected.items():
            if digest(p)!=sha:raise RuntimeError('protected source changed')
        if conditional:
            if not engineering:
                from .path_command_reporting import report_conditional_panel
                report_conditional_panel(root,cfg)
        else:
            from .path_command_reporting import report_final_panel
            report_final_panel(root,cfg)
            from .path_command_reporting import convergence_decision
            paired=json.loads((root/'report'/'paired_metrics.json').read_text());decisions={}
            qualified=all(json.loads((root/f'alpha{a}'/task_pointer_name(cfg)).read_text())['qualified_task'] for a in plan['alphas'])
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
            if coverage_mode:
                from .path_command_reporting import report_coverage_parent_comparison
                report_coverage_parent_comparison(root,cfg,fork_receipt)
        update_status(state='completed',stage='reported',protected_unchanged=True);notify('STTW path pipeline completed',str(root))
    except BaseException as e:
        if learner_ready and algo is not None and status.get('stage')!='optimizing':
            completed=int(status.get('completed_updates',0));checkpoint=out/'checkpoints'/f'update_{completed:04d}.pt'
            if not checkpoint.exists():checkpoint=save_checkpoint(out/'checkpoints',algo,completed,identity,rejections=rejections,route_counters=episodes.episode)
            atomic_json(out/'last_completed.json',dict(update=completed,checkpoint=str(checkpoint.resolve()),checkpoint_sha256=digest(checkpoint),schema=SCHEMA))
        if writer is not None:writer.close()
        update_status(state='paused' if isinstance(e,CoveragePause) else 'error',error_type=type(e).__name__,message=str(e));notify('STTW path error',str(e));raise
