"""Episode-level alpha sampling and actor-only initialization for one PPO learner."""
from pathlib import Path
import jax
import jax.numpy as j
from .preference_best import config_hash,digest

MODE='shared_conditional_v1'
PLAN_SCHEMA='sttw_path_conditional_training_plan_v1'
ALPHA_INDEX=336
ALPHA_GRID=[0.,.25,.5,.75,1.]


def is_conditional(plan):
    return plan.get('policy_mode')==MODE


def sample_episode_alpha(seed,env_id,episode):
    """Separate purpose/seed stream; the sampled value is held in PathState."""
    key=jax.random.fold_in(jax.random.PRNGKey(seed),j.uint32(0xA17A))
    key=jax.random.fold_in(jax.random.fold_in(key,env_id),episode)
    draw=jax.random.uniform(key,(),dtype=j.float32)
    return j.where(draw<.25,j.float32(0),j.where(draw<.5,j.float32(1),(draw-.5)*2))


def execution_dimensions(plan,config):
    if is_conditional(plan) and plan.get('engineering_check',False):return 8,16
    c=config['training_future_phase_C'];return c['num_envs'],c['rollout_steps']


def validate_conditional_plan(plan,config):
    c=config['training_future_phase_C'];a=config.get('alpha_conditioning',{})
    if plan.get('schema')!=PLAN_SCHEMA or not is_conditional(plan):raise ValueError('conditional plan schema/mode mismatch')
    expected=dict(mode=MODE,alpha_index=336,endpoint_probability_each=.25,uniform_probability=.5,constant_per_episode=True,evaluation_alphas=ALPHA_GRID)
    if {k:v for k,v in a.items() if k!='seed'}!=expected or config['network'].get('policy_type')!=MODE:raise ValueError('conditional alpha contract mismatch')
    seeds=[plan.get('seed'),a.get('seed')]
    if any(type(seed) is not int or not 0<=seed<2**32 for seed in seeds) or a['seed'] in (plan['seed'],config['training_sampler']['route_seed']):raise ValueError('independent integer alpha seed required')
    network=config['network']
    if network.get('actor_dim')!=351 or network.get('critic_dim')!=352 or network.get('hidden')!=[128,128,64] or network.get('activation')!='elu':raise ValueError('conditional network351/352 contract mismatch')
    if plan.get('alphas')!=ALPHA_GRID or plan.get('task_selection_schema')!='path_conditional_task_v1' or config.get('task_selection_schema')!='path_conditional_task_v1':raise ValueError('conditional evaluation/selection mismatch')
    if plan.get('automatic_extension') or plan.get('conditional_extension'):raise ValueError('conditional budget cannot extend')
    if plan.get('sampler_version')!='demand_exposure_v2' or plan['sampler_resolution'].get('training_sampler')!=config.get('training_sampler'):raise ValueError('conditional sampler disconnected')
    if plan.get('terminal_contract')!='terminal_contract_v2' or plan.get('http_failure_policy')!='error':raise ValueError('conditional terminal/visibility contract mismatch')
    if config['path_reward'].get('reward_version')!='path_safety_preference_v1':raise ValueError('conditional reward version mismatch')
    if c['num_envs']!=512 or c['rollout_steps']!=128 or c['episode_s']!=20 or c['validation_updates']!=[0,50,100,150,200]:raise ValueError('conditional training dimensions/schedule mismatch')
    if plan.get('engineering_check',False):
        if plan['updates']!=1 or plan.get('engineering_dimensions')!=[8,16] or not plan.get('discard_engineering_update'):raise ValueError('engineering check must discard exactly 8x16 update')
    elif plan['updates']!=200:raise ValueError('conditional formal budget is 200 updates')
    init=plan.get('initialization',{})
    if init.get('mode')!='actor_only_zero_alpha_column' or init.get('parent_alpha')!=0:raise ValueError('conditional actor-only alpha0 initialization required')
    if not init.get('checkpoint') or len(init.get('checkpoint_sha256',''))!=64:raise ValueError('conditional parent identity missing')
    if plan.get('coverage_monitor')!={'target':.05,'check_every':10,'consecutive_low_windows':2}:raise ValueError('conditional exposure coverage monitor mismatch')


def initialize_conditional_actor(algo,parent_checkpoint,expected_sha,*,expected_lower_sha=None):
    import torch
    if digest(parent_checkpoint)!=expected_sha:raise ValueError('parent checkpoint SHA mismatch')
    if algo.optimizer.state_dict()['state']:raise ValueError('conditional initialization requires fresh Adam')
    saved=torch.load(parent_checkpoint,map_location=algo.device,weights_only=False)
    from .path_command_policy import SCHEMA
    if saved.get('schema')!=SCHEMA or saved.get('identity',{}).get('alpha')!=0:raise ValueError('conditional parent must be geometric alpha0')
    if expected_lower_sha is None or saved['identity'].get('lower_actor_sha256')!=expected_lower_sha:raise ValueError('conditional parent lower identity mismatch')
    current=algo.policy.state_dict();parent=saved['policy'];keys={k for k in current if k.startswith('actor.')}
    if keys!={k for k in parent if k.startswith('actor.')}:raise ValueError('conditional parent actor keys mismatch')
    for key in keys:
        if parent[key].shape!=current[key].shape or not torch.isfinite(parent[key]).all():raise ValueError('invalid conditional parent actor tensor '+key)
    first_name,first=next((name,module) for name,module in algo.policy.actor.named_modules() if isinstance(module,torch.nn.Linear))
    if first.in_features!=351:raise ValueError('conditional actor schema351 required')
    for key in keys:current[key]=parent[key].to(device=algo.device,dtype=current[key].dtype).clone()
    current[f'actor.{first_name}.weight'][:,ALPHA_INDEX]=0
    algo.policy.load_state_dict(current,strict=True)
    return dict(mode='actor_only_zero_alpha_column',parent_checkpoint=str(Path(parent_checkpoint).resolve()),parent_checkpoint_sha256=expected_sha,
        restored=sorted(keys),zeroed_alpha_input_column=ALPHA_INDEX,critic='fresh',log_std='configured_fresh',optimizer='fresh',rng='fresh',update=0)


def conditional_identity(plan,config,source):
    from .path_command_policy import SCHEMA
    return dict(schema=SCHEMA,policy_mode=MODE,alpha_conditioning_sha256=config_hash(config['alpha_conditioning']),
        lower_actor_sha256=source['candidate']['actor_sha256'],source_sha256=config_hash(source),config_sha256=config_hash(config),
        sampler_sha256=config_hash(plan['sampler_resolution']),best_sha256=config_hash(plan['best_resolution']),seed=plan['seed'],
        parent_checkpoint_sha256=plan['initialization']['checkpoint_sha256'],engineering_check=bool(plan.get('engineering_check',False)))


def complete_resume_validation(parent,out,update,checkpoint,identity,config,params,evaluate,tick_count):
    """Finish one saved learner boundary before advancing PPO after interruption.

    Complete panels remain in their immutable parent directory. An incomplete
    panel is re-evaluated at this exact restored model in the new run directory;
    persisted repeated ticks and all newly executed ticks are recorded explicitly.
    """
    import json
    import numpy as np
    from .preference_best import atomic_json
    from .path_command_evaluation import _conditional_parameter_digest
    from .path_command_selection import cases,alpha_key,conditional_grid,summarize_conditional,persist_conditional_best
    boundary=Path(parent)/'shared/validation'/f'update_{update:04d}'
    grid={alpha_key(a):a for a in conditional_grid(config)}
    required=[boundary/key/(name+suffix) for key in grid for name,_,_ in cases(config) for suffix in ('.npz','_path.npz')]
    prior_ticks=0
    for key in grid:
        for name,_,_ in cases(config):
            trace=boundary/key/(name+'.npz')
            if trace.exists():
                with np.load(trace) as data:prior_ticks+=len(data['time'])
    meta_file=boundary/'panel_identity.json';source_file=boundary/'source.json'
    meta=json.loads(meta_file.read_text()) if meta_file.exists() else None
    expected=dict(schema='path_conditional_panel_v1',mode=MODE,alphas=grid,
        config_sha256=config_hash(config),actor_parameters_sha256=_conditional_parameter_digest(params))
    if meta is not None and any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('resume conditional panel identity mismatch')
    source=json.loads(source_file.read_text()) if source_file.exists() else None
    if source is not None:
        if source.get('source_update')!=update or any(source.get(k)!=v for k,v in identity.items()):raise ValueError('resume conditional source identity mismatch')
        for kind in ('checkpoint','actor'):
            if digest(source[kind])!=source.get(kind+'_sha256'):raise ValueError('resume conditional source SHA mismatch')
    complete=bool(meta and meta.get('completed') and source and all(p.exists() for p in required))
    receipt=dict(source_update=update,parent_validation_dir=str(boundary.resolve()),
        action='inherit_complete' if complete else 'evaluate_saved_boundary',
        parent_persisted_boundary_lower_ticks=prior_ticks,new_validation_lower_ticks=0,
        repeated_saved_lower_ticks=0 if complete else prior_ticks,
        repeated_tick_scope='persisted parent trace lower bound; parent run totals are carried in status',
        maximum_new_validation_lower_ticks=0 if complete else len(grid)*len(cases(config))*round(config['training_future_phase_C']['episode_s']/config['timing']['lower_dt_s']),completed=False)
    receipt_path=Path(out)/'resume_validation.json';atomic_json(receipt_path,receipt)
    before=tick_count()
    try:
        if complete:
            panels={key:{name:dict(np.load(boundary/key/(name+'.npz'))) for name,_,_ in cases(config)} for key in grid}
            summary=summarize_conditional(panels,config)
            persist_conditional_best(out,source['checkpoint'],source['actor'],summary,{**identity,'source_update':update,'validation_dir':str(boundary.resolve())})
        else:evaluate(update,checkpoint)
        receipt['completed']=True
    finally:
        receipt['new_validation_lower_ticks']=int(tick_count()-before)
        atomic_json(receipt_path,receipt)
    return receipt
