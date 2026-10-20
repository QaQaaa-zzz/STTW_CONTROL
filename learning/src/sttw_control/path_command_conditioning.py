"""Episode-level alpha sampling and actor-only initialization for one PPO learner."""
from pathlib import Path
import jax
import jax.numpy as j
from .path_command_policy import config_key

MODE='shared_conditional_v1'
PLAN_SCHEMA='sttw_path_conditional_training_plan_v1'
ALPHA_INDEX=336
ALPHA_GRID=[0.,.25,.5,.75,1.]
SCRATCH_SCHEMA='sttw_path_scratch_preference_training_plan_v1'
CONFLICT_SCHEMA='sttw_path_conflict_exposure_training_plan_v1'
STRONG_SCHEMA='sttw_path_strong_preference_training_plan_v1'
CURRICULUM=[dict(updates_inclusive=[1,100],endpoint_probability_each=.5,uniform_probability=0.),dict(updates_inclusive=[101,200],endpoint_probability_each=.4,uniform_probability=.2)]

def is_strong(config):
    return config.get('network',{}).get('architecture')=='alpha_modulated_v1'

def curriculum_probabilities(config,absolute_update):
    a=config['alpha_conditioning']
    if a.get('sampling_mode')=='fixed_mixture' or not is_strong(config):return j.asarray([a['endpoint_probability_each']]*2+[a['uniform_probability']],j.float32)
    stages=a['curriculum']
    return j.where(j.asarray(absolute_update)<=stages[0]['updates_inclusive'][1],
        j.asarray([stages[0]['endpoint_probability_each']]*2+[stages[0]['uniform_probability']],j.float32),
        j.asarray([stages[1]['endpoint_probability_each']]*2+[stages[1]['uniform_probability']],j.float32))


def is_conditional(plan):
    return plan.get('policy_mode')==MODE


def sample_episode_alpha(seed,env_id,episode,config=None,absolute_update=1):
    """Separate purpose/seed stream; the sampled value is held in PathState."""
    key=jax.random.fold_in(jax.random.PRNGKey(seed),j.uint32(0xA17A))
    key=jax.random.fold_in(jax.random.fold_in(key,env_id),episode)
    draw=jax.random.uniform(key,(),dtype=j.float32)
    p=j.asarray([.25,.25,.5]) if config is None else curriculum_probabilities(config,absolute_update)
    return j.where(draw<p[0],j.float32(0),j.where(draw<p[0]+p[1],j.float32(1),(draw-p[0]-p[1])/j.maximum(p[2],1e-8)))


def execution_dimensions(plan,config):
    if is_conditional(plan) and plan.get('engineering_check',False):return 8,16
    c=config['training_future_phase_C'];return c['num_envs'],c['rollout_steps']


def validate_conditional_plan(plan,config):
    if plan.get('schema')==CONFLICT_SCHEMA:return validate_conflict_plan(plan,config)
    c=config['training_future_phase_C'];a=config.get('alpha_conditioning',{})
    from .path_command_policy import architecture_identity
    architecture_identity(config)
    strong=is_strong(config)
    scratch=plan.get('schema')==SCRATCH_SCHEMA
    if plan.get('schema')!=(SCRATCH_SCHEMA if scratch else STRONG_SCHEMA if strong else PLAN_SCHEMA) or not is_conditional(plan):raise ValueError('conditional plan schema/mode mismatch')
    expected=dict(mode=MODE,alpha_index=336,endpoint_probability_each=.25,uniform_probability=.5,constant_per_episode=True,evaluation_alphas=ALPHA_GRID)
    if scratch:
        expected.update(sampling_mode='fixed_mixture',endpoint_probability_each=.4,uniform_probability=.2)
        if not strong:raise ValueError('scratch requires alpha_modulated_v1 architecture')
    elif strong:
        expected.pop('endpoint_probability_each');expected.pop('uniform_probability');expected['curriculum']=CURRICULUM
    if {k:v for k,v in a.items() if k!='seed'}!=expected or config['network'].get('policy_type')!=MODE:raise ValueError('conditional alpha contract mismatch')
    seeds=[plan.get('seed'),a.get('seed')]
    if any(type(seed) is not int or not 0<=seed<2**32 for seed in seeds) or a['seed'] in (plan['seed'],config['training_sampler']['route_seed']):raise ValueError('independent integer alpha seed required')
    network=config['network']
    if network.get('actor_dim')!=351 or network.get('critic_dim')!=352 or network.get('hidden')!=[128,128,64] or network.get('activation')!='elu':raise ValueError('conditional network351/352 contract mismatch')
    if plan.get('alphas')!=ALPHA_GRID or plan.get('task_selection_schema')!='path_conditional_task_v1' or config.get('task_selection_schema')!='path_conditional_task_v1':raise ValueError('conditional evaluation/selection mismatch')
    if plan.get('automatic_extension') or plan.get('conditional_extension'):raise ValueError('conditional budget cannot extend')
    if plan.get('sampler_version')!='demand_exposure_v2' or plan['sampler_resolution'].get('training_sampler')!=config.get('training_sampler'):raise ValueError('conditional sampler disconnected')
    if plan.get('terminal_contract')!='terminal_contract_v2' or plan.get('http_failure_policy')!='error':raise ValueError('conditional terminal/visibility contract mismatch')
    from .path_command_reward import reward_version
    reward_version(config)  # Reward choice is independent of alpha modulation.
    if c['num_envs']!=512 or c['rollout_steps']!=128 or c['episode_s']!=20 or c['validation_updates']!=[0,50,100,150,200]:raise ValueError('conditional training dimensions/schedule mismatch')
    if plan.get('engineering_check',False):
        if plan['updates']!=1 or plan.get('engineering_dimensions')!=[8,16] or not plan.get('discard_engineering_update'):raise ValueError('engineering check must discard exactly 8x16 update')
    elif plan['updates']!=200:raise ValueError('conditional formal budget is 200 updates')
    init=plan.get('initialization',{})
    if scratch:
        if init!={'mode':'scratch','parent_alpha':None,'checkpoint':None,'checkpoint_sha256':None}:raise ValueError('scratch initialization must have null upper parent identity')
        if config['network'].get('initialization')!='fresh_scratch':raise ValueError('scratch network initialization mismatch')
        if plan.get('preference_selection_schema')!=config.get('preference_selection_schema'):raise ValueError('scratch selection mismatch')
        from .path_command_selection import resolve_preference_acceptance
        if config.get('preference_selection_schema') is not None:resolve_preference_acceptance(config)
        if config.get('dynamic_preference_checks',{}).get('enabled'):
            from .path_command_dynamic import dynamic_contract
            if dynamic_contract(config).get('trigger')!='qualified_research_preference':raise ValueError('scratch explicit dynamic contract mismatch')
    elif init.get('mode')!=('actor_only_fold_alpha1' if strong else 'actor_only_zero_alpha_column') or init.get('parent_alpha')!=(1 if strong else 0):raise ValueError('conditional actor-only initialization mismatch')
    if strong and not scratch and (config['path_reward'].get('primary_speed_multiplier')!=2 or plan.get('preference_selection_schema')!='path_strong_preference_v1' or config.get('preference_selection_schema')!='path_strong_preference_v1'):raise ValueError('strong preference reward/selection contract mismatch')
    if not scratch and (not init.get('checkpoint') or len(init.get('checkpoint_sha256',''))!=64):raise ValueError('conditional parent identity missing')
    if plan.get('coverage_monitor')!={'target':.05,'check_every':10,'consecutive_low_windows':2}:raise ValueError('conditional exposure coverage monitor mismatch')


def validate_conflict_plan(plan,config):
    """Versioned exposure changes; inherited physical/reward/initialization contract."""
    import copy
    if not is_conditional(plan):raise ValueError('conflict shared conditional mode required')
    if plan.get('seed')!=911501 or config['alpha_conditioning'].get('seed')!=911502:raise ValueError('conflict independent seed contract')
    sampler=config['training_sampler']
    if sampler.get('version')!='path_conflict_exposure_v1' or sampler.get('route_seed')!=910501 or sampler.get('speed_event_seed')!=911503:raise ValueError('conflict sampler seed/version contract')
    if plan.get('sampler_version')!=sampler['version'] or plan['sampler_resolution'].get('training_sampler')!=sampler:raise ValueError('conflict sampler disconnected')
    if plan.get('preference_selection_schema')!='path_conflict_preference_v1' or config.get('preference_selection_schema')!='path_conflict_preference_v1':raise ValueError('conflict preference contract')
    if plan.get('scientific_threshold_policy')!='diagnostic_continue_declared_200':raise ValueError('scientific thresholds cannot stop conflict budget')
    c=config['training_future_phase_C']
    if c.get('validation_updates')!=[0,100,200] or c.get('lightweight_validation_updates')!=[50,150] or c.get('save_every')!=1:raise ValueError('conflict DEV/checkpoint schedule mismatch')
    if plan.get('budget')!=dict(training_policy_transitions=13107200,training_lower_ticks=52428800,full_dev_lower_ticks=720000,lightweight_lower_ticks=64000,preflight_lower_ticks=64000,engineering_lower_ticks=512,dynamic_lower_ticks=24000,total_lower_ticks=53301312):raise ValueError('conflict declared budget mismatch')
    # Reuse all inherited architecture, finite-mixture, scratch and PPO validation.
    legacy_plan=copy.deepcopy(plan);legacy_cfg=copy.deepcopy(config)
    legacy_plan['schema']=SCRATCH_SCHEMA;legacy_plan['sampler_version']='demand_exposure_v2'
    legacy_plan['preference_selection_schema']=legacy_cfg['preference_selection_schema']='path_preference_research_v1'
    legacy_cfg['training_future_phase_C']['validation_updates']=[0,50,100,150,200]
    legacy_cfg['dynamic_preference_checks'].update(first_change_s=.5,return_s=6,trigger='qualified_research_preference')
    legacy_cfg['dynamic_preference_checks'].pop('cases',None)
    # Acceptance has its own new-version validation in selection.
    reference=__import__('json').loads((Path(__file__).parents[2]/'configs/STTW_Path_Unified_Scratch.json').read_text())
    legacy_cfg['preference_acceptance']=reference['preference_acceptance']
    validate_conditional_plan(legacy_plan,legacy_cfg)
    from .path_command_selection import resolve_preference_acceptance
    resolve_preference_acceptance(config)


def initialize_conditional_actor(algo,parent_checkpoint,expected_sha,*,expected_lower_sha=None,config=None):
    import torch
    if algo.optimizer.state_dict()['state']:raise ValueError('conditional initialization requires fresh Adam')
    saved=torch.load(parent_checkpoint,map_location=algo.device,weights_only=False)
    from .path_command_policy import SCHEMA
    strong=is_strong(config or {})
    if saved.get('schema')!=SCHEMA or saved.get('identity',{}).get('alpha')!=(1 if strong else 0):raise ValueError('conditional parent alpha identity mismatch')
    if expected_lower_sha is None or saved['identity'].get('lower_actor_sha256')!=expected_lower_sha:raise ValueError('conditional parent lower identity mismatch')
    current=algo.policy.state_dict();parent=saved['policy'];keys={k for k in current if k.startswith('actor.')}
    parent_keys={k for k in parent if k.startswith('actor.')}
    dense_keys={k for k in keys if k.endswith(('.weight','.bias'))}
    if parent_keys!=(dense_keys if strong else keys):raise ValueError('conditional parent actor keys mismatch')
    for key in parent_keys:
        if parent[key].shape!=current[key].shape or not torch.isfinite(parent[key]).all():raise ValueError('invalid conditional parent actor tensor '+key)
    first_name,first=next((name,module) for name,module in algo.policy.actor.named_modules() if isinstance(module,torch.nn.Linear))
    if first.in_features!=351:raise ValueError('conditional actor schema351 required')
    for key in parent_keys:current[key]=parent[key].to(device=algo.device,dtype=current[key].dtype).clone()
    if strong:
        current[f'actor.{first_name}.bias']+=current[f'actor.{first_name}.weight'][:,ALPHA_INDEX].clone()
        for key in keys-parent_keys:current[key].zero_()
    current[f'actor.{first_name}.weight'][:,ALPHA_INDEX]=0
    algo.policy.load_state_dict(current,strict=True)
    return dict(mode='actor_only_fold_alpha1' if strong else 'actor_only_zero_alpha_column',parent_checkpoint=str(Path(parent_checkpoint).resolve()),parent_checkpoint_sha256=expected_sha,
        restored=sorted(parent_keys),folded_parent_alpha=1 if strong else None,zeroed_alpha_input_column=ALPHA_INDEX,critic='fresh',log_std='configured_fresh',optimizer='fresh',rng='fresh',update=0)


def initialize_upper_policy(algo,plan,config,source):
    """First-start dispatch only; self-checkpoint resumes use restore_checkpoint."""
    init=plan['initialization']
    if init['mode']=='scratch':
        if algo.optimizer.state_dict()['state']:raise ValueError('scratch initialization requires fresh Adam')
        return dict(mode='scratch',parent_checkpoint=None,parent_checkpoint_sha256=None,restored=[],
            critic='fresh',optimizer='fresh',log_std='configured_fresh',rng='fresh',update=0)
    return initialize_conditional_actor(algo,init['checkpoint'],init['checkpoint_sha256'],
        expected_lower_sha=source['candidate']['actor_sha256'],config=config)


def conditional_identity(plan,config,source):
    from .path_command_policy import SCHEMA
    identity=dict(schema=SCHEMA,policy_mode=MODE,alpha_conditioning_key=config_key(config['alpha_conditioning']),
        lower_actor_sha256=source['candidate']['actor_sha256'],source_key=config_key(source),config_key=config_key(config),
        sampler_key=config_key(plan['sampler_resolution']),best_key=config_key(plan['best_resolution']),seed=plan['seed'],
        parent_checkpoint_sha256=plan['initialization']['checkpoint_sha256'],engineering_check=bool(plan.get('engineering_check',False)))
    if is_strong(config):
        from .path_command_policy import architecture_identity
        identity.update(architecture=architecture_identity(config),reward_version=config['path_reward'].get('reward_version'),preference_selection_schema=config.get('preference_selection_schema'))
    if 'curriculum' in config['alpha_conditioning']:identity['curriculum_key']=config_key(config['alpha_conditioning']['curriculum'])
    if plan['initialization']['mode']=='scratch':
        from .path_command_selection import resolve_preference_acceptance
        n,steps=execution_dimensions(plan,config)
        identity.update(initialization_mode='scratch',
            declared_updates=plan['updates'],actual_batch_dimensions=[n,steps],declared_policy_transitions=n*steps*plan['updates'])
    if plan['initialization']['mode']=='scratch' and config.get('preference_selection_schema') is not None:
        identity['acceptance_key']=config_key(resolve_preference_acceptance(config))
    if config.get('training_sampler',{}).get('version')=='path_conflict_exposure_v1':
        from .path_command_scenarios import explicit_case_declarations
        identity['case_declarations_key']=config_key(explicit_case_declarations(config))
    return identity


def complete_resume_validation(parent,out,update,checkpoint,identity,config,params,evaluate,tick_count):
    """Finish one saved learner boundary before advancing PPO after interruption.

    Complete panels remain in their immutable parent directory. An incomplete
    panel is re-evaluated at this exact restored model in the new run directory;
    persisted repeated ticks and all newly executed ticks are recorded explicitly.
    """
    import json
    import numpy as np
    from .preference_best import atomic_json
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
        config_key=config_key(config))
    if config.get('preference_selection_schema') in ('path_preference_research_v1','path_conflict_preference_v1'):
        from .path_command_selection import resolve_preference_acceptance
        expected['acceptance_key']=config_key(resolve_preference_acceptance(config))
    if config.get('training_sampler',{}).get('version')=='path_conflict_exposure_v1':
        from .path_command_scenarios import explicit_case_declarations
        expected['case_declarations_key']=config_key(explicit_case_declarations(config))
    if meta is not None and any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('resume conditional panel identity mismatch')
    source=json.loads(source_file.read_text()) if source_file.exists() else None
    if source is not None:
        if source.get('source_update')!=update or any(source.get(k)!=v for k,v in identity.items()):raise ValueError('resume conditional source identity mismatch')
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
            if config.get('preference_selection_schema') in ('path_strong_preference_v1','path_preference_research_v1','path_conflict_preference_v1'):
                from .path_command_selection import summarize_preference,persist_preference_best
                persist_preference_best(out,source['checkpoint'],source['actor'],summarize_preference(panels,config),{**identity,'source_update':update,'validation_dir':str(boundary.resolve())})
        else:evaluate(update,checkpoint)
        receipt['completed']=True
    finally:
        receipt['new_validation_lower_ticks']=int(tick_count()-before)
        atomic_json(receipt_path,receipt)
    return receipt
