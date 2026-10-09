"""Resolve the supplied SmoothV4 endpoint specifications onto the real V3 interfaces."""
from pathlib import Path
import json

def resolve(alpha, *, fresh=False, train_wall=1800, total_wall=5400, preference_v5=False, preference_v51=False, stage=1):
    if preference_v51:
        return resolve_preference_v51(alpha)
    if preference_v5:
        return resolve_preference(alpha, stage=stage)
    root=Path(__file__).resolve().parents[3]
    s=json.loads((root/f'learning/configs/upper_STTW_R196_ALPHA1_alpha{alpha}.json').read_text())
    v=json.loads((root/f'docs/smooth_v4/attachment/STTW_R196_SmoothV4_alpha{alpha}.json').read_text())
    assert v['fixed_upper_alpha']==alpha
    s['smooth_v4']=v;s['reward']=v['reward'];s['actor_temporal_regularizer']=v['actor_temporal_regularizer']
    for k,x in v['action'].items():
        if k in s['action'] and isinstance(x,(float,int)):assert s['action'][k]==x,(k,s['action'][k],x)
    s['ppo'].update(v['ppo']);s['ppo'].update(seed=81,target_mean_kl_after_epoch=.01,hard_stop_mean_kl=.03,default_updates=60,future_total_updates_only_after_user_approval=60)
    c=s['commands'];t=v['training_commands']
    for k in ['nominal_fraction','conflict_fraction','random_fraction']:c[k]=t[k]
    c['random_seed']=81;c['nominal']['straight_probability']=t['nominal_straight_fraction']
    c['nominal']['steer_abs_max_rad']=t['nominal_steer_abs_max_rad']
    s['budget'].update(additional_compute_wall_seconds=5400,compile_wall_seconds=600,training_wall_seconds=1800,evaluation_wall_seconds=1200,engineering_wall_seconds=120,default_control_transitions_upper=2*62*512*128*4)
    if fresh:
        s['initialization_mode']='scratch'
        s['smooth_v4']['upper']['warm_start_actor_only']=False
        s['smooth_v4']['ppo']['policy_updates_per_endpoint']=150
        s['smooth_v4']['ppo']['validation_updates']=[60,150]
        s['smooth_v4']['evaluation']['stage150_cases']=list(v['evaluation']['stage60_cases'])
        s['ppo'].update(default_updates=150,future_total_updates_only_after_user_approval=150,policy_updates_per_endpoint=150,validation_updates=[60,150])
        s['smooth_v4']['budget'].update(train_wall_per_endpoint_s=train_wall,overall_new_compute_wall_s=total_wall)
        s['budget'].update(training_wall_seconds=train_wall,additional_compute_wall_seconds=total_wall,default_control_transitions_upper=2*152*512*128*4)
    return s


def resolve_preference(alpha, stage=1):
    """Adapt the frozen V5 semantic specification to the existing runtime keys."""
    from copy import deepcopy
    if alpha not in (0, 1) or stage not in (1, 2):
        raise ValueError("V5 requires endpoint alpha 0/1 and stage 1/2")
    alpha = int(alpha)
    s = resolve(alpha)
    root = Path(__file__).resolve().parents[3]
    v = json.loads((root / f'docs/preference_v5/attachment/STTW_R196_PreferenceV5_alpha{alpha}.json').read_text())
    s['preference_v5'] = v
    s['preference_v5_stage'] = stage
    s['training_stage'] = stage
    s['critic_horizon_normalizer_s'] = 16.0
    s['initialization_mode'] = 'scratch' if stage == 1 else 'learner_state_resume_environment_reset'
    s['network']['critic_horizon_normalizer_s'] = 16.0
    for key, value in v['action'].items():
        if key in s['action'] and isinstance(value, (int, float)):
            assert value == s['action'][key], (key, value, s['action'][key])
    r = s['reward'] = deepcopy(s['reward'])
    vr = v['reward']
    for key in ('scale', 'global_cost_cap', 'normal', 'recovery', 'safety', 'reference_priority', 'command_compatibility'):
        r[key] = deepcopy(vr[key])
    for target, source in {'conflict_alpha0':'alpha0_conflict', 'conflict_alpha1':'alpha1_conflict',
                           'offset_magnitude':'magnitude', 'independent_component_caps':'caps',
                           'failure_absorbing_cost_rate':'failure_cost_rate_bound', 'failure_extra_penalty':'failure_extra'}.items():
        r[target] = deepcopy(vr[source])
    smooth = vr['smoothness']
    r['rate_sample_dt_s'] = smooth['sample_dt_s']
    r['upper_rate'] = dict(weight=smooth['rate_weight'], normalizers=smooth['rate_normalizers'])
    r['upper_acceleration'] = dict(weight=smooth['acceleration_weight'], normalizers=smooth['acceleration_normalizers'])
    r['conflict_phi_start_rad'] = vr['context']['chi_start_phi_rad']
    r['conflict_phi_full_rad'] = vr['context']['chi_full_phi_rad']
    assert sum(r['independent_component_caps'].values()) == r['failure_absorbing_cost_rate']
    tr = v['temporal_regularizer']; mask = tr['mask_both_endpoints']
    reg = s['actor_temporal_regularizer'] = deepcopy(s['actor_temporal_regularizer'])
    reg.update(deepcopy(tr))
    reg.update(coefficient=tr['max_weight'], ramp_policy_updates=tr['linear_ramp_end_update']-tr['zero_until_update'],
               eligible_raw_speed_rate_m_s2=mask['raw_speed_rate_max_m_s2'], eligible_raw_steer_rate_rad_s=mask['raw_steer_rate_max_rad_s'],
               eligible_roll_rad=mask['abs_roll_max_rad'], eligible_roll_rate_rad_s=mask['abs_roll_rate_max_rad_s'])
    p = s['ppo']; vp = v['ppo']
    for key in ('seed','num_envs','rollout_policy_steps','epochs','minibatch_size','gamma','gae_lambda','clip_ratio','adam_betas','weight_decay','initial_latent_std'):
        p[key] = deepcopy(vp[key])
    mapping = {'actor_learning_rate':'actor_lr','critic_learning_rate':'critic_lr','minibatches':'num_minibatches',
               'value_loss_coefficient':'value_coef','entropy_coefficient':'entropy_coef','adam_epsilon':'adam_eps',
               'latent_std_min':'std_min','latent_std_max':'std_max','max_grad_norm_actor':'max_grad_actor',
               'max_grad_norm_critic':'max_grad_critic','target_mean_kl_after_epoch':'soft_kl','hard_stop_mean_kl':'hard_kl',
               'actor_lr_floor':'lr_floor','hard_reject_consecutive_stop':'consecutive_hard_reject_stop',
               'value_only_initial_rollouts':'fresh_value_only_rollouts','save_every_updates':'checkpoint_every_updates'}
    for target, source in mapping.items():
        p[target] = deepcopy(vp[source])
    t = v['training'][f'stage{stage}']
    s['episode_duration_s'] = t['episode_s']
    cumulative_updates = sum(v['training'][f'stage{i}']['updates'] for i in range(1,stage+1))
    p.update(default_updates=cumulative_updates, policy_updates_per_endpoint=cumulative_updates, stage_additional_updates=t['updates'],
             future_total_updates_only_after_user_approval=120,
             validation_updates=t.get('diagnostic_updates', t.get('diagnostic_cumulative_updates')),
             time_semantics='finite_stage_horizon_termination_no_bootstrap')
    c = s['commands']
    c.update(episode_seconds=t['episode_s'], random_seed=vp['seed'],
             nominal_fraction=t['nominal_probability'], conflict_fraction=t['conflict_probability'], random_fraction=t['random_probability'],
             episode_speed_slew_range_m_s2=t['speed_slew_range_m_s2'], episode_steer_slew_range_rad_s=t['steer_slew_range_rad_s'])
    c['nominal'].update(straight_probability=.6, steer_abs_max_rad=.08)
    if stage == 1:
        c.update(speed_min_m_s=t['nominal_speed_range_m_s'][0], speed_max_m_s=t['nominal_speed_range_m_s'][1])
        c['nominal'].update(first_switch_s=t['nominal_target_hold_range_s'], hold_s=t['nominal_target_hold_range_s'], last_random_target_s=t['episode_s'])
        c['conflict'].update(speed_range_m_s=t['conflict_speed_target_range_m_s'], turn_start_s=t['turn_start_range_s'],
                             steer_magnitude_rad=t['steer_magnitude_range_rad'], single_turn_fraction=1.0)
    else:
        c.update(speed_min_m_s=t['speed_range_m_s'][0], speed_max_m_s=t['speed_range_m_s'][1], steer_abs_max_rad=t['steer_abs_max_rad'])
        c['conflict'].update(speed_range_m_s=t['turn_speed_range_m_s'], turn_start_s=t['turn_start_range_s'],
                             steer_magnitude_rad=t['turn_magnitude_range_rad'], single_turn_fraction=t['turn_single_probability'],
                             tail_speed_range_m_s=t['tail_random_speed_range_m_s'])
        c['random'].update(hold_s=t['random_target_hold_range_s'], last_random_target_s=t['random_tail_start_s'])
        # The inherited smooth command generator consumes this adapter section.
        sc = s['smooth_v4']['training_commands']
        sc.update(nominal_fraction=t['nominal_probability'], conflict_fraction=t['conflict_probability'], random_fraction=t['random_probability'],
                  conflict_speed_range_m_s=t['turn_speed_range_m_s'], conflict_start_s=t['turn_start_range_s'],
                  conflict_steer_magnitude_rad=t['turn_magnitude_range_rad'], conflict_single_fraction=t['turn_single_probability'],
                  conflict_reversal_fraction=1-t['turn_single_probability'],
                  conflict_hold_after_limited_target_reached_s=t['hold_after_slew_range_s'],
                  conflict_tail_same_high_speed_probability=t['tail_keep_high_probability'],
                  conflict_other_tail_speed_range_m_s=t['tail_random_speed_range_m_s'])
    s['smooth_v4'] = {'training_commands': s['smooth_v4']['training_commands']}
    s['budget']['wall_limits_enabled'] = False
    s['budget'].update(default_policy_transitions=v['training']['total_policy_transitions_two_endpoints'],
                       default_control_transitions_upper=v['training']['total_control_ticks_two_endpoints'],
                       max_updates_per_endpoint=120, no_extra_wall_cutoff=True)
    return s


def resolve_preference_v51(alpha):
    """Apply the supplied V5.1 endpoint overlay without changing V5."""
    from copy import deepcopy
    if alpha not in (0, 1):
        raise ValueError("V5.1 requires endpoint alpha 0/1")
    alpha = int(alpha)
    s = resolve_preference(alpha, stage=2)
    root = Path(__file__).resolve().parents[3]
    overlay = json.loads((root / f'docs/preference_v51/attachment/STTW_V5_1_alpha{alpha}.json').read_text())
    assert overlay['run']['upper_alpha'] == alpha
    assert overlay['source_commit'] == '6a264f77468cc4090f8e79919027b333740e5dfa'
    s['priority_recovery_v51'] = True
    s['preference_v51'] = overlay
    s['preference_v5_stage'] = None
    s['training_stage'] = 51
    s['initialization_mode'] = 'scratch'
    s['episode_duration_s'] = float(overlay['run']['episode_seconds'])

    change = overlay['reward_changes']
    r = s['reward'] = deepcopy(s['reward'])
    primary = change['primary_excess']
    r['primary_excess'] = dict(
        weight=primary['weight'],
        alpha0_steer_tolerance_rad=primary['alpha0_actual_steer_tolerance_rad'],
        alpha0_steer_scale_rad=primary['alpha0_error_scale_rad'],
        alpha1_speed_tolerance_m_s=primary['alpha1_actual_speed_tolerance_m_s'],
        alpha1_speed_scale_m_s=primary['alpha1_error_scale_m_s'])
    yaw = change['replace_yaw_damping']
    r['recovery'] = deepcopy(r['recovery'])
    for key in ('yaw_damping_weight', 'yaw_damping_heading_scale_rad'):
        r['recovery'].pop(key, None)
    r['recovery'].update(yaw_recovery_weight=yaw['weight'], yaw_debt_gain_per_s=yaw['debt_gain_per_s'],
                         yaw_debt_rate_limit_rad_s=yaw['debt_rate_limit_rad_s'], yaw_scale_rad_s=yaw['error_scale_rad_s'])
    caps = r['independent_component_caps'] = deepcopy(r['independent_component_caps'])
    assert caps.pop('yaw_damping') == yaw['cap']
    caps['yaw_recovery'] = yaw['cap']
    caps['primary_excess'] = primary['cap']
    r['failure_absorbing_cost_rate'] = change['failure_cost_rate_bound']
    assert sum(caps.values()) == r['failure_absorbing_cost_rate'] == 2080.

    p = s['ppo']; vp = overlay['ppo']
    p.update(seed=vp['seed'], num_envs=vp['num_envs'], rollout_policy_steps=vp['rollout_policy_steps'],
             epochs=vp['epochs'], minibatches=vp['minibatches'], minibatch_size=vp['minibatch_size'],
             actor_learning_rate=vp['actor_lr'], critic_learning_rate=vp['critic_lr'], gamma=vp['gamma'],
             gae_lambda=vp['gae_lambda'], clip_ratio=vp['clip_ratio'], value_loss_coefficient=vp['value_coef'],
             entropy_coefficient=vp['entropy_coef'], adam_betas=vp['adam_betas'], adam_epsilon=vp['adam_eps'],
             max_grad_norm_actor=vp['max_grad_actor'], max_grad_norm_critic=vp['max_grad_critic'],
             initial_latent_std=vp['initial_latent_std'], latent_std_min=vp['latent_std_min'],
             latent_std_max=vp['latent_std_max'], target_mean_kl_after_epoch=vp['soft_kl'],
             hard_stop_mean_kl=vp['hard_kl'], actor_lr_floor=vp['actor_lr_floor'],
             hard_reject_consecutive_stop=vp['consecutive_hard_reject_stop'], fresh_value_only_rollouts=0,
             default_updates=overlay['run']['policy_updates'], policy_updates_per_endpoint=overlay['run']['policy_updates'],
             stage_additional_updates=overlay['run']['policy_updates'], validation_updates=overlay['run']['validation_updates'],
             future_total_updates_only_after_user_approval=overlay['run']['policy_updates'])
    family = overlay['training_families']; c = s['commands']
    c.update(episode_seconds=16., random_seed=vp['seed'], nominal_fraction=family['nominal_probability'],
             conflict_fraction=family['conflict_probability'], random_fraction=family['random_probability'],
             heading_recovery_start_fraction=family['heading_recovery_start_probability'])
    c['heading_recovery_start'] = deepcopy(family['recovery_start'])
    s['budget'].update(wall_limits_enabled=False,
                       default_policy_transitions=overlay['budget']['policy_transitions_both_endpoints'],
                       default_control_transitions_upper=overlay['budget']['control_ticks_both_upper_bound'],
                       max_updates_per_endpoint=overlay['run']['policy_updates'], no_extra_wall_cutoff=True)
    return s


def resolve_dimensions(spec, *, smoke=False, num_envs=None, rollout_steps=None):
    """Resolve execution dimensions and their derived budget in one place.

    Explicit overrides are engineering inputs, never new formal defaults.
    """
    from copy import deepcopy
    spec = deepcopy(spec)
    p = spec['ppo']
    if smoke and (num_envs is not None or rollout_steps is not None):
        raise ValueError('smoke and explicit dimensions are mutually exclusive')
    n = 8 if smoke else p['num_envs'] if num_envs is None else num_envs
    t = 16 if smoke else p['rollout_policy_steps'] if rollout_steps is None else rollout_steps
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (n,t,p['minibatches'])):
        raise ValueError('dimensions must be positive integers')
    if n*t % p['minibatches']:
        raise ValueError('N*T must divide evenly into minibatches')
    if smoke or num_envs is not None or rollout_steps is not None:
        p.update(num_envs=n, rollout_policy_steps=t, minibatch_size=n*t//p['minibatches'])
    if n*t != p['minibatches']*p['minibatch_size']:
        raise ValueError('N*T must equal minibatches*minibatch_size')
    # Preserve the original number of policy batches including legacy warmup.
    if spec.get('preference_v5'):
        transitions = 2*p['default_updates']*n*t
        spec['budget'].update(default_policy_transitions=transitions,
                              default_control_transitions_upper=4*transitions)
    if spec.get('preference_v52'):
        spec['budget']['default_control_transitions_upper']=2*(spec['budget']['max_updates_per_endpoint']+4)*n*t*4
        spec['budget']['value_only_control_ticks']=2*4*n*t*4
    spec['execution_dimensions'] = dict(num_envs=n, rollout_policy_steps=t,
                                        transitions_per_rollout=n*t,
                                        minibatches=p['minibatches'], minibatch_size=p['minibatch_size'])
    return spec


def resolve_preference_v52(alpha, additional_updates=200, max_additional_updates=400):
    """The supplied endpoint delta is the only reward change to frozen V5.1."""
    from copy import deepcopy
    if additional_updates not in (200,300,400) or not additional_updates<=max_additional_updates<=400:
        raise ValueError('V5.2 requires initial200/300/400 and bounded maximum<=400')
    s=resolve_preference_v51(alpha)
    root=Path(__file__).resolve().parents[3]
    v=json.loads((root/f'docs/preference_v52/attachment/STTW_V52_alpha{alpha}.json').read_text())
    assert v['fixed_upper_alpha']==alpha and v['source_commit']=='26c3fedac76b099ae98ff153f3f50f09f2730da9'
    s.update(preference_v52=v,training_stage=52,initialization_mode='parent_actor_and_log_std_new_critic_adam')
    d=v['reward_deltas'];r=s['reward']
    r['working_roll_excess']=deepcopy(d['working_roll_excess'])
    r['roll_use_substep_peak']=d['roll_use_substep_peak']
    r['reference_priority']['alpha1_downward_only']=True
    r['normal']['speed_deadband_m_s']=d['normal_speed_deadband_m_s']
    r['recovery']['speed_deadband_m_s']=d['recovery_speed_deadband_m_s']
    r['independent_component_caps']=deepcopy(d['independent_component_caps'])
    r['failure_absorbing_cost_rate']=sum(r['independent_component_caps'].values())
    assert r['failure_absorbing_cost_rate']==d['failure_absorbing_cost_rate']==5080
    p=s['ppo'];p.update(fresh_value_only_rollouts=4,default_updates=additional_updates,policy_updates_per_endpoint=additional_updates,
        stage_additional_updates=additional_updates,validation_updates=list(range(100,additional_updates+1,50)),save_every_updates=10,
        future_total_updates_only_after_user_approval=max_additional_updates)
    n=p['num_envs']*p['rollout_policy_steps']
    s['budget'].update(default_policy_transitions=2*additional_updates*n,default_control_transitions_upper=2*(max_additional_updates+4)*n*4,
        max_updates_per_endpoint=max_additional_updates,initial_updates_per_endpoint=additional_updates,value_only_control_ticks=2*4*n*4)
    s['lineage']=dict(parent_updates=100,additional_updates=0,reward_version='V5.2')
    return s
