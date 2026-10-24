"""Fresh frozen-policy task0 sampling with explicit lower-only physical tails.

No reset, optimization, or hypothetical upper observation occurs at handover.
The caller authenticates the sampling checkpoint/distribution and complete data.
"""
import math
import jax
import jax.numpy as j
from flax import struct


def sample_gaussian_action(mean, log_std, key):
    """Save the raw Gaussian sample and joint density before command tanh."""
    next_key, sample_key = jax.random.split(key)
    epsilon = jax.random.normal(sample_key, mean.shape, dtype=mean.dtype)
    action = mean+j.exp(log_std)*epsilon
    log_prob = -.5*j.sum(epsilon**2+2*log_std+j.log(2*j.pi))
    return action, log_prob, next_key


@struct.dataclass
class SurvivalCollectionState:
    state: object
    key: object
    held_reference: object
    handed_over: object


def evaluate_collection_networks(actor, task_critic, params, observation, critic_observation):
    """Pre-action means/values from one frozen parameter snapshot.

    The optional dual-value mode takes dynamic Actor/Critic/physical parameters.
    Legacy callers retain their constant-.5 baseline and no task value.
    """
    if task_critic is None:
        return actor.apply(params, observation), j.float32(j.nan), j.float32(.5)
    mean=actor.apply(params['actor'], observation)
    task_value=task_critic.apply(params['task_critic'], critic_observation).reshape(())[()]
    physical=jax.nn.sigmoid(critic_observation@params['physical']['kernel']+params['physical']['bias']).reshape(())[()]
    return mean,task_value,physical


def make_survival_collection_step(env, actor, *, horizon_s, task_critic=None):
    """One actual upper interval; output one decision and four lower log slots.

    env must already be the opt-in contact-only instance with zero_log installed.
    Legacy mode uses the declared constant .5 physical baseline. With optional
    task_critic, dynamic parameters provide both frozen collection values.
    Task scores retain the configured V3 costs
    under the diagnostic contact/domain terminal contract, never during hold.
    """
    from .balance_probe import make_lower_tick
    from .physical_continuation import make_hold_last_tick
    from .path_command_reward import configured_cost_keys
    dt = env.cc.dt
    limit = round(horizon_s/dt)
    if (not math.isfinite(horizon_s) or horizon_s <= 0 or limit % 4
            or not math.isclose(limit*dt, horizon_s, abs_tol=1e-8)):
        raise ValueError('positive horizon aligned to complete upper intervals required')
    if not hasattr(env.physics, 'original'):
        raise ValueError('explicit contact-only physics instance required')
    keys = tuple(configured_cost_keys(env.config))
    scalar = j.float32(0.)
    zero_tick = {name: scalar for name in ['time', 'actual_forward_speed', 'wheel_speed_proxy',
        'phi', 'phi_dot', 'actual_delta', 'peak_roll', 'path_cross_track', 'path_heading_error',
        'path_progress', 'goal_section_signed_distance', 'tick_reward', 'scored_tick_reward']}
    zero_tick.update({name: j.bool_(False) for name in ['active_tick', 'upper_active',
        'fallback_active', 'contact_failure', 'physical_failure', 'nonfinite', 'lower_fault',
        'policy_fault', 'domain_exit', 'horizon_reached', 'task_reward_valid']})
    zero_tick.update(actual_xy=j.zeros(2), reference_xy=j.zeros(2), reference=j.zeros(2),
                     applied_force_impulse=j.zeros(3), physical_tick=j.int32(-1),
                     raw_costs=j.zeros(len(keys)), effective_costs=j.zeros(len(keys)))
    zero_block = jax.tree.map(lambda x:j.repeat(x[None], 4, axis=0), zero_tick)
    blank_decision = dict(valid=j.bool_(False), observation=j.full((351,), j.nan),
        critic_observation=j.full((352,), j.nan), old_task_value=j.float32(j.nan),
        raw_action=j.full((2,), j.nan), mean=j.full((2,), j.nan), log_prob=j.float32(j.nan),
        physical_value=j.float32(j.nan), task_tick=j.int32(-1), rng_key=j.zeros(2, j.uint32))
    advance_hold = make_hold_last_tick(make_lower_tick(env, j.zeros(2)), dt=dt)

    def normal(carry, params, log_std, path, speed, pulse):
        state = carry.state
        obs, critic_obs, observation_fault = env.observation(state, path, speed, horizon_s)
        mean, task_value, physical_value = evaluate_collection_networks(actor, task_critic, params, obs, critic_obs)
        action, log_prob, next_key = sample_gaussian_action(mean, log_std, carry.key)
        fault = observation_fault | ~j.all(j.isfinite(mean)) | ~j.all(j.isfinite(action)) | ~j.isfinite(log_prob)
        if task_critic is not None:
            fault = fault | ~j.isfinite(task_value) | ~j.isfinite(physical_value)
        end, logs = env.policy_step(state.replace(fault=state.fault|fault), action,
                                    path, speed, horizon_s, pulse)
        data = dict(zero_block)
        for name in ['active_tick', 'actual_xy', 'reference_xy', 'actual_forward_speed',
                     'wheel_speed_proxy', 'phi', 'phi_dot', 'actual_delta', 'peak_roll',
                     'path_cross_track', 'path_heading_error', 'path_progress',
                     'goal_section_signed_distance', 'tick_reward', 'scored_tick_reward',
                     'contact_failure', 'physical_failure', 'nonfinite', 'lower_fault',
                     'policy_fault', 'domain_exit', 'applied_force_impulse', 'time']:
            data[name] = logs[name]
        tick = state.tick+j.arange(4, dtype=j.int32)
        data.update(reference=logs['governed'], upper_active=logs['active_tick'],
            task_reward_valid=logs['active_tick'],
            physical_tick=j.where(logs['active_tick'], tick, -1),
            horizon_reached=logs['active_tick']&(tick==limit-1),
            raw_costs=j.stack([logs['raw_components'][k] for k in keys], axis=-1),
            effective_costs=j.stack([logs['effective_components'][k] for k in keys], axis=-1))
        handover = end.domain_exit & ~carry.handed_over
        held = j.where(handover, end.physical.governor.current_reference, carry.held_reference)
        decision = dict(valid=j.any(logs['active_tick'])&~fault, observation=obs,
            critic_observation=critic_obs, old_task_value=task_value,
            raw_action=action, mean=mean, log_prob=log_prob, physical_value=physical_value,
            task_tick=state.tick, rng_key=carry.key)
        return carry.replace(state=end, key=next_key, held_reference=held,
                             handed_over=carry.handed_over|handover), (decision, data)

    def held(carry, _params, _log_std, _path, _speed, pulse):
        def tick(state, _):
            def active(state):
                end, log = advance_hold(state, pulse, state.tick*dt, carry.held_reference)
                data = dict(zero_tick)
                for name in ['active_tick', 'actual_xy', 'actual_forward_speed', 'wheel_speed_proxy',
                             'phi', 'phi_dot', 'actual_delta', 'peak_roll', 'contact_failure',
                             'physical_failure', 'nonfinite', 'lower_fault', 'applied_force_impulse',
                             'reference', 'time']:
                    data[name] = log[name]
                data.update(fallback_active=j.bool_(True), domain_exit=j.bool_(True),
                    physical_tick=state.tick, horizon_reached=state.tick==limit-1,
                    reference_xy=j.full((2,), j.nan), path_cross_track=j.float32(j.nan),
                    path_heading_error=j.float32(j.nan), path_progress=j.float32(j.nan),
                    goal_section_signed_distance=j.float32(j.nan),
                    raw_costs=j.full((len(keys),), j.nan), effective_costs=j.full((len(keys),), j.nan))
                return end, data
            stop = state.physical.failed|state.fault|(state.tick>=limit)
            return jax.lax.cond(stop, lambda x:(x, zero_tick), active, state)
        end, data = jax.lax.scan(tick, carry.state, None, length=4)
        return carry.replace(state=end), (blank_decision, data)

    def step(carry, params, log_std, path, speed, pulse):
        stopped = carry.state.physical.failed|carry.state.fault|(carry.state.tick>=limit)
        def active(carry):
            return jax.lax.cond(carry.state.domain_exit, held, normal,
                                carry, params, log_std, path, speed, pulse)
        return jax.lax.cond(stopped, lambda x:(x, (blank_decision, zero_block)), active, carry)

    return step
