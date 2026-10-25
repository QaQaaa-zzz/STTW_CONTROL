"""Evaluation actions with the existing frozen-lower physical continuation.

This opt-in collector supplies no PPO batch and no fitted survival baseline.
Active training modules are left unchanged. Stopped/held transitions reuse the
existing collector, so domain exit never invents an upper action or task reward.
"""
import copy
import math
import jax
import jax.numpy as j

from .survival_endpoint_protocol import choose_evaluation_action


def evaluation_config(common, *, speed_floor=None):
    """Change only the permitted lower bound of given speed references.

    Requested speed, path, rewards, observation scales, lower policy and plant
    remain the common configuration. Existing correction_tick performs both
    clipping and filter back-calculation under this declared legal restriction.
    """
    result = copy.deepcopy(common)
    if speed_floor is not None:
        bounds = result['residual_interface']['speed_reference_bounds_m_s']
        if not math.isfinite(speed_floor) or not bounds[0] <= speed_floor <= bounds[1]:
            raise ValueError('speed floor must be finite and inside common reference permissions')
        result['residual_interface']['speed_reference_bounds_m_s'] = [float(speed_floor), bounds[1]]
    return result


def normal_evaluation_interval(env, actor, carry, params, log_std, path, speed,
                               pulse, *, mode, horizon_s):
    """One real task-active policy interval; same action is applied and logged."""
    from .path_command_reward import configured_cost_keys
    state = carry.state
    obs, critic_obs, observation_fault = env.observation(state, path, speed, horizon_s)
    # An exact zero controller has no trained Actor or checkpoint identity.
    mean = j.zeros(2, obs.dtype) if mode == 'zero' else actor.apply(params, obs)
    action, log_prob, density_valid, next_key = choose_evaluation_action(mean, log_std, carry.key, mode=mode)
    fault = (observation_fault | ~j.all(j.isfinite(mean)) | ~j.all(j.isfinite(action))
             | (density_valid & ~j.isfinite(log_prob)))
    end, logs = env.policy_step(state.replace(fault=state.fault | fault), action,
                                path, speed, horizon_s, pulse)
    names = ['active_tick', 'actual_xy', 'reference_xy', 'actual_forward_speed',
             'wheel_speed_proxy', 'phi', 'phi_dot', 'actual_delta', 'peak_roll',
             'path_cross_track', 'path_heading_error', 'path_progress',
             'goal_section_signed_distance', 'tick_reward', 'scored_tick_reward',
             'contact_failure', 'physical_failure', 'nonfinite', 'lower_fault',
             'policy_fault', 'domain_exit', 'applied_force_impulse', 'time']
    data = {name: logs[name] for name in names}
    ticks = state.tick+j.arange(4, dtype=j.int32)
    keys = tuple(configured_cost_keys(env.config))
    data.update(reference=logs['governed'], upper_active=logs['active_tick'],
        fallback_active=j.zeros(4, dtype=bool), task_reward_valid=logs['active_tick'],
        physical_tick=j.where(logs['active_tick'], ticks, -1),
        horizon_reached=logs['active_tick'] & (ticks == round(horizon_s/env.cc.dt)-1),
        raw_costs=j.stack([logs['raw_components'][k] for k in keys], axis=-1),
        effective_costs=j.stack([logs['effective_components'][k] for k in keys], axis=-1))
    data.update(nominal_reference=logs['nominal'], target_offset=logs['target_offset'],
        filtered_offset=logs['filtered_offset'], correction_offset=logs['offsets'],
        reference_clip_channels=logs['reference_clip_channels'],
        correction_rate_clipped=logs['correction_rate_clipped'], chi=logs['chi'],
        requested_speed=j.repeat(j.asarray(speed), 4))
    handover = end.domain_exit & ~carry.handed_over
    held = j.where(handover, end.physical.governor.current_reference, carry.held_reference)
    decision = dict(valid=j.any(logs['active_tick']) & ~fault, observation=obs,
        critic_observation=critic_obs, old_task_value=j.float32(j.nan),
        raw_action=action, mean=mean, log_prob=log_prob, density_valid=density_valid,
        physical_value=j.float32(j.nan), task_tick=state.tick, rng_key=carry.key)
    return carry.replace(state=end, key=next_key, held_reference=held,
                         handed_over=carry.handed_over | handover), (decision, data)


class _ZeroCorrection:
    @staticmethod
    def apply(_params, observation):
        return j.zeros(2, observation.dtype)


def make_endpoint_collection_step(env, actor, *, horizon_s, mode):
    """Reuse shared physical tails; select an explicit evaluation action mode.

    The caller supplies an environment made with evaluation_config for a floor
    comparator, and the same original requested speed as all other roles.
    """
    from .survival_collection import make_survival_collection_step
    if mode not in ('mean', 'zero', 'gaussian'):
        raise ValueError('explicit evaluation inference mode required')
    if mode != 'zero' and actor is None:
        raise ValueError('trained Actor required for mean/gaussian inference')
    # Existing factory validates horizon/contact-only contract and owns all
    # lower-only hold/stop logic. Its Gaussian normal branch is never selected
    # by this evaluator, and its density/value placeholders are not evidence.
    shared = make_survival_collection_step(env, _ZeroCorrection() if mode == 'zero' else actor,
                                           horizon_s=horizon_s)
    limit = round(horizon_s/env.cc.dt)

    def tail(carry, params, log_std, path, speed, pulse):
        end, (decision, logs) = shared(carry, params, log_std, path, speed, pulse)
        logs = dict(logs)
        logs.update({name: j.full((4, 2), j.nan) for name in
                     ['nominal_reference', 'target_offset', 'filtered_offset', 'correction_offset']})
        logs.update(reference_clip_channels=j.zeros((4, 2), bool),
                    correction_rate_clipped=j.zeros(4, bool), chi=j.full(4, j.nan),
                    requested_speed=j.repeat(j.asarray(speed), 4))
        return end, (dict(decision, density_valid=j.bool_(False)), logs)

    def normal(carry, params, log_std, path, speed, pulse):
        return normal_evaluation_interval(env, actor, carry, params, log_std, path,
                                         speed, pulse, mode=mode, horizon_s=horizon_s)

    def step(carry, params, log_std, path, speed, pulse):
        stop = carry.state.physical.failed | carry.state.fault | (carry.state.tick >= limit)
        return jax.lax.cond(carry.state.domain_exit | stop, tail, normal,
                            carry, params, log_std, path, speed, pulse)

    return step
