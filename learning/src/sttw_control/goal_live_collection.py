"""Opt-in goal-informed collection on the existing live reference plant kernel.

This supplies factual pre-action inputs and separate frozen G/C/S values. It
neither chooses an objective hierarchy nor creates networks or an optimizer.
"""
import math
import jax
import jax.numpy as j
from flax import struct
from .reference_reentry import actor_permission
from .survival_collection import sample_gaussian_action


@struct.dataclass
class GoalLiveCollectionState:
    state: object
    key: object
    gate: object
    clean_blocks: object
    arrived: object  # Initialized false at original task start, never reset on reentry.


def original_goal_context(pose, goal_xy, *, tick, dt, horizon_s,
                          arrived, radius_m, distance_scale_m):
    """Current body-frame original-goal displacement, time, latch and radius.

    No disturbance direction, magnitude, onset or future trajectory is an input.
    Unclipped displacement preserves the original goal; scales are explicit.
    """
    delta = goal_xy - pose[:2]
    co, si = j.cos(pose[2]), j.sin(pose[2])
    return j.stack([(co * delta[0] + si * delta[1]) / distance_scale_m,
                    (-si * delta[0] + co * delta[1]) / distance_scale_m,
                    j.clip(1 - tick * dt / horizon_s, 0., 1.),
                    j.asarray(arrived, j.float32), radius_m / distance_scale_m])


def goal_latch_from_block(previous, logs, goal_xy, radius_m):
    """Use only measured active post-tick XY before any physical/fault stop."""
    xy = logs['actual_xy']
    active = logs['active_tick']
    bad = (logs['contact_failure'] | logs['nonfinite'] | logs['lower_fault'] |
           logs['policy_fault'] | ~j.all(j.isfinite(xy), axis=-1)) & active
    allowed = active & (j.cumsum(bad.astype(j.int32)) == 0)
    # Include the H sample itself, exclude any later slot in a malformed block.
    prior_horizon = j.cumsum((active & logs['horizon_reached']).astype(j.int32)) - (
        active & logs['horizon_reached']).astype(j.int32)
    hit = allowed & (prior_horizon == 0) & (j.linalg.norm(xy - goal_xy, axis=-1) <= radius_m)
    return previous | j.any(hit)


class _SampledAction:
    def apply(self, params, _observation):
        return params['latent']


def make_goal_live_collection_step(env, evaluate, *, horizon_s, radius_m,
                                   distance_scale_m):
    """Collect through the existing profile0/7 interface and reentry gate.

    evaluate(frozen_params, parent_obs, parent_critic_obs, goal_context) must
    return raw Gaussian mean[2], G probability, scalar C value, S probability.
    Ineligible rows receive finite zero dummy views, then are discarded. The
    caller must retain these exact sampled old values/logp across PPO epochs.
    Goal latch is physical telemetry; reaching a goal does not stop balance.
    """
    from .anticipatory_reference import make_reference_step
    dt = float(env.cc.dt)
    limit = round(horizon_s / dt)
    if (not all(math.isfinite(x) and x > 0 for x in
                (dt, horizon_s, radius_m, distance_scale_m)) or limit % 4 or
            not math.isclose(limit * dt, horizon_s, abs_tol=1e-8)):
        raise ValueError('positive aligned horizon, radius and explicit distance scale required')
    advance = make_reference_step(env, _SampledAction(), horizon_s=horizon_s, allow_reentry=True)
    lower_bound = j.float32(env.config['residual_interface']['speed_reference_bounds_m_s'][0])

    def step(carry, params, log_std, path, speed, pulse, profile, original_goal_xy):
        goal = j.asarray(original_goal_xy)
        if goal.shape != (2,):
            raise ValueError('explicit original world goal XY required, not path.goal arc length')
        s = carry.state
        stopped = s.physical.failed | s.fault | (s.tick >= limit)
        eligible = actor_permission(s.domain_exit, s.follower.invalid, carry.clean_blocks,
                                    j.bool_(False), stopped)
        obs, critic_obs, obs_fault = env.observation(s, path, speed, horizon_s)
        _, pose, _ = env.physics.observe(s.physical.data)
        context = original_goal_context(pose, goal, tick=s.tick, dt=dt,
            horizon_s=horizon_s, arrived=carry.arrived, radius_m=radius_m,
            distance_scale_m=distance_scale_m)
        protected = [j.where(eligible, x, j.zeros_like(x)) for x in (obs, critic_obs, context)]
        mean, g, c, survival = evaluate(params, *protected)
        mean = j.asarray(mean)
        if mean.shape != (2,) or any(j.shape(x) != () for x in (g, c, survival)):
            raise ValueError('two raw action means and three scalar values required')
        raw, logp, next_key = sample_gaussian_action(mean, log_std, carry.key)
        finite = (j.all(j.isfinite(mean)) & j.all(j.isfinite(raw)) & j.isfinite(logp) &
                  j.isfinite(g) & j.isfinite(c) & j.isfinite(survival) &
                  (g >= 0) & (g <= 1) & (survival >= 0) & (survival <= 1))
        fault = eligible & (obs_fault | ~j.all(j.isfinite(context)) | ~finite |
                            ~((profile == 0) | (profile == 7)))
        # A malformed original goal is always an engineering fault, even if
        # policy permission is temporarily unavailable.
        fault = fault | (~stopped & ~j.all(j.isfinite(goal)))
        prepared = carry.replace(state=s.replace(fault=s.fault | fault))
        end, logs = advance(prepared, dict(latent=raw, support_independent_permission=False),
                            path, speed, pulse, profile, lower_bound, j.bool_(False))
        valid = j.any(logs['actor_action_valid']) & ~fault
        key = jax.tree.map(lambda new, old: j.where(valid, new, old), next_key, carry.key)
        arrived = goal_latch_from_block(carry.arrived, logs, goal, radius_m)
        end = end.replace(key=key, arrived=arrived)
        decision = dict(valid=valid,
            observation=j.where(valid, obs, j.full_like(obs, j.nan)),
            critic_observation=j.where(valid, critic_obs, j.full_like(critic_obs, j.nan)),
            goal_context=j.where(valid, context, j.full_like(context, j.nan)),
            old_goal_value=j.where(valid, g, j.nan), old_cost_value=j.where(valid, c, j.nan),
            physical_value=j.where(valid, survival, j.nan),
            raw_action=j.where(valid, raw, j.full_like(raw, j.nan)),
            mean=j.where(valid, mean, j.full_like(mean, j.nan)), log_prob=j.where(valid, logp, j.nan),
            task_tick=j.where(valid, s.tick, j.int32(-1)), rng_key=carry.key, proposal_fault=fault)
        logs = dict(logs, upper_active=logs['active_tick'] & logs['actor_action_valid'],
                    fallback_active=logs['active_tick'] & ~logs['actor_action_valid'])
        return end, (decision, logs)

    return step
