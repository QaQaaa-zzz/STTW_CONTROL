"""Task-directed reference-bias opportunity tools around a frozen middle Actor.

This is an offline search interface, not a new Actor, online planner, disturbance
predictor or safety certificate. Search scoring must evaluate the full real
closed loop. Neither the Actor adapter nor the decision rule reads an event.
"""
import numpy as np
import jax.numpy as j

from .full_task_reference_sequence import sequence_target
from .path_command_policy import map_latent
from .reference_prefix import inverse_offset_target

BIAS_SCALE = (.25, .05)  # m/s and rad, before ORIGINAL target/filter/slew limits


def bias_targets(vector):
    """Six bounded coordinates; three internal knots, zero outer knots."""
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (6,) or not np.isfinite(vector).all() or np.any(abs(vector)>1):
        raise ValueError('six finite normalized bias coordinates in [-1,1] required')
    targets = np.zeros((5, 2), np.float32)
    targets[1:4] = vector.reshape(3, 2)*np.asarray(BIAS_SCALE)
    return targets


def add_physical_bias(native, bias, config):
    """Retain each zero-bias latent exactly, including saturated native outputs.

    Nonzero biases project to the same asymmetric original reference authority;
    downstream fixed feedback, filtering, slew and antiwindup still apply.
    Nonfinite native/bias values remain nonfinite, so faults cannot be hidden by
    tanh or the inverse's clip. No observation, alpha or internal state changes.
    """
    proposed, realized, clipped = inverse_offset_target(map_latent(native, config)+bias, config)
    action = j.where(bias == 0, native, proposed)
    action = j.where(j.isfinite(native), action, native)
    action = j.where(j.isfinite(bias), action, bias)
    return action, map_latent(action, config), (bias != 0) & clipped


class BiasedActor:
    def __init__(self, native, config):
        self.native, self.config = native, config

    def apply(self, parameters, observation):
        return add_physical_bias(self.native.apply(parameters['native'], observation),
                                 parameters['physical_bias'], self.config)[0]


def make_bias_step(env, actor, *, horizon_s, spec):
    """One unchanged goal/feedback/physical advance, with a causal progress bias.

    Parameters retain the original frozen Actor and full-channel mask. The
    complete incoming carry is forwarded once. Positions are task geometry and
    targets are selected before execution, not force/onset-conditioned data.
    """
    from .goal_preview_baseline import make_goal_step
    advance = make_goal_step(env, BiasedActor(actor, env.config), horizon_s=horizon_s, spec=spec)

    def step(carry, parameters, path, speed, pulse, mode):
        bias = sequence_target(carry.state.projection.progress,
                               parameters['bias_positions'], parameters['bias_targets'])
        source = parameters['actor']
        wrapped = dict(parameters, actor=dict(source, native=dict(
            native=source['native'], physical_bias=bias)))
        end, logs = advance(carry, wrapped, path, speed, pulse, mode)
        obs, _, _ = env.observation(carry.state, path, speed, horizon_s)
        bare = actor.apply(source['native'], j.where(logs['actor_action_valid'][0], obs, j.zeros_like(obs)))
        logs.update(coordinated_actor_latent=logs['original_actor_latent'],
                    original_actor_latent=j.where(logs['actor_action_valid'][:, None], j.tile(bare, (4, 1)), j.nan),
                    reference_bias_target=j.tile(bias, (4, 1)),
                    reference_bias_progress=j.repeat(carry.state.projection.progress, 4))
        return end, logs
    return step


def verify_bias_log(raw, *, positions, targets, case, config, spec, path):
    """Independent NumPy reconstruction, followed by all original goal audits."""
    from .goal_preview_baseline import verify_goal_log
    legal = np.asarray(raw['actor_action_valid'], bool)
    active = np.asarray(raw['active_tick'], bool)
    progress = np.asarray(raw['reference_bias_progress'])
    positions, targets = np.asarray(positions), np.asarray(targets)
    interval = np.clip(np.searchsorted(positions, progress, side='right')-1, 0, 3)
    u = np.clip((progress-positions[interval])/(positions[interval+1]-positions[interval]), 0, 1)
    bias = targets[interval]+(u*u*(3-2*u))[:, None]*(targets[interval+1]-targets[interval])
    np.testing.assert_allclose(raw['reference_bias_target'][active], bias[active], atol=1e-7)
    native = np.asarray(raw['original_actor_latent'])[legal]
    coordinated = np.asarray(raw['coordinated_actor_latent'])[legal]
    c = config['residual_interface'];t = np.tanh(native)
    scales = np.column_stack([np.where(t[:, 0]>=0, c['speed_positive_scale_m_s'], c['speed_negative_scale_m_s']), np.full(len(t), c['steer_scale_rad'])])
    desired = t*scales+bias[legal]
    normalized = np.column_stack([desired[:, 0]/np.where(desired[:, 0]>=0, c['speed_positive_scale_m_s'], c['speed_negative_scale_m_s']), desired[:, 1]/c['steer_scale_rad']])
    expected = np.where(bias[legal]==0, t, np.clip(normalized, -1+1e-6, 1-1e-6))
    np.testing.assert_allclose(np.tanh(coordinated), expected, atol=3e-6)
    zero = bias[legal]==0
    np.testing.assert_array_equal(coordinated[zero], native[zero])
    verify_goal_log(dict(raw, original_actor_latent=raw['coordinated_actor_latent']),
                    case, config, spec, path, expected_keep=np.ones((len(active), 2), bool))


def minimize_nominal_cost(evaluate, *, seed, population=16, generations=4, method='cem'):
    """Fixed-budget six-dimensional CEM, preserving a scored zero incumbent.

    evaluate(batch) returns aligned nominal objective values: +inf denotes an
    observed infeasible candidate; NaN/-inf is an unknown/error, never a failed
    control label. A baseline +inf stops this nominal-retention experiment.
    No actual disturbance event can be passed as a per-rollout query here.
    Every generation includes the current incumbent; ALL callback evaluations
    (including its duplicates) remain chargeable simulation interactions.
    """
    if population != 16 or generations != 4:
        raise ValueError('fixed pilot requires four generations of sixteen')
    if method not in ['cem', 'random']:
        raise ValueError('CEM and matched fixed-distribution random control only')
    rng = np.random.default_rng(seed)
    mean = np.zeros(6); std = np.full(6, .5)
    incumbent = np.zeros(6); best = float('inf'); history = []
    for generation in range(generations):
        candidates = np.clip(mean+std*rng.standard_normal((population, 6)), -1, 1)
        candidates[0] = incumbent
        scores = np.asarray(evaluate(candidates.copy()), dtype=float)
        if scores.shape != (population,) or np.isnan(scores).any() or np.isneginf(scores).any():
            raise ValueError('complete aligned scores required; unknown is not infeasible')
        if not np.isfinite(scores[0]):
            raise ValueError('retained nominal incumbent is infeasible; stop, do not replace baseline')
        winner = int(np.argmin(scores))
        if scores[winner] < best:
            incumbent, best = candidates[winner].copy(), float(scores[winner])
        feasible = np.flatnonzero(np.isfinite(scores))
        elite = feasible[np.argsort(scores[feasible], kind='stable')[:4]]
        if method == 'cem':
            mean = .4*mean+.6*candidates[elite].mean(axis=0)
            std = np.maximum(.1, .4*std+.6*candidates[elite].std(axis=0))
        history.append(dict(generation=generation, candidates=candidates.tolist(),
                            scores=[float(s) if np.isfinite(s) else None for s in scores],
                            feasible=np.isfinite(scores).tolist(),
                            incumbent=incumbent.tolist(), best_cost=best))
    return dict(vector=incumbent, cost=best, history=history, method=method,
                scored_candidates=population*generations)
