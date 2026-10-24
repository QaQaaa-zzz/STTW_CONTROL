"""Candidate extension of the DEV reference rule to the opposite turn side.

Only gate coordinates are transformed. Neither the plant nor either learned
policy is assumed reflection equivariant. This is not a tolerance estimator.
The two speed bands, alpha0 restriction and old event/action budget remain.
"""
import jax.numpy as j

from .causal_reference_baseline import (
    LATEST_ESO, PREVIOUS_ESO, CURVATURE_HERE, REQUESTED_SPEED,
    causal_reference_action,
)


def reflected_reference_action(original, observation, current_offset, tick, state, config):
    """Same causal interface and physical-coordinate GateState as the baseline.

    At 1.5m/s map either turn to the old right-turn event; elsewhere map to
    the old left-turn event, which can act only in its existing 1.8m/s band.
    State is transformed AND mapped back each decision so a later curvature
    reversal never reverses the stored physical target or fixed action.
    """
    turn = j.where(observation[CURVATURE_HERE] < 0, -1., 1.)
    speed = 3. * observation[REQUESTED_SPEED]
    factor = j.where(j.abs(speed - 1.5) <= .05, -turn, turn)
    channel_signs = j.array([1., factor])
    event_indices = j.array([LATEST_ESO, PREVIOUS_ESO, CURVATURE_HERE])
    canonical_observation = observation.at[event_indices].multiply(factor)
    canonical_state = state._replace(target_offset=state.target_offset * channel_signs,
                                     fixed_latent=state.fixed_latent * channel_signs)
    action, updated, telemetry = causal_reference_action(
        original * channel_signs, canonical_observation, current_offset * channel_signs,
        tick, canonical_state, config)
    updated = updated._replace(target_offset=updated.target_offset * channel_signs,
                               fixed_latent=updated.fixed_latent * channel_signs)
    telemetry = dict(telemetry, signal=telemetry['signal'] * factor,
                     realized_target_at_detection=telemetry['realized_target_at_detection'] * channel_signs)
    return action * channel_signs, updated, telemetry


def select_reference_rule(variant):
    if variant == 'baseline':
        return causal_reference_action
    if variant == 'reflected_v1':
        return reflected_reference_action
    if variant == 'parent_following_v1':
        from .following_causal_reference import following_causal_reference_action
        return following_causal_reference_action
    raise ValueError('unknown causal reference rule: ' + str(variant))


def initial_reference_state(variant):
    select_reference_rule(variant)  # Reject unknown variants before state creation.
    if variant == 'parent_following_v1':
        from .following_causal_reference import initial_following_state
        return initial_following_state()
    from .causal_reference_baseline import initial_gate
    return initial_gate()
