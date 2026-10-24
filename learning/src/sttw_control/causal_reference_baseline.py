"""Simple DEV baseline: causal ESO event, restricted reference rule, handback.

This is not a disturbance-tolerance estimator or a validated safety controller.
The function receives no pulse, force, onset, capture time or eventual outcome.
"""
from typing import NamedTuple

import jax.numpy as j

from .reference_prefix import inverse_offset_target


LATEST_ESO = 318
PREVIOUS_ESO = 298
PREVIOUS_FRAME_MASK = 334
LATEST_FRAME_MASK = 335
ALPHA = 336
CURVATURE_HERE = 340
REQUESTED_SPEED = 345
THRESHOLD = .015
DURATION_TICKS = 80  # Existing .4s prefix at the fixed .005s lower period.


def validate_causal_contract(config, frame_fields):
    """Preflight actual field names/scales and timing, not just total dimension."""
    n, timing = config['network'], config['timing']
    expected = dict(history_frames=16, frame_fields=20, mask_fields=16, actor_dim=351,
                    history_dt_s=.02)
    if any(n.get(k) != v for k, v in expected.items()):
        raise ValueError('causal baseline history contract differs')
    if timing['lower_dt_s'] != .005 or timing['upper_dt_s'] != .02:
        raise ValueError('causal baseline requires existing .005/.02 timing')
    if len(frame_fields) != 20 or frame_fields[18] != dict(name='eso_equilibrium_shift_rad', scale=.3):
        raise ValueError('ESO field identity or scale differs')
    context = n['context_fields']
    if len(context) != 15:
        raise ValueError('context field count differs')
    for index, name, scale in ((0, 'alpha', 1.), (4, 'curvature_here_1_m', .6),
                               (9, 'external_speed_m_s', 3.)):
        if context[index] != dict(name=name, scale=scale):
            raise ValueError('causal context field identity or scale differs')


class GateState(NamedTuple):
    fired: object
    start_tick: object
    target_offset: object
    fixed_latent: object
    mask: object


def initial_gate():
    return GateState(j.bool_(False), j.int32(-1), j.zeros(2), j.zeros(2), j.zeros(2, dtype=j.bool_))


def validate_full_task_origin(task_tick, physical_tick, initial_physical_tick, gate):
    """Host preflight: do not silently restart detection at an onset snapshot."""
    if int(task_tick) != 0 or int(physical_tick) != int(initial_physical_tick):
        raise ValueError('causal validation must start at the declared complete task origin')
    if (bool(gate.fired) or int(gate.start_tick) != -1 or bool(j.any(gate.mask))
            or bool(j.any(gate.target_offset != 0)) or bool(j.any(gate.fixed_latent != 0))):
        raise ValueError('gate must initialize once at task origin, never at pulse onset')


def case_has_engineering_error(final_fault, balance_label):
    """Transient logged numerical faults cannot become completed fall labels."""
    return bool(final_fault or balance_label == 'ENGINEERING_ERROR')


def causal_reference_action(original, observation, current_offset, tick, state, config):
    """One 50Hz decision, using only current/past deployable information.

    Restricted DEV rules: alpha0, requested 1.5m/s right turn and negative
    innovation -> speed target +.15; requested 1.8m/s left turn and positive
    innovation -> steer target -.04. Other first events latch without action.
    Targets use the *detection-time* offset, unlike the known-onset diagnostic.
    """
    if observation.shape != (351,):
        raise ValueError('requires the existing 351D actor observation')
    engineering_fault = (~j.all(j.isfinite(current_offset)) | ~j.all(j.isfinite(original))
                         | ~j.all(j.isfinite(observation)) | (tick < 0) | (tick % 4 != 0))
    signal = -(observation[LATEST_ESO] - observation[PREVIOUS_ESO])
    valid = (j.all(j.isfinite(observation))
             & (observation[PREVIOUS_FRAME_MASK] > .5) & (observation[LATEST_FRAME_MASK] > .5)
             & (j.abs(observation[LATEST_ESO]) < 5) & (j.abs(observation[PREVIOUS_ESO]) < 5))
    detected = ~state.fired & valid & ~engineering_fault & (j.abs(signal) >= THRESHOLD)
    alpha0 = j.abs(observation[ALPHA]) < 1e-6
    speed = 3. * observation[REQUESTED_SPEED]
    curvature = .6 * observation[CURVATURE_HERE]
    speed_rule = alpha0 & (j.abs(speed - 1.5) <= .05) & (curvature < 0) & (signal < 0)
    steer_rule = alpha0 & (j.abs(speed - 1.8) <= .05) & (curvature > 0) & (signal > 0)
    mask = j.stack([speed_rule, steer_rule])
    target = current_offset + j.array([j.where(speed_rule, .15, 0.), j.where(steer_rule, -.04, 0.)])
    latent, realized, clipped = inverse_offset_target(target, config)
    next_state = GateState(state.fired | detected,
                           j.where(detected, tick, state.start_tick),
                           j.where(detected, target, state.target_offset),
                           j.where(detected, latent, state.fixed_latent),
                           j.where(detected, mask, state.mask))
    active = ~engineering_fault & next_state.fired & j.any(next_state.mask) & (tick >= next_state.start_tick) & (tick < next_state.start_tick + DURATION_TICKS)
    action = j.where(active & next_state.mask, next_state.fixed_latent, original)
    return action, next_state, dict(signal=signal, valid=valid, detected=detected,
                                   engineering_fault=engineering_fault,
                                   override_active=active,
                                   target_clipped_channels=detected & mask & clipped,
                                   realized_target_at_detection=j.where(detected, realized, j.zeros(2)))
