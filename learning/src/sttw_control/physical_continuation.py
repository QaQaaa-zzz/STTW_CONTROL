"""Opt-in fixed-reference lower continuation, separate from task-policy data.

Caller latches the last governed reference at handover and guards physical
contact/fault/time termination. No task observation is constructed here. The
returned structured risk view has no trained Critic attached and is not a claim
that sensor measurements alone form a Markov state.
"""
import math
import jax
import jax.numpy as j


def lower_risk_observation(state, *, measurement, pose, true_speed,
                           held_reference, remaining_s):
    """Live lower view, including controller/actuator/local recurrent state.

    Privileged simulator speed is explicitly identified. Upper history is stale
    after handover and intentionally absent. `valid` identifies an actionable
    nonterminal observation, not a survival label or a complete risk dataset.
    Full simulator state is still retained by the caller, never reset here.
    """
    view = dict(measurement=measurement, pose=pose,
        privileged_true_speed=true_speed, held_reference=held_reference,
        remaining_s=remaining_s, controller_state=state.physical.controller,
        actuator_state=state.physical.actuator, lower_state=state.lower,
        physical_tick=state.physical.physical_tick,
        task_tick=state.tick, tracking_domain_exit=state.domain_exit)
    finite = j.all(j.stack([j.all(j.isfinite(x)) for x in jax.tree.leaves(view)]))
    view.update(valid=finite & ~state.fault & ~state.physical.failed & (remaining_s > 0),
                upper_action_valid=j.bool_(False))
    return view


def make_hold_last_tick(lower_tick, *, dt):
    """Reuse actual lower kernel, never invoke the task Actor outside its domain.

    One active physical tick only; termination/padding belongs to the caller.
    Original upper history/projection/filter fields remain untouched and must
    not be treated as live task observations after this handover.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('positive finite physical dt required')

    def advance(state, pulse, relative_time, held_reference):
        physical, lower, log = lower_tick(state.physical, state.lower,
            pulse, relative_time, held_reference)
        end = state.replace(physical=physical, lower=lower, tick=state.tick+1,
            fault=state.fault | log['lower_fault'] | log['nonfinite'],
            yaw_rate=(physical.yaw_unwrapped-state.physical.yaw_unwrapped)/dt)
        return end, dict(log, upper_active=j.bool_(False), path_features_valid=j.bool_(False))

    return advance
