"""Alpha-conditioned geometric tracking rewards shared by CPU, MJX and audits.

All ordinary coefficients are reward/second. Failure replaces the transition;
recovery is a once-per-episode event. Alpha changes preference/tolerance only,
never actuator authority or the common roll/final-return requirements.
"""
from dataclasses import dataclass, fields
from typing import NamedTuple
import math
import jax.numpy as jp


@dataclass(frozen=True)
class TrackingConfig:
    priority_ratio: float = 10.
    tracking_rate: float = 4.
    speed_wide: float = .5
    speed_fine: float = .15
    lateral_wide: float = .4
    lateral_fine: float = .1
    heading_wide: float = .35
    heading_fine: float = .1
    tail_rate: float = .1
    speed_scale: float = .2
    lateral_scale: float = .2
    heading_scale: float = .15
    heading_tail_weight: float = .3
    budget_rate: float = 2.
    speed_tight: float = .2
    speed_relaxed: float = .5
    lateral_tight: float = .1
    lateral_relaxed: float = .4
    roll_working_limit: float = .3
    roll_weight: float = 100.
    roll_rate_weight: float = 1.
    action_weight: float = .01
    action_delta_weight: float = .02
    return_rate: float = .5
    return_bonus: float = 2.
    return_seconds: float = 3.
    hold_seconds: float = .5
    final_speed_tolerance: float = .2
    final_lateral_tolerance: float = .1
    final_heading_tolerance: float = .15
    final_roll_rate_tolerance: float = .3
    start_seconds: float = 1.

    def __post_init__(self):
        nonnegative = {'start_seconds', 'tail_rate', 'budget_rate', 'return_rate',
                       'return_bonus', 'action_weight', 'action_delta_weight'}
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or (value < 0 if field.name in nonnegative else value <= 0):
                raise ValueError(f'invalid tracking parameter {field.name}')
        if self.priority_ratio < 1 or self.return_seconds < self.hold_seconds:
            raise ValueError('invalid priority ratio or recovery window')
        if self.speed_tight > self.speed_relaxed or self.lateral_tight > self.lateral_relaxed:
            raise ValueError('tight tolerance must not exceed relaxed tolerance')
        if any(wide < fine for wide, fine in ((self.speed_wide, self.speed_fine),
               (self.lateral_wide, self.lateral_fine), (self.heading_wide, self.heading_fine))):
            raise ValueError('wide tracking scale must not be narrower than fine scale')


class ReturnState(NamedTuple):
    pending: object
    elapsed: object
    hold: object
    credited: object
    ever_left: object
    deadline_missed: object
    previous_action: object


def initial_return(*, xp=jp):
    return ReturnState(xp.asarray(False), xp.asarray(0.), xp.asarray(0.),
                       xp.asarray(False), xp.asarray(False), xp.asarray(False), xp.zeros(2))


def return_observation(state, config, *, xp=jp):
    # Raw seconds; normalization is declared in training.normalization.
    return xp.concatenate((xp.asarray(state.previous_action), xp.asarray([
        state.pending, state.elapsed, state.hold, state.credited,
        state.ever_left, state.deadline_missed])))


def preference_weights(alpha, config, *, xp=jp):
    alpha = xp.clip(xp.asarray(alpha), 0., 1.)
    ratio = config.priority_ratio
    return ((1. + (ratio - 1.) * alpha) / (ratio + 1.),
            (ratio - (ratio - 1.) * alpha) / (ratio + 1.))


def tolerances(alpha, config, *, xp=jp):
    alpha = xp.clip(xp.asarray(alpha), 0., 1.)
    return (config.speed_relaxed + alpha * (config.speed_tight - config.speed_relaxed),
            config.lateral_tight + alpha * (config.lateral_relaxed - config.lateral_tight))


def huber_tail(value, *, xp=jp):
    absolute = xp.abs(value)
    # The capped square avoids overflow in the unselected branch for large errors.
    return xp.minimum(absolute, 1.) ** 2 + 2. * xp.maximum(absolute - 1., 0.)


def within_final(roll, roll_rate, speed_error, lateral_error, heading_error, config, *, xp=jp):
    return ((xp.abs(roll) <= config.roll_working_limit)
            & (xp.abs(roll_rate) <= config.final_roll_rate_tolerance)
            & (xp.abs(speed_error) <= config.final_speed_tolerance)
            & (xp.abs(lateral_error) <= config.final_lateral_tolerance)
            & (xp.abs(heading_error) <= config.final_heading_tolerance))


def transition(state, *, roll, roll_rate, speed_error, lateral_error, heading_error,
               action, alpha, dt, alive_rate, failure_penalty, failed,
               enabled, config, xp=jp):
    """One transition, using pre-action alpha and the resulting physical errors.

    The clock starts at observed departure, not an oracle disturbance-end label.
    It runs during forcing too: the allowance bounds total off-target duration.
    ``enabled`` excludes the declared initialization settling window only.
    Deployment needs the same localization and forward-velocity estimate used
    for these tracking-state updates; wheel odometry alone is not ground truth.
    """
    c = config
    failed = xp.asarray(failed)
    action = xp.asarray(action)
    # Simulator invalid-state termination must still return a finite -penalty.
    ev, ey, ep, phi, rate = [xp.nan_to_num(xp.asarray(x), nan=0., posinf=0., neginf=0.)
                            for x in (speed_error, lateral_error, heading_error, roll, roll_rate)]
    a = xp.nan_to_num(action, nan=0., posinf=0., neginf=0.)
    alpha = xp.clip(xp.nan_to_num(xp.asarray(alpha), nan=.5), 0., 1.)
    wv, wp = preference_weights(alpha, c, xp=xp)
    bv, by = tolerances(alpha, c, xp=xp)
    final = within_final(phi, rate, ev, ey, ep, c, xp=xp) & ~failed
    left = xp.asarray(enabled) & ~final & ~failed
    armed = (state.pending | left) & ~failed
    clock_on = xp.asarray(enabled) & ~failed
    # Recover integer tick counts before incrementing: repeated float32 second
    # addition otherwise trips a 3 s deadline one tick early on MJX.
    elapsed_ticks = xp.where(armed, xp.rint(state.elapsed / dt).astype(xp.int32)
                            + clock_on.astype(xp.int32), 0)
    hold_ticks = xp.where(clock_on & final, xp.rint(state.hold / dt).astype(xp.int32) + 1, 0)
    elapsed, hold = elapsed_ticks * dt, hold_ticks * dt
    completed = armed & clock_on & final & (hold_ticks >= math.ceil(c.hold_seconds / dt - 1e-9))
    earned = completed & ~state.credited
    pending = armed & ~completed
    missed = state.deadline_missed | (armed & clock_on & (elapsed_ticks > math.floor(c.return_seconds / dt + 1e-9)))
    nxt = ReturnState(pending, xp.where(pending, elapsed, 0.),
                      xp.minimum(hold, c.hold_seconds), state.credited | earned,
                      state.ever_left | left, missed, a)
    speed_bonus = .5 * (xp.exp(-(ev / c.speed_wide) ** 2) + xp.exp(-(ev / c.speed_fine) ** 2))
    path_bonus = .5 * (xp.exp(-(ey / c.lateral_wide) ** 2 - (ep / c.heading_wide) ** 2)
                       + xp.exp(-(ey / c.lateral_fine) ** 2 - (ep / c.heading_fine) ** 2))
    rates = dict(
        alive=xp.asarray(alive_rate),
        speed_tracking=c.tracking_rate * wv * speed_bonus,
        path_tracking=c.tracking_rate * wp * path_bonus,
        speed_tail=-c.tail_rate * wv * huber_tail(ev / c.speed_scale, xp=xp),
        path_tail=-c.tail_rate * wp * (huber_tail(ey / c.lateral_scale, xp=xp)
                   + c.heading_tail_weight * huber_tail(ep / c.heading_scale, xp=xp)),
        speed_budget=-c.budget_rate * huber_tail(xp.maximum(xp.abs(ev) - bv, 0.) / c.speed_scale, xp=xp),
        path_budget=-c.budget_rate * huber_tail(xp.maximum(xp.abs(ey) - by, 0.) / c.lateral_scale, xp=xp),
        attitude=-c.roll_weight * xp.maximum(xp.abs(phi) - c.roll_working_limit, 0.) ** 2,
        roll_rate=-c.roll_rate_weight * rate ** 2,
        action=-c.action_weight * xp.sum(a ** 2, axis=-1),
        action_delta=-c.action_delta_weight * xp.sum((a - state.previous_action) ** 2, axis=-1),
        return_time=-c.return_rate * pending * clock_on * (1. + xp.minimum(elapsed / c.return_seconds, 1.)),
    )
    parts = {name: xp.where(failed, 0., dt * value) for name, value in rates.items()}
    parts['recovery'] = xp.where(failed, 0., c.return_bonus * earned)
    parts['failure'] = xp.where(failed, -failure_penalty, 0.)
    return nxt, parts
