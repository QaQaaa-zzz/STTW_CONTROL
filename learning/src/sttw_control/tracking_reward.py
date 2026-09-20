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
    precision_reward: bool = False
    precision_under_rate_low: float = .12
    precision_under_rate_high: float = 8.
    precision_under_scale_low: float = .1
    precision_under_scale_high: float = .05
    precision_over_rate: float = 8.
    precision_over_scale: float = .05
    precision_path_rate_low: float = 4.
    precision_path_rate_high: float = .1
    precision_reward_scale: float = .1
    precision_cost_cap: float = 100.
    geometric: bool = False
    reward_mode: str = 'gaussian'
    shrink_tolerances: bool = False
    deadline_penalty: float = 0.
    over_deadline_rate: float = 0.
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
    final_overspeed_tolerance: float = .2
    overspeed_band: float = .2
    final_lateral_tolerance: float = .1
    final_heading_tolerance: float = .15
    final_roll_rate_tolerance: float = .3
    start_seconds: float = 1.
    timed: bool = False
    objective: str = 'legacy'  # 'geometric_huber': pure speed vs geometric path
    overdue_rate: float = 0.  # per second while still pending after deadline
    longitudinal_wide: float = .5
    longitudinal_fine: float = .15
    longitudinal_scale: float = .2
    longitudinal_tight: float = .15
    longitudinal_relaxed: float = .5
    yaw_rate_wide: float = .4
    yaw_rate_fine: float = .1
    yaw_rate_scale: float = .2
    yaw_rate_tight: float = .15
    yaw_rate_relaxed: float = .4
    final_longitudinal_tolerance: float = .15
    final_yaw_rate_tolerance: float = .15

    def __post_init__(self):
        nonnegative = {'start_seconds', 'tail_rate', 'budget_rate', 'return_rate',
                       'return_bonus', 'action_weight', 'action_delta_weight',
                       'deadline_penalty', 'over_deadline_rate', 'overdue_rate'}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == 'reward_mode':
                if value not in ('gaussian', 'huber'):
                    raise ValueError('reward_mode must be gaussian or huber')
                continue
            if field.name == 'objective':
                if value not in ('legacy', 'geometric_huber', 'asymmetric_geometric_huber'):
                    raise ValueError('unsupported tracking objective')
                continue
            if field.name in ('timed', 'geometric', 'shrink_tolerances', 'precision_reward'):
                if not isinstance(value, bool):
                    raise ValueError(f'{field.name} must be boolean')
                continue
            if not math.isfinite(value) or (value < 0 if field.name in nonnegative else value <= 0):
                raise ValueError(f'invalid tracking parameter {field.name}')
        if self.shrink_tolerances and (
                self.return_seconds <= self.hold_seconds
                or self.speed_tight < self.final_speed_tolerance
                or self.lateral_tight < self.final_lateral_tolerance):
            raise ValueError('shrinking bands require T > hold and initial bands >= final bands')
        if self.reward_mode == 'huber' and self.timed and not self.geometric:
            raise ValueError('timed huber tracking requires geometric mode')
        if self.objective == 'geometric_huber' and (self.timed or self.return_bonus != 0 or self.tail_rate != 0):
            raise ValueError('geometric_huber excludes timed objectives, recovery bonus and duplicate tails')
        if self.objective == 'asymmetric_geometric_huber' and ((self.timed and not self.geometric) or self.return_bonus != 0 or self.tail_rate != 0):
            raise ValueError('asymmetric geometric huber requires geometric tracking and excludes bonus/tails')

        if self.priority_ratio < 1 or self.return_seconds < self.hold_seconds:
            raise ValueError('invalid priority ratio or recovery window')
        if self.precision_reward and self.objective != 'asymmetric_geometric_huber':
            raise ValueError('precision_reward requires asymmetric_geometric_huber')
        if any(tight > relaxed for tight, relaxed in (
                (self.speed_tight, self.speed_relaxed), (self.lateral_tight, self.lateral_relaxed),
                (self.longitudinal_tight, self.longitudinal_relaxed),
                (self.yaw_rate_tight, self.yaw_rate_relaxed))):
            raise ValueError('tight tolerance must not exceed relaxed tolerance')
        if any(wide < fine for wide, fine in ((self.speed_wide, self.speed_fine),
               (self.lateral_wide, self.lateral_fine), (self.heading_wide, self.heading_fine),
               (self.longitudinal_wide, self.longitudinal_fine),
               (self.yaw_rate_wide, self.yaw_rate_fine))):
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


def directional_speed_path_rates(speed_error, lateral_error, heading_error, alpha,
                                 return_elapsed, config, *, xp=jp):
    """One authoritative asymmetric speed/path cost implementation.

    Error is actual forward speed minus the already-active external reference.
    The overspeed weight and band are independent of alpha. ``return_elapsed``
    is the existing non-resetting recovery debt clock.
    """
    c = config
    ev, ey, ep = (xp.nan_to_num(xp.asarray(value), nan=0., posinf=0., neginf=0.)
                  for value in (speed_error, lateral_error, heading_error))
    alpha = xp.clip(xp.nan_to_num(xp.asarray(alpha), nan=.5), 0., 1.)
    under, over = xp.maximum(-ev, 0.), xp.maximum(ev, 0.)
    wv, wp = preference_weights(alpha, c, xp=xp)
    wover = c.priority_ratio / (c.priority_ratio + 1.)
    under_band, over_band, path_band = directional_tolerances(
        alpha, return_elapsed, c, xp=xp)
    under_rate, over_rate, path_rate = c.tracking_rate * wv, c.tracking_rate * wover, c.tracking_rate * wp
    under_scale = over_scale = c.speed_scale
    if c.precision_reward:
        under_rate = c.precision_under_rate_low + alpha * (c.precision_under_rate_high-c.precision_under_rate_low)
        path_rate = c.precision_path_rate_low + alpha * (c.precision_path_rate_high-c.precision_path_rate_low)
        under_scale = c.precision_under_scale_low + alpha * (c.precision_under_scale_high-c.precision_under_scale_low)
        over_rate, over_scale = c.precision_over_rate, c.precision_over_scale
    return {
        'underspeed_tracking': -under_rate * huber_tail(under / under_scale, xp=xp),
        'overspeed_tracking': -over_rate * huber_tail(over / over_scale, xp=xp),
        'path_tracking': -path_rate * (
            huber_tail(ey / c.lateral_scale, xp=xp)
            + c.heading_tail_weight * huber_tail(ep / c.heading_scale, xp=xp)),
        'underspeed_budget': -c.budget_rate * huber_tail(
            xp.maximum(under - under_band, 0.) / under_scale, xp=xp),
        'overspeed_budget': -c.budget_rate * huber_tail(
            xp.maximum(over - over_band, 0.) / over_scale, xp=xp),
        'path_budget': -c.budget_rate * huber_tail(
            xp.maximum(xp.abs(ey) - path_band, 0.) / c.lateral_scale, xp=xp),
    }


def directional_tolerances(alpha, return_elapsed, config, *, xp=jp):
    alpha = xp.clip(xp.asarray(alpha), 0., 1.)
    initial_under = config.speed_relaxed + alpha * (config.speed_tight - config.speed_relaxed)
    initial_path = config.lateral_tight + alpha * (config.lateral_relaxed - config.lateral_tight)
    fraction = (xp.clip(return_elapsed / (config.return_seconds - config.hold_seconds), 0., 1.)
                if config.shrink_tolerances else xp.asarray(0.))
    return (initial_under + fraction * (config.final_speed_tolerance - initial_under),
            xp.asarray(config.overspeed_band),
            initial_path + fraction * (config.final_lateral_tolerance - initial_path))


def timed_tolerances(alpha, config, *, xp=jp):
    """Along-track position loosens and yaw-rate tightens toward command priority."""
    alpha = xp.clip(xp.asarray(alpha), 0., 1.)
    return (config.longitudinal_tight + alpha * (config.longitudinal_relaxed - config.longitudinal_tight),
            config.yaw_rate_relaxed + alpha * (config.yaw_rate_tight - config.yaw_rate_relaxed))


def within_final(roll, roll_rate, speed_error, lateral_error, heading_error, config, *,
                 longitudinal_error=None, yaw_rate_error=None, xp=jp):
    speed_ok = ((speed_error >= -config.final_speed_tolerance)
                & (speed_error <= config.final_overspeed_tolerance)
                if config.objective == 'asymmetric_geometric_huber'
                else xp.abs(speed_error) <= config.final_speed_tolerance)
    final = ((xp.abs(roll) <= config.roll_working_limit)
            & (xp.abs(roll_rate) <= config.final_roll_rate_tolerance)
            & speed_ok
            & (xp.abs(lateral_error) <= config.final_lateral_tolerance)
            & (xp.abs(heading_error) <= config.final_heading_tolerance))
    if config.timed and not config.geometric:
        if longitudinal_error is None or yaw_rate_error is None:
            raise ValueError('timed tracking requires longitudinal_error and yaw_rate_error')
        final = (final & (xp.abs(longitudinal_error) <= config.final_longitudinal_tolerance)
                 & (xp.abs(yaw_rate_error) <= config.final_yaw_rate_tolerance))
    return final


def transition(state, *, roll, roll_rate, speed_error, lateral_error, heading_error,
               action, alpha, dt, alive_rate, failure_penalty, failed,
               enabled, config, longitudinal_error=None, yaw_rate_error=None,
               recovery_trigger=False, clock_from_departure=True, xp=jp):
    """One transition, using pre-action alpha and the resulting physical errors.

    The clock starts at observed departure, not an oracle disturbance-end label.
    It runs during forcing too: the allowance bounds total off-target duration.
    ``enabled`` excludes the declared initialization settling window only.
    Deployment needs the same localization and forward-velocity estimate used
    for these tracking-state updates; wheel odometry alone is not ground truth.
    """
    c = config
    if c.timed and (longitudinal_error is None or yaw_rate_error is None):
        raise ValueError('timed tracking requires longitudinal_error and yaw_rate_error')
    failed = xp.asarray(failed)
    action = xp.asarray(action)
    # Simulator invalid-state termination must still return a finite -penalty.
    ev, ey, ep, phi, rate = [xp.nan_to_num(xp.asarray(x), nan=0., posinf=0., neginf=0.)
                            for x in (speed_error, lateral_error, heading_error, roll, roll_rate)]
    a = xp.nan_to_num(action, nan=0., posinf=0., neginf=0.)
    alpha = xp.clip(xp.nan_to_num(xp.asarray(alpha), nan=.5), 0., 1.)
    wv, wp = preference_weights(alpha, c, xp=xp)
    bv, by = tolerances(alpha, c, xp=xp)
    ex, ew = (None, None)
    if c.timed:
        ex, ew = [xp.nan_to_num(xp.asarray(x), nan=0., posinf=0., neginf=0.)
                  for x in (longitudinal_error, yaw_rate_error)]
    final = within_final(phi, rate, ev, ey, ep, c, longitudinal_error=ex,
                         yaw_rate_error=ew, xp=xp) & ~failed
    left = xp.asarray(enabled) & ~final & ~failed
    armed = (state.pending | xp.asarray(recovery_trigger)
             | (left & xp.asarray(clock_from_departure))) & ~failed
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
    if c.objective in ('geometric_huber', 'asymmetric_geometric_huber'):
        # Completion remains diagnostic; late return must not erase a violation.
        earned = earned & ~missed
    nxt = ReturnState(pending, xp.where(pending, elapsed, 0.),
                      xp.minimum(hold, c.hold_seconds), state.credited | earned,
                      state.ever_left | left, missed, a)
    if c.shrink_tolerances:
        fraction = xp.clip(elapsed / (c.return_seconds - c.hold_seconds), 0., 1.)
        bv = bv + fraction * (c.final_speed_tolerance - bv)
        by = by + fraction * (c.final_lateral_tolerance - by)
    speed_bonus = .5 * (xp.exp(-(ev / c.speed_wide) ** 2) + xp.exp(-(ev / c.speed_fine) ** 2))
    path_bonus = .5 * (xp.exp(-(ey / c.lateral_wide) ** 2 - (ep / c.heading_wide) ** 2)
                       + xp.exp(-(ey / c.lateral_fine) ** 2 - (ep / c.heading_fine) ** 2))
    if c.timed and not c.geometric:
        path_bonus = .5 * (xp.exp(-(ex / c.longitudinal_wide) ** 2
                                 - (ey / c.lateral_wide) ** 2 - (ep / c.heading_wide) ** 2)
                           + xp.exp(-(ex / c.longitudinal_fine) ** 2
                                    - (ey / c.lateral_fine) ** 2 - (ep / c.heading_fine) ** 2))
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
    if c.objective == 'asymmetric_geometric_huber':
        directional = directional_speed_path_rates(ev, ey, ep, alpha, elapsed, c, xp=xp)
        rates.update(directional)
        rates['speed_tracking'] = xp.zeros_like(ev)
        rates['speed_tail'] = xp.zeros_like(ev)
        rates['path_tail'] = xp.zeros_like(ey)
        rates['speed_budget'] = xp.zeros_like(ev)
    if c.reward_mode == 'huber' and c.objective != 'asymmetric_geometric_huber':
        rates['speed_tracking'] = -c.tracking_rate * wv * huber_tail(ev / c.speed_scale, xp=xp)
        rates['path_tracking'] = -c.tracking_rate * wp * (
            huber_tail(ey / c.lateral_scale, xp=xp)
            + c.heading_tail_weight * huber_tail(ep / c.heading_scale, xp=xp))
        rates['speed_tail'] = xp.zeros_like(ev)
        rates['path_tail'] = xp.zeros_like(ey)
    if c.timed and c.geometric:
        # Retain timed diagnostic names without rewarding time phase or yaw-rate.
        rates.update({name: xp.zeros_like(ev) for name in (
            'yaw_rate_tracking', 'yaw_rate_tail', 'yaw_rate_budget', 'longitudinal_budget')})
    if c.timed and not c.geometric:
        bx, bw = timed_tolerances(alpha, c, xp=xp)
        rates['speed_tracking'] *= .5
        rates['speed_tail'] *= .5
        rates['speed_budget'] *= .5
        rates['path_budget'] *= .5
        rates['path_tail'] -= c.tail_rate * wp * huber_tail(ex / c.longitudinal_scale, xp=xp)
        yaw_bonus = .5 * (xp.exp(-(ew / c.yaw_rate_wide) ** 2)
                          + xp.exp(-(ew / c.yaw_rate_fine) ** 2))
        rates.update(
            yaw_rate_tracking=.5 * c.tracking_rate * wv * yaw_bonus,
            yaw_rate_tail=-.5 * c.tail_rate * wv * huber_tail(ew / c.yaw_rate_scale, xp=xp),
            yaw_rate_budget=-.5 * c.budget_rate * huber_tail(
                xp.maximum(xp.abs(ew) - bw, 0.) / c.yaw_rate_scale, xp=xp),
            longitudinal_budget=-.5 * c.budget_rate * huber_tail(
                xp.maximum(xp.abs(ex) - bx, 0.) / c.longitudinal_scale, xp=xp),
        )
    overdue = armed & clock_on & (elapsed_ticks > math.floor(c.return_seconds / dt + 1e-9))
    if c.over_deadline_rate:
        rates['over_deadline'] = -c.over_deadline_rate * pending * overdue
    if c.objective == 'geometric_huber':
        rates['speed_tracking'] = -c.tracking_rate * wv * huber_tail(ev / c.speed_scale, xp=xp)
        rates['path_tracking'] = -c.tracking_rate * wp * (
            huber_tail(ey / c.lateral_scale, xp=xp)
            + c.heading_tail_weight * huber_tail(ep / c.heading_scale, xp=xp))
        rates['return_overdue'] = -c.overdue_rate * pending * clock_on * missed
    if c.precision_reward:
        # All non-alive rates are nonpositive in the asymmetric objective.
        # Preserve signed component accounting while bounding their sum once.
        cost = -sum(value for name,value in rates.items() if name != 'alive')
        factor = xp.minimum(1., c.precision_cost_cap / xp.maximum(cost, 1e-12))
        rates = {name: c.precision_reward_scale * value * (1. if name == 'alive' else factor)
                 for name,value in rates.items()}
    parts = {name: xp.where(failed, 0., dt * value) for name, value in rates.items()}
    if c.deadline_penalty or c.objective in ('geometric_huber', 'asymmetric_geometric_huber'):
        parts['deadline'] = xp.where(failed, 0., -c.deadline_penalty * (missed & ~state.deadline_missed))
    parts['recovery'] = xp.where(failed, 0., c.return_bonus * earned)
    parts['failure'] = xp.where(failed, -failure_penalty, 0.)
    return nxt, parts
