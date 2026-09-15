"""Alpha-conditioned geometric tracking, shared by simulation and reward audit.

All regular terms are reward *rates*: multiply by the control period once.
A real failure replaces the entire transition reward. No positive clipping.
Timers depend only on observed path/attitude errors, never event labels.
This is a soft objective, not a stability certificate or a reference governor.
"""
from dataclasses import dataclass
import math
from typing import NamedTuple
import numpy as np


@dataclass(frozen=True)
class TrackingRewardConfig:
    priority_ratio: float = 10.0
    tracking_rate: float = 4.0
    speed_wide: float = 0.5
    speed_fine: float = 0.15
    lateral_wide: float = 0.4
    lateral_fine: float = 0.1
    heading_wide: float = 0.35
    heading_fine: float = 0.1
    tail_weight: float = 0.1
    speed_tail_scale: float = 0.2
    lateral_tail_scale: float = 0.2
    heading_tail_scale: float = 0.15
    heading_tail_weight: float = 0.3
    speed_tolerance_path_priority: float = 0.5
    speed_tolerance_speed_priority: float = 0.2
    lateral_tolerance_path_priority: float = 0.1
    lateral_tolerance_speed_priority: float = 0.4
    budget_weight: float = 2.0
    speed_excess_scale: float = 0.2
    lateral_excess_scale: float = 0.2
    roll_working_limit: float = 0.3
    roll_excess_weight: float = 100.0
    roll_rate_weight: float = 1.0
    action_weight: float = 0.01
    action_change_weight: float = 0.02
    return_lateral_tolerance: float = 0.1
    return_heading_tolerance: float = 0.15
    return_speed_tolerance: float = 0.2
    return_roll_rate_tolerance: float = 0.5
    hold_seconds: float = 0.5
    return_deadline_seconds: float = 3.0
    return_time_weight: float = 0.5
    return_overdue_weight: float = 2.0

    def __post_init__(self):
        nonnegative = {
            'tail_weight', 'heading_tail_weight', 'budget_weight',
            'roll_excess_weight', 'roll_rate_weight', 'action_weight',
            'action_change_weight', 'return_time_weight', 'return_overdue_weight',
        }
        for name, value in vars(self).items():
            if not math.isfinite(value) or (value < 0 if name in nonnegative else value <= 0):
                raise ValueError(f'invalid tracking reward parameter: {name}')
        if self.priority_ratio < 1:
            raise ValueError('priority_ratio must be >= 1')
        if self.speed_tolerance_path_priority < self.speed_tolerance_speed_priority:
            raise ValueError('speed tolerance must tighten as alpha increases')
        if self.lateral_tolerance_path_priority > self.lateral_tolerance_speed_priority:
            raise ValueError('path tolerance must relax as alpha increases')
        if self.return_deadline_seconds < self.hold_seconds:
            raise ValueError('return deadline must contain a complete hold interval')
        for wide, fine in ((self.speed_wide, self.speed_fine),
                           (self.lateral_wide, self.lateral_fine),
                           (self.heading_wide, self.heading_fine)):
            if wide < fine:
                raise ValueError('wide tracking scale must not be narrower than fine scale')


class TrackingState(NamedTuple):
    previous_action: object
    pending: object
    age_ticks: object
    path_hold_ticks: object
    ever_departed: object
    joint_hold_ticks: object
    task_recovered: object
    deadline_missed: object


def initial_tracking(*, xp=np):
    return TrackingState(xp.zeros(2, dtype=xp.float32), xp.asarray(False),
                         xp.asarray(0, dtype=xp.int32), xp.asarray(0, dtype=xp.int32),
                         xp.asarray(False), xp.asarray(0, dtype=xp.int32),
                         xp.asarray(False), xp.asarray(False))


def tracking_context(state, config, dt, *, xp=np):
    """Observable history fields; true speed and disturbance labels are excluded."""
    return xp.concatenate((state.previous_action,
        xp.stack((state.pending.astype(xp.float32),
                  xp.minimum(state.age_ticks * dt / config.return_deadline_seconds, 2.0),
                  xp.minimum(state.path_hold_ticks * dt / config.hold_seconds, 1.0)))))


def priority_weights(alpha, config, *, xp=np):
    alpha = xp.clip(xp.asarray(alpha), 0.0, 1.0)
    ratio = config.priority_ratio
    return ((1 + (ratio - 1) * alpha) / (ratio + 1),
            (ratio - (ratio - 1) * alpha) / (ratio + 1))


def tolerances(alpha, config, *, xp=np):
    alpha = xp.clip(xp.asarray(alpha), 0.0, 1.0)
    v = ((1 - alpha) * config.speed_tolerance_path_priority
         + alpha * config.speed_tolerance_speed_priority)
    y = ((1 - alpha) * config.lateral_tolerance_path_priority
         + alpha * config.lateral_tolerance_speed_priority)
    return v, y


def robust_cost(z, *, xp=np):
    z = xp.abs(xp.asarray(z))
    # Square only the bounded branch, to avoid overflow in the inactive branch.
    small = xp.minimum(z, 1.0)
    return small * small + 2.0 * xp.maximum(z - 1.0, 0.0)


def advance_tracking(state, lateral, heading, speed_error, roll, roll_rate,
                     action, failed, config, dt, *, xp=np):
    """Track observed path departure and a full-task hold without an event oracle.

    Path timers reset only after a complete geometric/attitude hold. True speed
    is used solely by the task-success diagnostic, never by Actor context.
    A deadline is measured from observed departure, not hidden disturbance end.
    It records a violation, but is NOT a physical termination condition.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    needed = max(1, int(math.ceil(config.hold_seconds / dt - 1e-9)))
    deadline = max(needed, int(math.ceil(config.return_deadline_seconds / dt - 1e-9)))
    failed = xp.asarray(failed)
    path_ok = ((xp.abs(lateral) <= config.return_lateral_tolerance)
               & (xp.abs(heading) <= config.return_heading_tolerance))
    attitude_ok = ((xp.abs(roll) <= config.roll_working_limit)
                   & (xp.abs(roll_rate) <= config.return_roll_rate_tolerance))
    departed = ~path_ok & ~failed
    ever = state.ever_departed | departed
    pending = state.pending | departed
    path_hold = xp.where(path_ok & attitude_ok & ~failed,
                         xp.minimum(state.path_hold_ticks + 1, needed), 0)
    age = xp.where(pending, state.age_ticks + 1, 0)
    missed = state.deadline_missed | (pending & (age > deadline))
    pending = pending & (path_hold < needed) & ~failed
    age = xp.where(pending, age, 0)
    joint_ok = path_ok & attitude_ok & (xp.abs(speed_error) <= config.return_speed_tolerance) & ~failed
    joint_hold = xp.where(joint_ok, xp.minimum(state.joint_hold_ticks + 1, needed), 0)
    recovered = (state.task_recovered | (ever & (joint_hold >= needed))) & ~failed
    return TrackingState(xp.asarray(action), pending, age, path_hold, ever,
                         joint_hold, recovered, missed)


def reward_components(roll, roll_rate, speed_error, lateral, heading, action,
                      previous_action, alpha, tracking, config, dt,
                      alive_rate=1.0, failure_penalty=100.0, failed=False, *, xp=np):
    """Signed per-transition components, evaluated against the ORIGINAL path.

    alpha=0: path priority; alpha=1: speed priority. Roll, return and authority
    never depend on alpha. Steering-angle, foot, arm and energy terms excluded.
    """
    c = config
    wv, wp = priority_weights(alpha, c, xp=xp)
    bv, by = tolerances(alpha, c, xp=xp)
    gv = 0.5 * (xp.exp(-(speed_error / c.speed_wide) ** 2)
                + xp.exp(-(speed_error / c.speed_fine) ** 2))
    gp = 0.5 * (xp.exp(-(lateral / c.lateral_wide) ** 2 - (heading / c.heading_wide) ** 2)
                + xp.exp(-(lateral / c.lateral_fine) ** 2 - (heading / c.heading_fine) ** 2))
    rates = dict(
        alive=xp.asarray(alive_rate),
        speed_tracking=c.tracking_rate * wv * gv,
        path_tracking=c.tracking_rate * wp * gp,
        speed_tail=-c.tail_weight * wv * robust_cost(speed_error / c.speed_tail_scale, xp=xp),
        path_tail=-c.tail_weight * wp * (robust_cost(lateral / c.lateral_tail_scale, xp=xp)
            + c.heading_tail_weight * robust_cost(heading / c.heading_tail_scale, xp=xp)),
        speed_budget=-c.budget_weight * robust_cost(xp.maximum(xp.abs(speed_error) - bv, 0.) / c.speed_excess_scale, xp=xp),
        path_budget=-c.budget_weight * robust_cost(xp.maximum(xp.abs(lateral) - by, 0.) / c.lateral_excess_scale, xp=xp),
        roll=-c.roll_excess_weight * xp.maximum(xp.abs(roll) - c.roll_working_limit, 0.) ** 2,
        roll_rate=-c.roll_rate_weight * roll_rate ** 2,
        action=-c.action_weight * xp.sum(xp.asarray(action) ** 2, axis=-1),
        action_change=-c.action_change_weight * xp.sum((xp.asarray(action) - xp.asarray(previous_action)) ** 2, axis=-1),
        return_time=-c.return_time_weight * tracking.pending * xp.minimum(tracking.age_ticks * dt / c.return_deadline_seconds, 1.),
        return_overdue=-c.return_overdue_weight * tracking.pending * (tracking.age_ticks * dt > c.return_deadline_seconds),
    )
    # Terminal replacement, never negative-reward clipping or dt-scaled failure.
    parts = {name: xp.where(failed, 0.0, dt * rate) for name, rate in rates.items()}
    parts['failure'] = xp.where(failed, -failure_penalty, 0.0)
    return parts
