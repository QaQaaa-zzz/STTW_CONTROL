"""Exogenous time reference from slew-limited forward-speed/yaw-rate commands.

The reference never uses the actual vehicle pose, progress or disturbances.
At tick k its command drives reference integration to k+1; only then is the
command updated for tick k+1. Headings remain unwrapped in the reference state.
"""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jp
import numpy as np


@dataclass(frozen=True)
class TimedReferenceConfig:
    speed_min: float = 1.7
    speed_max: float = 2.5
    yaw_rate_max: float = .6
    speed_slew: float = .5
    yaw_slew: float = .6
    switch_windows: tuple = ((1.5, 2.5), (3., 4.), (5., 6.))
    fixed: tuple | None = None
    yaw_feedback: float = 1.
    lateral_feedback: float = .4
    max_steer: float = .35

    def __post_init__(self):
        positive = (self.speed_min, self.speed_max, self.speed_slew,
                    self.yaw_slew, self.max_steer)
        nonnegative = (self.yaw_rate_max, self.yaw_feedback, self.lateral_feedback)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError('speeds, slew rates and steering bound must be positive finite')
        if any(not math.isfinite(x) or x < 0 for x in nonnegative):
            raise ValueError('yaw-rate bound and feedback gains must be nonnegative finite')
        if self.speed_max < self.speed_min:
            raise ValueError('speed_max must be at least speed_min')
        windows = tuple(tuple(row) for row in self.switch_windows)
        previous_end = 0.
        for row in windows:
            if (len(row) != 2 or not all(math.isfinite(x) for x in row)
                    or row[0] <= previous_end or row[1] < row[0]):
                raise ValueError('switch windows must be finite, positive and ordered without overlap')
            previous_end = row[1]
        object.__setattr__(self, 'switch_windows', windows)
        if self.fixed is not None:
            fixed = tuple(tuple(row) for row in self.fixed)
            if not fixed or len(fixed[0]) != 3 or fixed[0][0] != 0:
                raise ValueError('fixed schedule must start at time zero')
            previous_time = -1.
            for row in fixed:
                if (len(row) != 3 or not all(math.isfinite(x) for x in row)
                        or row[0] <= previous_time or row[1] <= 0):
                    raise ValueError('fixed rows require increasing time, positive speed and finite yaw rate')
                previous_time = row[0]
            object.__setattr__(self, 'fixed', fixed)


def schedule(key, config, initial_speed):
    """Sample continuous reset targets; fixed evaluation rows are returned verbatim.

    The default random schedule has four (time, speed, yaw-rate) rows. Its
    initial row is (0, initial_speed, 0), followed by one draw per window.
    This function supports JIT/vmap over keys and initial speeds.
    """
    if config.fixed is not None:
        return jp.asarray(config.fixed, dtype=float)
    n = len(config.switch_windows)
    draws = jax.random.uniform(key, (n, 3))
    windows = jp.asarray(config.switch_windows, dtype=float).reshape(n, 2)
    times = windows[:, 0] + draws[:, 0] * (windows[:, 1] - windows[:, 0])
    speeds = config.speed_min + draws[:, 1] * (config.speed_max - config.speed_min)
    yaws = (2 * draws[:, 2] - 1) * config.yaw_rate_max
    initial = jp.asarray([0., initial_speed, 0.])[None, :]
    return jp.concatenate((initial, jp.stack((times, speeds, yaws), axis=1)), axis=0)


def command_at(tick, previous_command, schedule, dt, config, *, xp=jp):
    """Update toward the active time target with per-second slew bounds."""
    rows = xp.asarray(schedule)
    previous_command = xp.asarray(previous_command)
    time = tick * dt
    # Float32 multiplication may put an exact switch tick just below its
    # declared time. Use a shared sub-tick tolerance on CPU and JAX backends.
    tolerance = xp.minimum(dt * .001, 4 * np.finfo(np.float32).eps * xp.maximum(1., xp.abs(time)))
    index = xp.maximum(xp.sum(rows[:, 0] <= time + tolerance) - 1, 0)
    target = rows[index, 1:3]
    limit = xp.asarray([config.speed_slew, config.yaw_slew]) * dt
    return previous_command + xp.clip(target - previous_command, -limit, limit)


def advance_reference(pose, command, dt, *, xp=jp):
    """Exact constant-twist integration, including the zero-yaw-rate limit."""
    pose, command = xp.asarray(pose), xp.asarray(command)
    angle = command[1] * dt
    midpoint_heading = pose[2] + angle / 2
    distance = command[0] * dt * xp.sinc(angle / (2 * xp.pi))
    return pose + xp.stack((distance * xp.cos(midpoint_heading),
                            distance * xp.sin(midpoint_heading), angle))


def errors(actual_pose, reference_pose, command, *, xp=jp):
    """Return right [m], wrapped heading [rad], curvature [1/m], along [m].

    Both position errors use actual minus reference; along < 0 means lagging.
    Reference curvature is yaw-rate / positive commanded forward speed.
    """
    actual_pose, reference_pose = xp.asarray(actual_pose), xp.asarray(reference_pose)
    command = xp.asarray(command)
    delta = actual_pose - reference_pose
    sine, cosine = xp.sin(reference_pose[2]), xp.cos(reference_pose[2])
    return xp.stack((sine * delta[0] - cosine * delta[1],
                     xp.arctan2(xp.sin(delta[2]), xp.cos(delta[2])),
                     command[1] / command[0], cosine * delta[0] + sine * delta[1]))


def reference_trace(config_dict, dt, horizon, initial_speed,
                    initial_pose=(0., 0., 0.), seed=0):
    """Reconstruct N+1 reference poses/commands, including the reset sample.

    Accept either a TimedReferenceConfig, its field dictionary, or a task
    dictionary containing ``timed_reference``. Horizon must span whole steps.
    Random schedules use the rollout's dedicated fold_in(seed_key, 51) stream.
    """
    if isinstance(config_dict, TimedReferenceConfig):
        config = config_dict
    else:
        config = TimedReferenceConfig(**config_dict.get('timed_reference', config_dict))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be positive finite')
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError('horizon must be nonnegative finite')
    if not math.isfinite(initial_speed) or initial_speed <= 0:
        raise ValueError('initial_speed must be positive finite')
    pose = np.asarray(initial_pose, dtype=float)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError('initial_pose must contain three finite values')
    steps = round(horizon / dt)
    if not math.isclose(steps * dt, horizon, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('horizon must be an integer multiple of dt')
    if config.fixed is None:
        key = jax.random.fold_in(jax.random.PRNGKey(seed), 51)
        rows = np.asarray(schedule(key, config, initial_speed), dtype=float)
    else:
        rows = np.asarray(config.fixed, dtype=float)
    poses = np.empty((steps + 1, 3), dtype=float)
    commands = np.empty((steps + 1, 2), dtype=float)
    poses[0] = pose
    commands[0] = (initial_speed, 0.)
    for tick in range(steps):
        poses[tick + 1] = advance_reference(poses[tick], commands[tick], dt, xp=np)
        commands[tick + 1] = command_at(tick + 1, commands[tick], rows, dt, config, xp=np)
    return {'reference_pose': poses, 'reference_command': commands}
