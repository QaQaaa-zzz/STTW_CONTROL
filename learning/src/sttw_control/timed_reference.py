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
    recovery_probability: float = 0.
    recovery_start: float = 4.
    conflict_start_window: tuple = (.5, 1.)
    training_mix: bool = False
    priority_v2_screen: bool = False
    fixed_scenario: str | None = None
    fast_speed_slew: float = 1.
    fast_yaw_slew: float = 2.4
    gentle_yaw_rate: float = .35

    mode: str = 'time'  # 'geometry': fixed original curve, not time-position tracking
    geometry_stride: int = 10  # downsample exact control-tick integration at reset
    projection_margin: float = .05
    extension_seconds: float = 5.

    def __post_init__(self):
        if type(self.priority_v2_screen) is not bool or (self.priority_v2_screen and not self.training_mix):
            raise ValueError("priority_v2_screen requires training_mix")
        if type(self.training_mix) is not bool:
            raise ValueError('training_mix must be boolean')
        if self.fixed_scenario not in (None, 'nominal', 'ordinary_accel', 'core_left', 'core_right', 'disturbance_left', 'disturbance_right'):
            raise ValueError('invalid fixed training scenario')
        if self.fixed_scenario is not None and not self.training_mix:
            raise ValueError('fixed_scenario requires training_mix')
        if not math.isfinite(self.recovery_probability) or not 0 <= self.recovery_probability <= 1:
            raise ValueError('recovery_probability must be in [0,1]')
        window = tuple(self.conflict_start_window)
        if len(window) != 2 or not all(math.isfinite(x) for x in window) or not 0 < window[0] <= window[1]:
            raise ValueError('invalid conflict_start_window')
        if not math.isfinite(self.recovery_start) or self.recovery_start <= window[1]:
            raise ValueError('recovery_start must follow conflict start window')
        object.__setattr__(self, 'conflict_start_window', window)
        if self.recovery_probability and not self.switch_windows:
            raise ValueError('recovery mixture requires random target windows')
        if self.mode not in ('time', 'geometry'):
            raise ValueError('reference mode must be time or geometry')
        if type(self.geometry_stride) is not int or self.geometry_stride < 1:
            raise ValueError('geometry_stride must be a positive integer')
        if any(not math.isfinite(x) or x <= 0 for x in (self.projection_margin, self.extension_seconds)):
            raise ValueError('projection margin and extension must be positive finite')
        positive = (self.speed_min, self.speed_max, self.speed_slew,
                    self.yaw_slew, self.fast_speed_slew, self.fast_yaw_slew,
                    self.gentle_yaw_rate, self.max_steer)
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
    if config.training_mix:
        # Columns: time, raw speed/yaw request, speed/yaw slew, recovery-entry,
        # scenario (0 nominal, 1 finite conflict, 2 lateral disturbance).
        draw = jax.random.uniform(key, (6,))
        sign = jp.where(draw[1] < .5, -1., 1.)
        nominal_kind = jp.floor(draw[2] * 3.).astype(jp.int32)
        nominal_yaw = jp.where(nominal_kind == 0, 0., sign * config.gentle_yaw_rate)
        nominal_speed = config.speed_min + draw[3] * (config.speed_max - config.speed_min)
        def row(time, speed, yaw, speed_slew, yaw_slew, recovery, scenario):
            return jp.asarray([time, speed, yaw, speed_slew, yaw_slew, recovery, scenario])
        nominal = jp.stack((
            row(0., initial_speed, 0., config.speed_slew, config.yaw_slew, 0., 0.),
            row(1., nominal_speed, nominal_yaw, config.speed_slew, config.yaw_slew, 0., 0.),
            row(4., nominal_speed, 0., config.speed_slew, config.yaw_slew, 0., 0.),
            row(9., nominal_speed, 0., config.speed_slew, config.yaw_slew, 0., 0.)))
        conflict = jp.stack((
            row(0., initial_speed, 0., config.fast_speed_slew, config.fast_yaw_slew, 0., 1.),
            row(1., 2.5, sign * 1.8, config.fast_speed_slew, config.fast_yaw_slew, 0., 1.),
            row(1.95, 2.5, 0., config.fast_speed_slew, config.fast_yaw_slew, 1., 1.),
            row(9., 2.5, 0., config.fast_speed_slew, config.fast_yaw_slew, 0., 1.)))
        disturbance = jp.stack((
            row(0., initial_speed, 0., config.speed_slew, config.yaw_slew, 0., 2.),
            row(1., initial_speed, sign * config.gentle_yaw_rate, config.speed_slew, config.yaw_slew, 0., 2.),
            row(6., initial_speed, 0., config.speed_slew, config.yaw_slew, 0., 2.),
            row(9., initial_speed, 0., config.speed_slew, config.yaw_slew, 0., 2.)))
        if config.fixed_scenario is not None:
            if config.fixed_scenario == 'ordinary_accel':
                return nominal.at[1,1:3].set(jp.array([2.5,0.])).at[2:,1:3].set(jp.array([2.5,0.]))
            code = 0 if config.fixed_scenario == 'nominal' else (1 if config.fixed_scenario.startswith('core') else 2)
            forced_sign = -1. if config.fixed_scenario.endswith('right') else 1.
            chosen = (nominal, conflict, disturbance)[code]
            if code:
                chosen = chosen.at[1, 2].set(abs(chosen[1, 2]) * forced_sign)
            return chosen
        if config.priority_v2_screen:
            ordinary=nominal.at[1:,1:3].set(jp.array([2.5,0.]))
            conflict=conflict.at[1,2].set(jp.where(draw[0]<.65,1.8,-1.8))
            return jp.where(draw[0]<.30,ordinary,conflict)
        scenario = jp.where(draw[0] < .4, 0, jp.where(draw[0] < .8, 1, 2))
        return jp.where(scenario == 0, nominal, jp.where(scenario == 1, conflict, disturbance))
    n = len(config.switch_windows)
    draws = jax.random.uniform(key, (n, 3))
    windows = jp.asarray(config.switch_windows, dtype=float).reshape(n, 2)
    times = windows[:, 0] + draws[:, 0] * (windows[:, 1] - windows[:, 0])
    speeds = config.speed_min + draws[:, 1] * (config.speed_max - config.speed_min)
    yaws = (2 * draws[:, 2] - 1) * config.yaw_rate_max
    initial = jp.asarray([0., initial_speed, 0.])[None, :]
    ordinary = jp.concatenate((initial, jp.stack((times, speeds, yaws), axis=1)), axis=0)
    if config.recovery_probability == 0:
        return ordinary
    # Separate random stream leaves the original stress schedules byte-identical.
    recover = jax.random.uniform(jax.random.fold_in(key, 73)) < config.recovery_probability
    start = config.conflict_start_window[0] + draws[0, 0] * (config.conflict_start_window[1] - config.conflict_start_window[0])
    # Repeated identical targets preserve a static shape without further switches.
    conflict_times = start + jp.arange(n) * (config.recovery_start - start) / n
    conflict = jp.stack((conflict_times, jp.full((n,), speeds[0]), jp.full((n,), yaws[0])), axis=1)
    recovery_rows = jp.concatenate((initial, conflict, jp.array([[config.recovery_start, speeds[0], 0.]])), axis=0)
    end_time = max(config.recovery_start, config.switch_windows[-1][1]) + .001
    stress_rows = jp.concatenate((ordinary, jp.array([[end_time, speeds[-1], yaws[-1]]])), axis=0)
    return jp.where(recover, recovery_rows, stress_rows)


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
    limit = (rows[index, 3:5] if config.training_mix
             else xp.asarray([config.speed_slew, config.yaw_slew])) * dt
    return previous_command + xp.clip(target - previous_command, -limit, limit)


def raw_request_at(tick, schedule, dt, *, xp=jp):
    """Return the externally published unslewed speed/yaw request."""
    rows = xp.asarray(schedule)
    time = tick * dt
    tolerance = xp.minimum(dt * .001, 4 * np.finfo(np.float32).eps * xp.maximum(1., xp.abs(time)))
    index = xp.maximum(xp.sum(rows[:, 0] <= time + tolerance) - 1, 0)
    return rows[index, 1:3]


def recovery_entry_at(tick, schedule, dt, *, xp=jp):
    """True once when a published row enters the declared recovery segment."""
    rows = xp.asarray(schedule)
    if rows.shape[1] < 6:
        return xp.asarray(False)
    time = tick * dt
    tolerance = xp.minimum(dt * .001, 4 * np.finfo(np.float32).eps * xp.maximum(1., xp.abs(time)))
    index = xp.maximum(xp.sum(rows[:, 0] <= time + tolerance) - 1, 0)
    previous_time = xp.maximum(tick - 1, 0) * dt
    previous_index = xp.maximum(xp.sum(rows[:, 0] <= previous_time + tolerance) - 1, 0)
    return (rows[index, 5] > .5) & (index != previous_index)


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


def committed_geometry_table(initial_pose, initial_command, rows, dt, config, steps):
    """Frozen original reference, columns arc,x,y,heading,speed,yaw-rate.

    Future rows exist for efficient reset-time construction but projection only
    reads committed segments. The table is never translated to the vehicle.
    """
    def step(carry, tick):
        pose, command, arc = carry
        pose = advance_reference(pose, command, dt)
        arc = arc + command[0] * dt
        command = command_at(tick, command, rows, dt, config)
        return (pose, command, arc), jp.concatenate((jp.atleast_1d(arc), pose, command))
    initial = jp.concatenate((jp.zeros(1), initial_pose, initial_command))
    _, tail = jax.lax.scan(step, (initial_pose, initial_command, jp.asarray(0.)), jp.arange(1, steps + 1))
    return jp.concatenate((initial[None], tail))


def project_committed_geometry(pose, table, previous_segment, previous_progress, displacement, frontier, *, xp=jp):
    """Causal continuous local projection; return path features, arc, segment.

    A 129-segment search is intersected with the already committed prefix and
    a (twice displacement + 5 cm) arc window. This prevents jumping to another loop at
    self intersections. Ahead of the committed frontier, projection clamps to
    its endpoint; the endpoint tangent is held, with no future command preview.
    Piecewise linear XY geometry approximates each exact integrated interval.
    """
    pose, table = xp.asarray(pose), xp.asarray(table)
    frontier = xp.clip(frontier, 0, len(table)-1)
    indices = xp.clip(previous_segment + xp.arange(-64,65), 0, xp.maximum(frontier-1,0)).astype(int)
    end_indices = xp.minimum(indices+1, frontier).astype(int)
    a,b = table[indices],table[end_indices]
    vector=b[:,1:3]-a[:,1:3]
    length2=xp.sum(vector*vector,axis=1)
    fraction=xp.sum((pose[:2]-a[:,1:3])*vector,axis=1)/xp.maximum(length2,1e-12)
    span=b[:,0]-a[:,0]
    allowance=2*xp.maximum(displacement,0.)+.05
    lo=xp.maximum(a[:,0],previous_progress-allowance)
    hi=xp.minimum(b[:,0],previous_progress+allowance)
    fraction=xp.clip(fraction,xp.clip((lo-a[:,0])/xp.maximum(span,1e-12),0,1),xp.clip((hi-a[:,0])/xp.maximum(span,1e-12),0,1))
    point=a[:,1:3]+fraction[:,None]*vector
    distance=xp.sum((pose[:2]-point)**2,axis=1)
    distance=xp.where(lo<=hi+1e-7,distance,xp.inf)
    chosen=xp.argmin(distance)
    row=a[chosen];f=fraction[chosen];endpoint=b[chosen]
    tangent=row[3]+f*(endpoint[3]-row[3])
    delta=pose[:2]-point[chosen]
    heading=pose[2]-tangent
    features=xp.stack((xp.sin(tangent)*delta[0]-xp.cos(tangent)*delta[1],
                       xp.arctan2(xp.sin(heading),xp.cos(heading)),row[5]/row[4]))
    return features,row[0]+f*span[chosen],indices[chosen]


def geometry_table(rows, config, initial_speed, initial_pose, horizon, dt):
    """Fixed curve integrated at the original control tick, then downsampled.

    Table columns: arc length, world X/Y, unwrapped heading, curvature. The
    extra suffix prevents an artificial end-of-path during the episode. It is
    part of the declared geometry, not a recovery-time extension.
    """
    steps = math.ceil((horizon + config.extension_seconds) / dt)
    stride = config.geometry_stride
    steps = math.ceil(steps / stride) * stride
    pose = jp.asarray(initial_pose)
    command = jp.asarray([initial_speed, 0.])
    def integrate(carry, tick):
        pose, command, arc = carry
        nxt_pose = advance_reference(pose, command, dt)
        arc = arc + command[0] * dt
        nxt_cmd = command_at(tick + 1, command, rows, dt, config)
        row = jp.concatenate((jp.reshape(arc, (1,)), nxt_pose,
                              jp.reshape(nxt_cmd[1] / nxt_cmd[0], (1,))))
        return (nxt_pose, nxt_cmd, arc), row
    _, table = jax.lax.scan(integrate, (pose, command, jp.asarray(0.)), jp.arange(steps))
    first = jp.concatenate((jp.zeros(1), pose, jp.zeros(1)))
    return jp.concatenate((first[None], table[stride-1::stride]), axis=0)


def project_geometry(pose, table, previous_progress, window, *, xp=jp):
    """Continuity-windowed segment projection, shared with offline audit.

    Never reanchors or translates the original path. The search window depends
    on actual displacement, not speed-command time or alpha. At crossings a
    spatially close but arc-distant branch is excluded.
    """
    pose, table = xp.asarray(pose), xp.asarray(table)
    start, end = table[:-1, 1:3], table[1:, 1:3]
    delta = end - start
    ds = table[1:, 0] - table[:-1, 0]
    lo = xp.maximum(table[0, 0], previous_progress - window)
    hi = xp.minimum(table[-1, 0], previous_progress + window)
    valid = (table[1:, 0] >= lo) & (table[:-1, 0] <= hi)
    u = xp.sum((pose[:2] - start) * delta, axis=1) / xp.maximum(xp.sum(delta**2, axis=1), 1e-12)
    u = xp.clip(u, xp.clip((lo-table[:-1, 0])/ds, 0., 1.),
                xp.clip((hi-table[:-1, 0])/ds, 0., 1.))
    foot = start + u[:, None] * delta
    index = xp.argmin(xp.where(valid, xp.sum((foot-pose[:2])**2, axis=1), xp.inf))
    progress = table[index, 0] + u[index] * ds[index]
    yaw = table[index, 3] + u[index] * (table[index+1, 3]-table[index, 3])
    curvature = table[index, 4] + u[index] * (table[index+1, 4]-table[index, 4])
    return progress, xp.concatenate((foot[index], xp.reshape(yaw, (1,)))), curvature
