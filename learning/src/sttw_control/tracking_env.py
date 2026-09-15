"""Optional geometric reward path; ECBC/ESO, actuator and physics stay shared."""
import math
import jax.numpy as jp
from .observation import TRACKING_FIELDS, observation_fields
from .tracking_reward import (initial_tracking, advance_tracking, tracking_context,
                              reward_components)


def reset_tracking_state(env, state):
    c = env.config
    tracking = initial_tracking(xp=jp)
    parts = reward_components(0., 0., 0., 0., 0., jp.zeros(2), jp.zeros(2),
                              state.priority_alpha, tracking, c.tracking_reward,
                              c.controller.dt, c.alive_reward_rate,
                              c.failure_penalty, False, xp=jp)
    return state.replace(tracking=tracking,
                         reward_parts={k:jp.zeros_like(v) for k,v in parts.items()})


def finish_tracking_step(env, state, measurement, actuator, action, true_speed,
                         pose, tick, prepared, invalid, physical_contact, fallen):
    c, rc = env.config, env.config.tracking_reward
    controller, history, obs, base, reference = prepared
    failed = invalid | physical_contact | fallen
    path = env.path_features(pose)
    speed_error = true_speed - env.command(state.tick, state.pose)[1]
    tracking = advance_tracking(state.tracking, path[0], path[1], speed_error,
                                measurement[0], measurement[1], action, failed,
                                rc, c.controller.dt, xp=jp)
    parts = reward_components(measurement[0], measurement[1], speed_error,
                              path[0], path[1], action, state.tracking.previous_action,
                              state.priority_alpha, tracking, rc, c.controller.dt,
                              c.alive_reward_rate, c.failure_penalty, failed, xp=jp)
    indices = jp.asarray([observation_fields(c.observation).index(k) for k in TRACKING_FIELDS])
    frame = history.frames[-1].at[indices].set(tracking_context(tracking, rc, c.controller.dt, xp=jp))
    history = history.replace(frames=history.frames.at[-1].set(frame))
    obs = jp.concatenate((history.frames.flatten(), history.mask))
    timeout = (tick >= env.horizon) & ~failed
    code = jp.where(invalid, 3, jp.where(physical_contact, 4, jp.where(fallen, 1, jp.where(timeout, 2, 0))))
    needed = max(1, int(math.ceil(rc.hold_seconds/c.controller.dt - 1e-9)))
    balanced = (jp.abs(measurement[0]) <= rc.roll_working_limit) & (jp.abs(measurement[1]) <= rc.return_roll_rate_tolerance) & ~failed
    balance_ticks = jp.where(balanced, jp.minimum(state.recovery.balance_ticks+1, needed), 0)
    recovery = state.recovery.replace(balance_ticks=balance_ticks,
        task_ticks=tracking.joint_hold_ticks,
        balance_recovered=(state.recovery.balance_recovered | (tracking.ever_departed & (balance_ticks>=needed))) & ~failed,
        task_recovered=tracking.task_recovered)
    return state.replace(controller=controller, actuator=actuator, history=history,
        recovery=recovery, measurement=measurement, pose=pose, reference=reference,
        base=base, obs=jp.nan_to_num(obs, nan=0., posinf=0., neginf=0.), tick=tick,
        reward=sum(parts.values()), reward_parts=parts, tracking=tracking,
        done=failed|timeout, terminated=failed, truncated=timeout, end_code=code)
