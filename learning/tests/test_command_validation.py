"""CPU protocol checks for the real batched command validator."""
from types import SimpleNamespace

from flax import struct
import jax
import jax.numpy as jp
import numpy as np
import pytest

from sttw_control.command_env import make_command_validator
from sttw_control.motion_commands import reference_at, MotionCommands


@struct.dataclass
class Data:
    qvel: object
    xmat: object


@struct.dataclass
class State:
    obs: object
    command_schedule: object
    tick: object
    done: object
    terminated: object
    measurement: object
    data: object
    yaw_rate: object
    reward: object


class TrajectoryEnv:
    """Known responses isolate validation semantics from GPU-only MJX stepping."""
    def __init__(self, speed_errors=(0., 0., 0., 0.),
                 yaw_errors=(0., 0., 0., 0.), failure_tick=99):
        self.config = SimpleNamespace(
            controller=SimpleNamespace(dt=1.), speed_reference=2.,
            priority=SimpleNamespace(validation_alphas=(0., .5, 1.)),
            motion_commands=MotionCommands(roll_working_limit=.3))
        self.horizon = 4
        self.bundle = SimpleNamespace(chassis=0)
        self.speed_errors = jp.asarray(speed_errors)
        self.yaw_errors = jp.asarray(yaw_errors)
        self.failure_tick = failure_tick

    def reset(self, key, *, command_schedule=None):
        schedule = (jp.asarray(((0., 2., 0., .7), (2., 2., 0., .9)))
                    if command_schedule is None else command_schedule)
        return State(jp.zeros(1), schedule, jp.asarray(0), jp.asarray(False),
                     jp.asarray(False), jp.zeros(2),
                     Data(jp.zeros(3), jp.eye(3)[None]), jp.asarray(0.),
                     jp.asarray(0.))

    def set_priority(self, state, alpha):
        return state.replace(obs=jp.asarray([alpha]))

    def requested(self, tick, schedule):
        return reference_at(tick, schedule)

    def step(self, state, action):
        def advance(s):
            raw = self.requested(s.tick, s.command_schedule)
            speed = raw[0] + self.speed_errors[s.tick]
            yaw = raw[1] + self.yaw_errors[s.tick]
            tick = s.tick + 1
            failed = tick >= self.failure_tick
            return s.replace(tick=tick, done=failed | (tick >= self.horizon),
                             terminated=failed, yaw_rate=yaw,
                             reward=raw[0] + 10 * raw[1] + action[0],
                             data=Data(jp.asarray([speed, 0., 0.]), jp.eye(3)[None]))
        return jax.lax.cond(state.done, lambda s: s, advance, state)


class AlphaActor:
    def apply(self, params, obs):
        return obs


def validation_config(schedules=None, **kw):
    values = dict(validation_seeds=(11, 12), command_validation_schedules=schedules,
                  validation_hold_seconds=2., validation_speed_tolerance=.2,
                  validation_yaw_tolerance=.2)
    return SimpleNamespace(**(values | kw))


def test_fixed_development_panel_covers_cases_alphas_seeds_without_training_change():
    env = TrajectoryEnv()
    schedules = (((0., 2., 0., .9), (2., 3., .1, .1)),
                 ((0., 2., .2, .1), (2., 1., -.1, .9)))
    before = env.reset(jax.random.PRNGKey(7)).command_schedule
    validate = make_command_validator(env, AlphaActor(), jp.ones(1),
                                      validation_config(schedules))
    result = validate({'actor': {}})
    # Each case spans all fixed alphas and both paired seeds; scheduled alphas
    # never replace the evaluation alpha, including after the command switch.
    np.testing.assert_allclose(result['episode_return'],
                               [12., 12., 14., 14., 16., 16.,
                                8., 8., 10., 10., 12., 12.])
    np.testing.assert_array_equal(result['terminal_tracking_hold'], np.ones(12, bool))
    np.testing.assert_allclose(validate({'actor': {}}, zero=True)['episode_return'],
                               [12.] * 6 + [8.] * 6)
    np.testing.assert_array_equal(env.reset(jax.random.PRNGKey(7)).command_schedule, before)


@pytest.mark.parametrize('speed,yaw,failed,want', [
    ((.9, .9, .15, .15), (0., 0., 0., 0.), 99, True),
    ((0., 0., 0., .25), (0., 0., 0., 0.), 99, False),
    ((0., 0., 0., 0.), (0., 0., .25, 0.), 99, False),
    ((0., 0., 0., 0.), (0., 0., 0., 0.), 3, False),
])
def test_terminal_hold_checks_every_requested_final_step_not_only_rmse(speed, yaw, failed, want):
    env = TrajectoryEnv(speed, yaw, failed)
    validate = make_command_validator(env, AlphaActor(), jp.ones(1), validation_config())
    result = validate({'actor': {}})
    assert np.all(np.asarray(result['terminal_tracking_hold']) == want)
    if failed == 99:
        np.testing.assert_allclose(result['final_speed_rmse'],
                                   np.sqrt(np.mean(np.square(speed[-2:]))), atol=1e-7)
        np.testing.assert_allclose(result['final_yaw_rmse'],
                                   np.sqrt(np.mean(np.square(yaw[-2:]))), atol=1e-7)
    # Whole-episode statistics remain available, with the original active mask.
    np.testing.assert_array_equal(result['steps'], [min(failed, 4)] * 6)


def test_terminal_hold_requires_the_entire_declared_window():
    result = make_command_validator(TrajectoryEnv(), AlphaActor(), jp.ones(1),
                                    validation_config(validation_hold_seconds=5.))({'actor': {}})
    assert not np.any(result['terminal_tracking_hold'])


def test_failure_before_terminal_window_marks_unobserved_error_instead_of_zero():
    result = make_command_validator(TrajectoryEnv(failure_tick=1), AlphaActor(),
                                    jp.ones(1), validation_config())({'actor': {}})
    np.testing.assert_array_equal(result['final_window_steps'], np.zeros(6, int))
    assert not np.any(result['terminal_tracking_hold'])
    assert np.all(np.isnan(result['final_speed_rmse']))
    assert np.all(np.isnan(result['final_yaw_rmse']))


def test_default_development_schedule_preserves_old_summary_and_alpha_order():
    result = make_command_validator(TrajectoryEnv(), AlphaActor(), jp.ones(1),
                                    validation_config())({'actor': {}})
    np.testing.assert_allclose(result['episode_return'], [8., 8., 10., 10., 12., 12.])
    for key in ('initial_speed_rmse', 'initial_yaw_rmse', 'speed_rmse', 'yaw_rmse'):
        np.testing.assert_array_equal(result[key], np.zeros(6))


def test_fixed_panel_rejects_initial_speed_that_disagrees_with_reset_physics():
    schedules = (((0., 2.1, 0., .5), (2., 2., 0., .5)),)
    with pytest.raises(ValueError, match='initial speed'):
        make_command_validator(TrajectoryEnv(), AlphaActor(), jp.ones(1),
                               validation_config(schedules))


def test_explicit_reset_schedule_initializes_command_observation_and_keeps_default_reset():
    from sttw_control.env import RecoveryEnv, TaskConfig
    from sttw_control.motion_commands import MotionCommands
    from sttw_control.observation import ObservationConfig, observation_fields
    from sttw_control.priority import PriorityConfig
    env = RecoveryEnv(TaskConfig(
        motion_commands=MotionCommands(), priority=PriorityConfig(risk_gate=False),
        observation=ObservationConfig(include_motion=True, include_priority=True,
                                      include_attitude_risk=False), horizon_seconds=.025))
    original = env.reset(3)
    schedule = jp.asarray(((0., 2., .3, .7), (.01, 2.2, -.4, .2)))
    overridden = env.reset(3, command_schedule=schedule)
    np.testing.assert_array_equal(overridden.command_schedule, schedule)
    assert not np.allclose(overridden.base, original.base)
    fields = observation_fields(env.config.observation)
    assert np.isclose(overridden.history.frames[-1, fields.index('yaw_rate_reference')], .3)
    np.testing.assert_array_equal(env.reset(3).command_schedule, original.command_schedule)
