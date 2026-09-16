"""Time reference follows declared commands, independently of actual motion."""
from dataclasses import asdict

import jax
import jax.numpy as jp
import numpy as np
import pytest

from sttw_control.timed_reference import (
    TimedReferenceConfig, advance_reference, command_at, errors,
    reference_trace, schedule,
)


def test_exact_circle_and_zero_yaw_integration_preserve_unwrapped_heading():
    np.testing.assert_allclose(
        advance_reference(np.zeros(3), np.array([2., 1.]), np.pi / 2, xp=np),
        [2., 2., np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(
        advance_reference(np.array([1., 2., 4*np.pi]), np.array([2., 0.]), .5, xp=np),
        [2., 2., 4*np.pi], atol=1e-12)
    out = jax.jit(lambda p, c: advance_reference(p, c, .5))(jp.zeros(3), jp.array([2., 0.]))
    np.testing.assert_allclose(out, [1., 0., 0.], atol=1e-7)


def test_errors_report_right_and_along_actual_minus_reference():
    # Facing +Y: right is +X, and an actual point at y=1 trails y=3.
    got = errors(np.array([4., 1., np.pi/2+.2]),
                 np.array([1., 3., np.pi/2]), np.array([2., .6]), xp=np)
    np.testing.assert_allclose(got, [3., .2, .3, -2.], atol=1e-12)
    got = errors(np.array([0., 0., -np.pi+.1]),
                 np.array([0., 0., np.pi-.1]), np.array([2., 0.]), xp=np)
    assert got[1] == pytest.approx(.2)


def test_schedule_randomizes_continuous_targets_and_resets_initial_command():
    config = TimedReferenceConfig()
    a = np.asarray(schedule(jax.random.PRNGKey(5), config, 2.))
    b = np.asarray(schedule(jax.random.PRNGKey(6), config, 2.))
    assert a.shape == (4, 3)
    np.testing.assert_allclose(a[0], [0., 2., 0.])
    assert not np.array_equal(a, b)
    for row, (lo, hi) in zip(a[1:], config.switch_windows):
        assert lo <= row[0] <= hi
        assert config.speed_min <= row[1] <= config.speed_max
        assert abs(row[2]) <= config.yaw_rate_max
    compiled = jax.jit(lambda key: schedule(key, config, 2.))(jax.random.PRNGKey(5))
    np.testing.assert_allclose(compiled, a, atol=1e-6)


def test_commands_switch_by_time_and_obey_both_slew_limits():
    config = TimedReferenceConfig(fixed=((0., 2., 0.), (.5, 2.5, .6)))
    rows = schedule(jax.random.PRNGKey(0), config, 2.)
    np.testing.assert_allclose(command_at(4, np.array([2., 0.]), rows, .1, config, xp=np), [2., 0.])
    np.testing.assert_allclose(command_at(5, np.array([2., 0.]), rows, .1, config, xp=np), [2.05, .06])
    result = reference_trace(asdict(config), .1, 2., 2.)
    differences = np.diff(result['reference_command'], axis=0)
    assert np.max(np.abs(differences[:, 0])) <= config.speed_slew*.1+1e-12
    assert np.max(np.abs(differences[:, 1])) <= config.yaw_slew*.1+1e-12
    np.testing.assert_allclose(result['reference_command'][-1], [2.5, .6])


def test_fixed_trace_rebuilds_with_pre_step_command_and_ignores_actual_disturbance():
    config = TimedReferenceConfig(fixed=((0., 2., .6),))
    trace = reference_trace(asdict(config), .1, .2, 2., initial_pose=(1., 2., .3), seed=17)
    same = reference_trace(asdict(config), .1, .2, 2., initial_pose=(1., 2., .3), seed=93)
    assert trace['reference_pose'].shape == (3, 3)
    assert trace['reference_command'].shape == (3, 2)
    np.testing.assert_array_equal(trace['reference_pose'], same['reference_pose'])
    np.testing.assert_allclose(trace['reference_command'], [[2., 0.], [2., .06], [2., .12]])
    np.testing.assert_allclose(trace['reference_pose'][1], [1+.2*np.cos(.3), 2+.2*np.sin(.3), .3])
    pose = jp.array([1., 2., .3]); command = jp.array([2., 0.])
    rows = schedule(jax.random.PRNGKey(0), config, 2.)
    actual = np.array([100., -5., -1.])
    for tick in range(2):
        errors(actual, pose, command)
        actual += np.array([3., 6., .7])
        pose = advance_reference(pose, command, .1)
        command = command_at(tick+1, command, rows, .1, config)
        np.testing.assert_allclose(pose, trace['reference_pose'][tick+1], atol=2e-7)


def test_fixed_switch_on_tick_agrees_between_numpy_and_jit():
    config = TimedReferenceConfig(fixed=((0., 2., 0.), (.025, 2.5, .6)))
    rows = schedule(jax.random.PRNGKey(0), config, 2.)
    compiled = jax.jit(lambda tick: command_at(tick, jp.array([2., 0.]), rows, .005, config))
    np.testing.assert_allclose(compiled(4), [2., 0.], atol=1e-7)
    np.testing.assert_allclose(compiled(5), [2.0025, .003], atol=1e-7)
    trace = reference_trace(asdict(config), .005, .025, 2.)
    np.testing.assert_allclose(compiled(5), trace['reference_command'][-1], atol=1e-7)


def test_random_trace_uses_same_seed_stream_as_environment_reset():
    config = TimedReferenceConfig(switch_windows=((.01, .015),))
    rows = schedule(jax.random.fold_in(jax.random.PRNGKey(71), 51), config, 2.1)
    trace = reference_trace(asdict(config), .005, 1., 2.1, seed=71)
    pose, command = jp.zeros(3), jp.array([2.1, 0.])
    for tick in range(200):
        pose = advance_reference(pose, command, .005)
        command = command_at(tick+1, command, rows, .005, config)
    np.testing.assert_allclose(trace['reference_pose'][-1], pose, atol=2e-5)
    np.testing.assert_allclose(trace['reference_command'][-1], command, atol=2e-5)


@pytest.mark.parametrize('kwargs', [
    {'speed_min': 0.}, {'speed_max': 1.}, {'yaw_rate_max': float('inf')},
    {'speed_slew': 0.}, {'yaw_slew': float('nan')},
    {'switch_windows': ((2., 3.), (1., 2.))},
    {'switch_windows': ((1., 3.), (2., 4.))},
    {'switch_windows': ((1., float('inf')),)},
    {'fixed': ((1., 2., 0.),)}, {'fixed': ((0., 0., 0.),)},
    {'fixed': ((0., 2., 0.), (0., 2., .1))},
    {'fixed': ((0., 2., float('nan')),)},
    {'lateral_feedback': -1.}, {'max_steer': 0.},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TimedReferenceConfig(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'dt': 0.}, {'horizon': -.1}, {'horizon': .25},
    {'initial_speed': 0.}, {'initial_pose': (0., float('nan'), 0.)},
])
def test_trace_rejects_invalid_boundary_arguments(kwargs):
    args = dict(config_dict={}, dt=.1, horizon=.2, initial_speed=2.)
    args.update(kwargs)
    with pytest.raises(ValueError):
        reference_trace(**args)
