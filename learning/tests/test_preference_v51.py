"""Focused contract checks for the V5.1 reward and reset overlay."""
import numpy as np
import jax
import jax.numpy as jp

from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_reward import failure_reward, interval_cost
from sttw_control.direct_command_scenarios import heading_recovery_initial_error, schedule
from sttw_control.smooth_command_config import resolve


def cost(alpha, *, ev=0., ed=0., debt=0., yaw_rate=0., chi=1., g=0.):
    spec = resolve(alpha, preference_v51=True)
    raw = jp.array([2.6, .0])
    return interval_cost(
        alpha=alpha, chi=chi, g=g, raw=raw,
        actual_speed=raw[0] + ev, actual_steer=raw[1] + ed,
        heading_error=debt, roll=0., roll_rate=0., executed_offsets=jp.zeros(2),
        final_command=jp.zeros(2), previous_final_command=jp.zeros(2), spec=spec,
        yaw_rate=yaw_rate, cc=ControllerConfig())


def test_primary_excess_selects_only_the_fixed_primary_channel():
    np.testing.assert_allclose(cost(0, ev=-.4, ed=.08)['raw_components']['primary_excess'], 40., rtol=2e-6)
    np.testing.assert_allclose(cost(1, ev=-.15, ed=-.10)['raw_components']['primary_excess'], 40., rtol=2e-6)
    assert float(cost(0, ev=-.4, ed=.02)['raw_components']['primary_excess']) == 0.
    assert float(cost(1, ev=-.03, ed=-.10)['raw_components']['primary_excess']) == 0.
    assert float(cost(0, ed=.08, g=1.)['raw_components']['primary_excess']) == 0.


def test_yaw_recovery_sign_and_reference_values():
    assert float(cost(0, debt=.8, yaw_rate=.4, chi=0., g=1.)['raw_components']['yaw_recovery']) == 0.
    assert float(cost(0, debt=.8, yaw_rate=0., chi=0., g=1.)['raw_components']['yaw_recovery']) == 6.
    assert float(cost(0, debt=.8, yaw_rate=-.4, chi=0., g=1.)['raw_components']['yaw_recovery']) == 14.
    assert float(cost(1, debt=-.8, yaw_rate=-.4, chi=0., g=1.)['raw_components']['yaw_recovery']) == 0.


def test_caps_and_failure_bound_are_2080_without_global_clip():
    spec = resolve(1, preference_v51=True)
    assert sum(spec['reward']['independent_component_caps'].values()) == 2080.
    assert spec['reward']['failure_absorbing_cost_rate'] == 2080.
    assert spec['reward']['global_cost_cap'] is None
    gamma = spec['ppo']['gamma']; n = 800
    expected = -5. - .1 * .02 * 2080 * (1-gamma**n)/(1-gamma)
    np.testing.assert_allclose(failure_reward(n, spec), expected, rtol=3e-6)


def test_v51_runtime_is_fresh_16_second_100_update_contract():
    spec = resolve(0, preference_v51=True)
    base = resolve(0, preference_v5=True, stage=2)
    assert spec['priority_recovery_v51'] and spec['initialization_mode'] == 'scratch'
    assert spec['episode_duration_s'] == 16.
    assert spec['ppo']['seed'] == 87
    assert spec['ppo']['default_updates'] == 100
    assert spec['ppo']['validation_updates'] == [25, 100]
    assert spec['ppo']['initial_latent_std'] == [.30, .10]
    assert spec['network'] == base['network']
    assert spec['action'] == base['action']
    assert spec['plant'] == base['plant'] and spec['limits'] == base['limits']
    assert spec['lower_controller'] == base['lower_controller'] and spec['lower_reference_centered']
    assert [spec['commands'][k] for k in ('nominal_fraction','conflict_fraction','random_fraction','heading_recovery_start_fraction')] == [.3,.4,.1,.2]


def test_heading_family_is_deterministic_signed_and_constant_command():
    spec = resolve(0, preference_v51=True)
    ids = jp.arange(200, dtype=jp.int32)
    rows_all, families, _ = jax.jit(jax.vmap(lambda env_id: schedule(env_id, 0, 2.37, spec)))(ids)
    errors = jax.jit(jax.vmap(lambda env_id, family: heading_recovery_initial_error(env_id, 0, spec, family)))(ids, families)
    found = []
    for rows, family, error in zip(np.asarray(rows_all), np.asarray(families), np.asarray(errors)):
        if int(family) == 3:
            e0 = float(error)
            active = rows[rows[:,0] < 99]
            assert active.shape == (1,3)
            np.testing.assert_allclose(active[0], [0.,2.37,0.])
            assert .03 <= abs(e0) <= 1.
            found.append(e0)
    assert any(x < 0 for x in found) and any(x > 0 for x in found)


def test_existing_v5_component_identity_is_unchanged():
    old = resolve(0, preference_v5=True, stage=2)
    assert 'yaw_damping' in old['reward']['independent_component_caps']
    assert 'primary_excess' not in old['reward']['independent_component_caps']
    assert old['reward']['failure_absorbing_cost_rate'] == 1680.
