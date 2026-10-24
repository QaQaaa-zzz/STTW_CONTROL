"""Information and handback contract for a simple causal development baseline."""
import numpy as np
import jax
import jax.numpy as j
import pytest
from copy import deepcopy

from sttw_control.causal_reference_baseline import initial_gate, causal_reference_action


CONFIG = {'residual_interface': {'speed_negative_scale_m_s': 1.,
                               'speed_positive_scale_m_s': .25,
                               'steer_scale_rad': .2}}


def observation(signal=0., speed=1.5, curvature=-.9, alpha=0.):
    obs = j.zeros(351).at[334:336].set(1.)
    obs = obs.at[298].set(0.).at[318].set(-signal)
    return obs.at[336].set(alpha).at[340].set(curvature/.6).at[345].set(speed/3.)


def act(obs, tick, state=None, offset=None):
    return jax.jit(causal_reference_action)(
        j.array([-.3, .7]), obs, j.array([-.4, .02]) if offset is None else offset,
        j.int32(tick), initial_gate() if state is None else state, CONFIG)


def test_trigger_uses_current_offset_and_ignores_unselected_actor_channel():
    action, state, record = act(observation(signal=-.03), 300)
    assert record['detected'] and record['override_active']
    assert state.fired and int(state.start_tick) == 300
    np.testing.assert_allclose(state.target_offset, [-.25, .02], atol=1e-7)
    np.testing.assert_array_equal(state.mask, [True, False])
    assert float(action[1]) == np.float32(.7)


def test_detected_target_is_latched_and_handback_keeps_parent_actions():
    _, state, _ = act(observation(signal=-.03), 300)
    action, later, record = act(observation(signal=.03), 376, state, j.array([.1, -.1]))
    np.testing.assert_array_equal(later.target_offset, state.target_offset)
    assert not record['detected'] and record['override_active']
    for tick in (380, 384, 800):
        action, _, record = act(observation(signal=-.04), tick, state)
        np.testing.assert_array_equal(action, j.array([-.3, .7]))
        assert not record['override_active']


def test_opposite_direction_is_detected_but_never_retriggered_as_a_tail_event():
    action, state, record = act(observation(signal=.03), 20)
    assert record['detected'] and state.fired and not record['override_active']
    np.testing.assert_array_equal(action, j.array([-.3, .7]))
    action, _, record = act(observation(signal=-.05), 24, state)
    assert not record['detected'] and not record['override_active']


def test_nominal_invalid_history_and_nonmatching_preferences_do_not_override():
    for obs in (observation(.014), observation(-.03).at[334].set(0.),
                observation(-.03).at[318].set(5.), observation(-.03).at[298].set(j.nan)):
        action, state, record = act(obs, 4)
        np.testing.assert_array_equal(action, j.array([-.3, .7]))
        assert not state.fired and not record['override_active']
    for obs in (observation(-.03, alpha=1.), observation(-.03, speed=2.5),
                observation(-.03, curvature=0.)):
        action, state, record = act(obs, 4)
        assert state.fired and not record['override_active']
        np.testing.assert_array_equal(action, j.array([-.3, .7]))


def test_left_context_selects_steer_only_and_threshold_equality_triggers():
    action, state, record = act(observation(.015, speed=1.8, curvature=.9), 100)
    assert record['detected'] and record['override_active']
    np.testing.assert_allclose(state.target_offset, [-.4, -.02], atol=1e-7)
    np.testing.assert_array_equal(state.mask, [False, True])
    assert float(action[0]) == np.float32(-.3)


def test_nonfinite_offset_or_wrong_decision_boundary_is_an_explicit_engineering_fault():
    for tick, offset in [(4, j.array([j.nan, 0.])), (3, j.array([0., 0.])), (-4, j.array([0., 0.]))]:
        _, state, record = act(observation(-.03), tick, offset=offset)
        assert record['engineering_fault'] and not state.fired and not record['override_active']


def test_contract_rejects_same_dimension_with_different_field_order_or_timing():
    from sttw_control.causal_reference_baseline import validate_causal_contract
    context = [{'name': 'unused', 'scale': 1.} for _ in range(15)]
    context[0] = {'name': 'alpha', 'scale': 1.}
    context[4] = {'name': 'curvature_here_1_m', 'scale': .6}
    context[9] = {'name': 'external_speed_m_s', 'scale': 3.}
    config = dict(timing=dict(lower_dt_s=.005, upper_dt_s=.02),
                  network=dict(history_frames=16, frame_fields=20, mask_fields=16, actor_dim=351,
                               history_dt_s=.02, context_fields=context))
    fields = [{'name': 'unused', 'scale': 1.} for _ in range(20)]
    fields[18] = {'name': 'eso_equilibrium_shift_rad', 'scale': .3}
    validate_causal_contract(config, fields)
    bad = deepcopy(config); bad['timing']['lower_dt_s'] = .01
    with pytest.raises(ValueError): validate_causal_contract(bad, fields)
    bad = deepcopy(config); bad['network']['context_fields'][9]['scale'] = 2.
    with pytest.raises(ValueError): validate_causal_contract(bad, fields)
    bad_fields = deepcopy(fields); bad_fields[18]['name'] = 'different_sensor'
    with pytest.raises(ValueError): validate_causal_contract(config, bad_fields)
