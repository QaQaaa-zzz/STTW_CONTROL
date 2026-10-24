"""Reference-coordinate extension; no physical symmetry is assumed."""
import jax
import jax.numpy as j
import numpy as np
import pytest

from sttw_control.causal_reference_baseline import initial_gate, causal_reference_action
from sttw_control.reflected_causal_reference import reflected_reference_action, select_reference_rule


CONFIG = {'residual_interface': {'speed_negative_scale_m_s': 1.,
                               'speed_positive_scale_m_s': .25,
                               'steer_scale_rad': .2}}


def obs(speed, curvature, signal, alpha=0.):
    return (j.zeros(351).at[334:336].set(1.).at[318].set(-signal)
            .at[336].set(alpha).at[340].set(curvature/.6).at[345].set(speed/3.))


run = jax.jit(reflected_reference_action)


def compare(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_original_contexts_preserve_all_actions_states_and_telemetry():
    for speed, curvature, signal in ((1.5, -.9, -.03), (1.8, .9, .03)):
        old, new = initial_gate(), initial_gate()
        for tick in range(0, 120, 4):
            current = obs(speed, curvature, signal if tick >= 8 else 0.)
            action, offset = j.array([-.3 + tick*.001, .7]), j.array([-.4, .02])
            a = causal_reference_action(action, current, offset, j.int32(tick), old, CONFIG)
            b = run(action, current, offset, j.int32(tick), new, CONFIG)
            compare(a, b)
            old, new = a[1], b[1]


def test_opposite_turn_selects_same_speed_or_reflected_steer_and_then_hands_back():
    original, offset = j.array([-.3, .7]), j.array([-.4, .02])
    for speed, curvature, signal, target, mask in (
        (1.5, .9, .03, [-.25, .02], [True, False]),
        (1.8, -.9, -.03, [-.4, .06], [False, True]),
    ):
        action, state, telemetry = run(original, obs(speed, curvature, signal), offset, j.int32(100), initial_gate(), CONFIG)
        assert telemetry['detected'] and telemetry['override_active']
        np.testing.assert_allclose(state.target_offset, target, atol=1e-7)
        np.testing.assert_array_equal(state.mask, mask)
        np.testing.assert_array_equal(np.asarray(action)[~np.array(mask)], np.asarray(original)[~np.array(mask)])
        # A later curvature sign change must NOT turn a latched target around.
        held, later, log = run(original, obs(speed, -curvature, -signal), offset, j.int32(176), state, CONFIG)
        np.testing.assert_array_equal(later.target_offset, state.target_offset)
        np.testing.assert_array_equal(held, action)
        assert not log['detected'] and log['override_active']
        returned, _, log = run(original, obs(speed, -curvature, -signal), offset, j.int32(180), later, CONFIG)
        np.testing.assert_array_equal(returned, original)
        assert not log['override_active']


def test_inactive_preferences_opposite_events_and_invalid_histories_stay_inactive():
    original, offset = j.array([-.3, .7]), j.array([-.4, .02])
    for current in (obs(1.5, .9, .03, alpha=1.), obs(1.8, -.9, .03),
                    obs(1.5, .9, .03).at[334].set(0.), obs(2.4, .9, .03)):
        action, _, log = run(original, current, offset, j.int32(4), initial_gate(), CONFIG)
        np.testing.assert_array_equal(action, original)
        assert not log['override_active']
    _, state, log = run(original, obs(1.5, .9, .03).at[340].set(j.nan), offset, j.int32(4), initial_gate(), CONFIG)
    assert log['engineering_fault'] and not state.fired


def test_explicit_rule_identity_never_silently_defaults_an_unknown_variant():
    assert select_reference_rule('baseline') is causal_reference_action
    assert select_reference_rule('reflected_v1') is reflected_reference_action
    with pytest.raises(ValueError):
        select_reference_rule('reflected_typo')
