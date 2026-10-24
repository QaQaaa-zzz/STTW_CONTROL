import numpy as np
import jax
import jax.numpy as j

from sttw_control.reference_prefix import inverse_offset_target, prefix_action
from sttw_control.path_command_policy import map_latent


CONFIG = {'residual_interface': {'speed_negative_scale_m_s': 1.,
                               'speed_positive_scale_m_s': .25,
                               'steer_scale_rad': .2}}


def test_asymmetric_inverse_roundtrip_and_saturation_receipt():
    for target in [[-.6, .04], [.15, -.04], [0., 0.], [.5, -.4]]:
        latent, realized, clipped = inverse_offset_target(j.array(target), CONFIG)
        np.testing.assert_allclose(map_latent(latent, CONFIG), realized, atol=1e-6)
        assert np.isfinite(latent).all()
        if target[0] == .5:
            assert np.asarray(clipped).all()
            assert realized[0] < .25 and realized[1] > -.2
        else:
            assert not np.asarray(clipped).any()
            np.testing.assert_allclose(realized, target, atol=1e-7)


def test_integer_boundary_hands_back_original_action_without_blending():
    original = j.array([-.31, .77])
    fixed = j.array([.2, -.4])
    run = jax.jit(jax.vmap(lambda tick: prefix_action(original, fixed, tick, 80, True)))
    actions, active = run(j.array([0, 76, 80, 84], dtype=j.int32))
    np.testing.assert_array_equal(actions[:2], np.broadcast_to(fixed, (2, 2)))
    np.testing.assert_array_equal(actions[2:], np.broadcast_to(original, (2, 2)))
    np.testing.assert_array_equal(active, [True, True, False, False])


def test_continue_arm_never_replaces_parent_action():
    original = j.array([-.31, .77])
    action, active = prefix_action(original, j.array([3., 4.]), 0, 80, False)
    np.testing.assert_array_equal(action, original)
    assert not active


def test_speed_rule_preserves_live_actor_steering_channel():
    original = j.array([-.31, .77])
    action, active = prefix_action(original, j.array([.2, -.4]), 0, 80, True,
                                   mask=j.array([True, False]))
    np.testing.assert_array_equal(action, j.array([.2, .77]))
    assert active
