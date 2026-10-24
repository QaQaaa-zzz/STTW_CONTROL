"""Sampling must retain the raw action and its actual collection density."""
import numpy as np
import jax
import jax.numpy as j

from sttw_control.survival_collection import sample_gaussian_action


def test_recorded_density_is_joint_raw_gaussian_not_tanh_density():
    import torch
    mean = j.array([1.5, -.4], dtype=j.float32)
    log_std = j.array([-.5, -1.], dtype=j.float32)
    action, logp, next_key = jax.jit(sample_gaussian_action)(mean, log_std, jax.random.PRNGKey(71))
    expected = torch.distributions.Normal(torch.tensor(np.asarray(mean)),
                                         torch.tensor(np.exp(np.asarray(log_std))))
    np.testing.assert_allclose(logp, expected.log_prob(torch.tensor(np.asarray(action))).sum(), atol=1e-6)
    assert np.max(np.abs(np.asarray(action))) > 1  # no premature tanh or clipping
    assert not np.array_equal(next_key, jax.random.PRNGKey(71))


def test_saved_key_reproduces_action_but_advanced_key_gives_new_action():
    mean, scale, key = j.zeros(2), j.array([-.2, -.8]), jax.random.PRNGKey(19)
    first = sample_gaussian_action(mean, scale, key)
    repeat = sample_gaussian_action(mean, scale, key)
    second = sample_gaussian_action(mean, scale, first[2])
    np.testing.assert_array_equal(first[0], repeat[0])
    assert not np.array_equal(first[0], second[0])
