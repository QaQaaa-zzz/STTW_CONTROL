import pickle
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np

from prepare_preference_history_baselines import zero_output_actor
from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
from sttw_control.reference_reserve_gate import GateConfig, gate_step, initial_gate


def test_zero_head_cannot_be_overridden_by_alpha_modulation():
    cfg = {'network': {'architecture': 'alpha_modulated_v1'}}
    actor = make_path_actor(cfg)
    obs = j.zeros((9, 351)).at[:, 336].set(j.linspace(-5., 5., 9))
    params = actor.init(jax.random.PRNGKey(7), obs)
    params['params']['Dense_3']['bias'] = j.array([2., -3.])
    params['params']['gain_2'] = j.ones(128)
    params['params']['shift_3'] = j.ones(64)
    original_bias = np.asarray(params['params']['Dense_3']['bias']).copy()
    zero = zero_output_actor(params)
    validate_actor_parameters(zero, cfg)
    np.testing.assert_array_equal(actor.apply(zero, obs), np.zeros((9, 2)))
    np.testing.assert_array_equal(params['params']['Dense_3']['bias'], original_bias)


def test_original_always_support_changes_only_speed_lower_bound():
    cfg = GateConfig()
    for speed in (-3., 0., 4.):
        action = j.array([speed, -.42])
        _, result, log = gate_step(initial_gate(), action, rho=j.nan, roll_rate=j.nan,
                                    mode=3, dt=.02, config=cfg)
        np.testing.assert_allclose(result, [max(speed, cfg.boost_latent), -.42], atol=1e-6)
        assert bool(log['active']) and not bool(log['fault'])
