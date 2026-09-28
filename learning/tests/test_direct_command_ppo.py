"""Focused behavioral checks for the V3 raw-latent PPO adapter."""

import copy
import math

import pytest
import torch
from tensordict import TensorDict

from sttw_control.direct_command_ppo import make_algorithm


def _spec(*, epochs=2, target_kl=.01, hard_kl=.03):
    return {"network": {"actor_input_dim": 345, "critic_input_dim": 346,
                        "actor_output_dim": 2, "hidden_sizes": [128, 128, 64],
                        "activation": "elu"},
            "ppo": {"epochs": epochs, "minibatches": 2, "actor_learning_rate": 1e-4,
                    "critic_learning_rate": 3e-4, "gamma": .997, "gae_lambda": .97,
                    "clip_ratio": .2, "value_loss_coefficient": .5,
                    "entropy_coefficient": .001, "adam_betas": [.9, .999],
                    "adam_epsilon": 1e-5, "weight_decay": 0.,
                    "initial_latent_std": [.2, .2], "latent_std_min": .05,
                    "latent_std_max": .5, "max_grad_norm_actor": 1.,
                    "max_grad_norm_critic": 1.,
                    "target_mean_kl_after_epoch": target_kl,
                    "hard_stop_mean_kl": hard_kl}}


def _obs(n=4):
    actor = torch.zeros(n, 345)
    actor[:n // 2, 336] = 0.
    actor[n // 2:, 336] = 1.
    return TensorDict({"policy": actor, "critic": torch.zeros(n, 346)}, batch_size=[n])


def _rollout(algo, obs, steps=2):
    for step in range(steps):
        algo.act(obs)
        algo.process_env_step(obs, torch.full((len(obs),), float(step)),
                              torch.zeros(len(obs), dtype=torch.bool), {})
    algo.compute_returns(obs)


def test_initial_gaussian_is_raw_two_latents_and_separate_optimizers():
    obs = _obs()
    algo = make_algorithm(obs, 2, _spec(), "cpu")
    assert sum(p.numel() for p in algo.policy.actor.parameters()) == 69186
    assert algo.storage.actions.shape == (2, 4, 2)
    assert algo.policy.obs_groups == {"policy": ["policy"], "critic": ["critic"]}
    torch.testing.assert_close(algo.policy.act_inference(obs), torch.zeros(4, 2))
    torch.testing.assert_close(algo.policy.log_std.exp(), torch.full((2,), .2))
    assert [group["lr"] for group in algo.optimizer.param_groups] == [1e-4, 3e-4]
    assert [group["eps"] for group in algo.optimizer.param_groups] == [1e-5, 1e-5]


def test_storage_retains_raw_sample_and_log_probability():
    obs = _obs()
    algo = make_algorithm(obs, 2, _spec(), "cpu")
    z = algo.act(obs).clone()
    old_logprob = algo.transition.actions_log_prob.clone()
    algo.process_env_step(obs, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {})
    torch.testing.assert_close(algo.storage.actions[0], z)
    torch.testing.assert_close(algo.storage.actions_log_prob[0, :, 0], old_logprob)
    algo.policy.act(obs)
    ratio = torch.exp(algo.policy.get_actions_log_prob(z) - old_logprob)
    torch.testing.assert_close(ratio, torch.ones(4))


def test_true_task_end_does_not_bootstrap_but_live_rollout_edge_does():
    obs = _obs()
    algo = make_algorithm(obs, 2, _spec(), "cpu")
    critic_head = [layer for layer in algo.policy.critic.modules()
                   if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        critic_head.bias.fill_(2.)
    for step in range(2):
        algo.act(obs)
        done = torch.zeros(4, dtype=torch.bool)
        if step == 1:
            done[0] = True  # finite task end or physical failure
        algo.process_env_step(obs, torch.zeros(4), done, {})
    algo.compute_returns(obs)
    assert float(algo.storage.returns[1, 0, 0]) == 0.
    assert math.isclose(float(algo.storage.returns[1, 1, 0]),
                        .997 * 2., rel_tol=1e-6)


def test_epoch_records_alpha_kl_and_projected_std():
    obs = _obs()
    algo = make_algorithm(obs, 2, _spec(), "cpu")
    _rollout(algo, obs)
    assert abs(float(algo.storage.advantages.mean())) < 1e-5
    result = algo.update()
    assert result["attempted_epochs"] >= result["accepted_epochs"] >= 1
    assert result["attempted_minibatches"] >= result["accepted_minibatches"] >= 2
    assert set(result["kl_by_alpha"]) == {"0", "1"}
    assert all(math.isfinite(v) for v in result["kl_by_alpha"].values())
    assert math.isfinite(result["actor_grad_norm_preclip"])
    assert math.isfinite(result["critic_grad_norm_preclip"])
    assert all(.05 <= value <= .5 for value in result["projected_std"])
    assert algo.storage.step == 0


def test_hard_kl_rolls_back_rejected_epoch_and_stops_pilot():
    obs = _obs()
    algo = make_algorithm(obs, 2, _spec(hard_kl=-1.), "cpu")
    _rollout(algo, obs)
    before_policy = copy.deepcopy(algo.policy.state_dict())
    before_optimizer = copy.deepcopy(algo.optimizer.state_dict())
    before_rng = torch.get_rng_state().clone()
    # Generator shuffling consumes RNG; rollback returns exactly to epoch start.
    result = algo.update()
    assert result["hard_kl_stop"]
    assert result["attempted_epochs"] == 1
    assert result["accepted_epochs"] == 0
    assert result["accepted_minibatches"] == 0
    assert algo.optimizer.state_dict()["state"] == before_optimizer["state"]
    torch.testing.assert_close(torch.get_rng_state(), before_rng)
    for key, value in before_policy.items():
        torch.testing.assert_close(algo.policy.state_dict()[key], value)
    with pytest.raises(RuntimeError, match="stopped"):
        algo.update()


def test_nonfinite_rollout_stops_and_preserves_model():
    obs = _obs()
    algo = make_algorithm(obs, 2, _spec(), "cpu")
    _rollout(algo, obs)
    before = copy.deepcopy(algo.policy.state_dict())
    algo.storage.advantages[0, 0, 0] = float("nan")
    result = algo.update()
    assert result["nonfinite_stop"]
    assert result["accepted_epochs"] == 0
    for key, value in before.items():
        torch.testing.assert_close(algo.policy.state_dict()[key], value)


def test_rejects_partial_rollout_or_wrong_observation_identity():
    obs = _obs()
    algo = make_algorithm(obs, 2, _spec(), "cpu")
    with pytest.raises(ValueError, match="full rollout"):
        algo.update()
    wrong = _obs()
    wrong["critic"] = torch.zeros(4, 345)
    with pytest.raises(ValueError, match="critic"):
        make_algorithm(wrong, 2, _spec(), "cpu")
