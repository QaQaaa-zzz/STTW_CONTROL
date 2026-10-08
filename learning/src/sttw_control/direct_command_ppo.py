"""RSL-RL 3.2.0 PPO adapter for the V3 direct command policy.

Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause

RSL owns rollout storage, action sampling, and GAE. This task-local update
adds separate learning rates, bounded latent std, and full-rollout epoch KL.
The stored PPO action is the raw Gaussian sample z, before tanh or any motor
mapping.
"""

import copy
import importlib.metadata
import math
import random

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from .direct_command_policy import initialize_policy


RSL_VERSION = "3.2.0"
ALPHA_INDEX = 336  # 16 frames * 20 fields + 16 valid masks


def _finite_tensors(*tensors):
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


class DirectCommandPPO(PPO):
    """RSL rollout/GAE with V3's fixed-rate, epoch-guarded PPO update."""

    def __init__(self, *args, actor_lr, critic_lr, adam_betas, adam_epsilon,
                 weight_decay, std_min, std_max, hard_kl, actor_grad_limit,
                 critic_grad_limit, kl_group_index=ALPHA_INDEX, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_group_index = kl_group_index
        self.actor_parameters = list(self.policy.actor.parameters()) + [self.policy.log_std]
        self.critic_parameters = list(self.policy.critic.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": self.actor_parameters, "lr": actor_lr},
            {"params": self.critic_parameters, "lr": critic_lr},
        ], betas=tuple(adam_betas), eps=adam_epsilon, weight_decay=weight_decay)
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        self.hard_kl = float(hard_kl)
        self.actor_grad_limit = float(actor_grad_limit)
        self.critic_grad_limit = float(critic_grad_limit)
        self.hard_kl_stop = False
        self.nonfinite_stop = False
        self.last_update = None

    def _snapshot(self):
        return {
            "policy": copy.deepcopy(self.policy.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()]
                        if torch.cuda.is_initialized() else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }

    def _restore(self, snapshot):
        self.policy.load_state_dict(snapshot["policy"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        torch.set_rng_state(snapshot["torch_rng"].cpu())
        if snapshot["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in snapshot["cuda_rng"]])
        np.random.set_state(snapshot["numpy_rng"])
        random.setstate(snapshot["python_rng"])

    @torch.no_grad()
    def _full_rollout_kl(self):
        observations = self.storage.observations.flatten(0, 1)
        old_mu = self.storage.mu.flatten(0, 1)
        old_std = self.storage.sigma.flatten(0, 1)
        alpha = (observations["policy"][:, self.kl_group_index] if self.kl_group_index is not None
                 else torch.zeros(old_mu.shape[0], device=old_mu.device))
        std = self.policy.log_std.exp()
        if not _finite_tensors(old_mu, old_std, std, alpha) or bool((old_std <= 0).any()):
            return float("nan"), {"0": float("nan"), "1": float("nan")}
        totals = {"0": 0., "1": 0.}
        counts = {"0": 0, "1": 0}
        for start in range(0, old_mu.shape[0], 16384):
            end = min(start + 16384, old_mu.shape[0])
            new_mu = self.policy.act_inference(observations[start:end])
            row_kl = (torch.log(std / old_std[start:end])
                      + (old_std[start:end].square()
                         + (old_mu[start:end] - new_mu).square()) / (2 * std.square())
                      - .5).sum(dim=-1)
            if not _finite_tensors(row_kl):
                return float("nan"), {"0": float("nan"), "1": float("nan")}
            for key, value in (("0", 0.), ("1", 1.)):
                mask = alpha[start:end] == value
                totals[key] += float(row_kl[mask].sum())
                counts[key] += int(mask.sum())
        by_alpha = {key: totals[key] / counts[key] if counts[key] else float("nan")
                    for key in ("0", "1")}
        count = sum(counts.values())
        mean = sum(totals.values()) / count if count else float("nan")
        return (mean, by_alpha) if self.kl_group_index is not None else (mean, {"all": mean})

    def update(self):
        if self.hard_kl_stop or self.nonfinite_stop:
            raise RuntimeError("direct command PPO is stopped; do not start another update")
        if self.storage.step != self.storage.num_transitions_per_env:
            raise ValueError("full rollout required before PPO update")
        sums = {"value": 0., "surrogate": 0., "entropy": 0.,
                "actor_grad_norm": 0., "critic_grad_norm": 0.}
        attempted_grad_sums = {"actor": 0., "critic": 0.}
        attempted_grad_count = 0
        attempted_epochs = accepted_epochs = 0
        attempted_minibatches = accepted_minibatches = 0
        mean_kl = float("nan")
        kl_by_alpha = {"0": float("nan"), "1": float("nan")}
        stop_reason = None
        try:
            for _ in range(self.num_learning_epochs):
                snapshot = self._snapshot()  # rejected epoch only; retain earlier epochs
                attempted_epochs += 1
                epoch_sums = {key: 0. for key in sums}
                epoch_minibatches = 0
                bad = False
                for (obs, actions, _old_values, advantages, returns, old_logprob,
                     _old_mu, _old_std, _hidden, _masks) in self.storage.mini_batch_generator(
                         self.num_mini_batches, 1):
                    attempted_minibatches += 1
                    if not _finite_tensors(obs["policy"], obs["critic"], actions,
                                           advantages, returns, old_logprob):
                        bad = True
                        break
                    self.policy.act(obs)
                    logprob = self.policy.get_actions_log_prob(actions)
                    value = self.policy.evaluate(obs)
                    entropy = self.policy.entropy.mean()
                    if not _finite_tensors(logprob, value, entropy):
                        bad = True
                        break
                    ratio = torch.exp(logprob - old_logprob.squeeze(-1))
                    surrogate = -(advantages.squeeze(-1) * ratio)
                    clipped = -(advantages.squeeze(-1)
                                * ratio.clamp(1 - self.clip_param, 1 + self.clip_param))
                    surrogate_loss = torch.maximum(surrogate, clipped).mean()
                    value_loss = (returns - value).square().mean()
                    loss = (surrogate_loss + self.value_loss_coef * value_loss
                            - self.entropy_coef * entropy)
                    if not _finite_tensors(loss):
                        bad = True
                        break
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradients = [p.grad for p in self.actor_parameters + self.critic_parameters
                                 if p.grad is not None]
                    if not _finite_tensors(*gradients):
                        bad = True
                        break
                    actor_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor_parameters, self.actor_grad_limit)
                    critic_norm = torch.nn.utils.clip_grad_norm_(
                        self.critic_parameters, self.critic_grad_limit)
                    if not _finite_tensors(actor_norm, critic_norm):
                        bad = True
                        break
                    self.optimizer.step()
                    with torch.no_grad():
                        self.policy.log_std.clamp_(math.log(self.std_min),
                                                   math.log(self.std_max))
                    optimizer_tensors = [item for state in self.optimizer.state.values()
                                         for item in state.values() if isinstance(item, torch.Tensor)]
                    if not _finite_tensors(*self.policy.parameters(), *optimizer_tensors):
                        bad = True
                        break
                    epoch_sums["value"] += float(value_loss.detach())
                    epoch_sums["surrogate"] += float(surrogate_loss.detach())
                    epoch_sums["entropy"] += float(entropy.detach())
                    epoch_sums["actor_grad_norm"] += float(actor_norm)
                    epoch_sums["critic_grad_norm"] += float(critic_norm)
                    attempted_grad_sums["actor"] += float(actor_norm)
                    attempted_grad_sums["critic"] += float(critic_norm)
                    attempted_grad_count += 1
                    epoch_minibatches += 1
                if bad:
                    self._restore(snapshot)
                    self.nonfinite_stop = True
                    stop_reason = "nonfinite"
                    break
                mean_kl, kl_by_alpha = self._full_rollout_kl()
                if not math.isfinite(mean_kl) or not all(map(math.isfinite, kl_by_alpha.values())):
                    self._restore(snapshot)
                    self.nonfinite_stop = True
                    stop_reason = "nonfinite_kl"
                    break
                if mean_kl > self.hard_kl:
                    self._restore(snapshot)
                    self.hard_kl_stop = True
                    stop_reason = "hard_kl"
                    break
                accepted_epochs += 1
                accepted_minibatches += epoch_minibatches
                for key in sums:
                    sums[key] += epoch_sums[key]
                if mean_kl > self.desired_kl:
                    stop_reason = "target_kl"
                    break
        except Exception:
            if "snapshot" in locals():
                self._restore(snapshot)
            self.storage.clear()
            self.nonfinite_stop = True
            raise
        self.storage.clear()
        result = {key: value / accepted_minibatches if accepted_minibatches else 0.
                  for key, value in sums.items()}
        result.update(mean_kl=mean_kl, exact_kl=mean_kl,
                      kl_by_alpha=kl_by_alpha, kl_alpha0=kl_by_alpha.get("0"),
                      kl_alpha1=kl_by_alpha.get("1"),
                      actor_grad_norm_preclip=result["actor_grad_norm"],
                      critic_grad_norm_preclip=result["critic_grad_norm"],
                      attempted_actor_grad_norm=(attempted_grad_sums["actor"] /
                                                 attempted_grad_count if attempted_grad_count else 0.),
                      attempted_critic_grad_norm=(attempted_grad_sums["critic"] /
                                                  attempted_grad_count if attempted_grad_count else 0.),
                      projected_std=self.policy.log_std.detach().exp().cpu().tolist(),
                      attempted_epochs=attempted_epochs, accepted_epochs=accepted_epochs,
                      attempted_minibatches=attempted_minibatches,
                      accepted_minibatches=accepted_minibatches,
                      completed_epochs=accepted_epochs,
                      completed_minibatches=accepted_minibatches,
                      hard_kl_stop=self.hard_kl_stop,
                      nonfinite_stop=self.nonfinite_stop,
                      stop_reason=stop_reason,
                      actor_lr=self.optimizer.param_groups[0]["lr"],
                      critic_lr=self.optimizer.param_groups[1]["lr"])
        self.last_update = result
        return result


def make_algorithm(obs, steps, spec, device):
    """Construct the 345/346/2 V3 PPO using the frozen numeric V3 spec."""
    installed = importlib.metadata.version("rsl-rl-lib")
    if installed != RSL_VERSION:
        raise RuntimeError(f"rsl-rl-lib=={RSL_VERSION} required; found {installed}")
    network = spec["network"]
    ppo = spec["ppo"]
    for name, dim, spec_key in (("policy", 345, "actor_input_dim"),
                                ("critic", 346, "critic_input_dim")):
        if name not in obs or len(obs[name].shape) != 2 or obs[name].shape[-1] != dim:
            raise ValueError(f"{name} observation must have shape [envs,{dim}]")
        if network[spec_key] != dim:
            raise ValueError(f"{name} spec identity mismatch")
    if network["actor_output_dim"] != 2:
        raise ValueError("raw latent action dimension must be 2")
    if network["hidden_sizes"] != [128, 128, 64] or network["activation"] != "elu":
        raise ValueError("V3 network identity mismatch")
    if steps <= 0 or len(obs.batch_size) != 1 or obs.batch_size[0] <= 0:
        raise ValueError("positive rollout steps and environment batch required")
    minibatches = int(ppo["minibatches"])
    if minibatches <= 0 or steps * obs.batch_size[0] % minibatches:
        raise ValueError("rollout size must divide evenly into minibatches")
    if len(ppo["initial_latent_std"]) != 2 or ppo["initial_latent_std"][0] != ppo["initial_latent_std"][1]:
        raise ValueError("RSL fixed Gaussian requires equal initial latent std")
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 2,
                         actor_hidden_dims=network["hidden_sizes"],
                         critic_hidden_dims=network["hidden_sizes"],
                         activation="elu", init_noise_std=ppo["initial_latent_std"][0],
                         noise_std_type="log", state_dependent_std=False,
                         actor_obs_normalization=False,
                         critic_obs_normalization=False).to(device)
    initialize_policy(policy)
    storage = RolloutStorage("rl", obs.batch_size[0], steps, obs, [2], device=device)
    return DirectCommandPPO(
        policy, storage, num_learning_epochs=ppo["epochs"],
        num_mini_batches=minibatches, clip_param=ppo["clip_ratio"],
        gamma=ppo["gamma"], lam=ppo["gae_lambda"],
        value_loss_coef=ppo["value_loss_coefficient"],
        entropy_coef=ppo["entropy_coefficient"],
        use_clipped_value_loss=False, schedule="fixed",
        desired_kl=ppo["target_mean_kl_after_epoch"],
        normalize_advantage_per_mini_batch=False, device=device,
        actor_lr=ppo["actor_learning_rate"], critic_lr=ppo["critic_learning_rate"],
        adam_betas=ppo["adam_betas"], adam_epsilon=ppo["adam_epsilon"],
        weight_decay=ppo["weight_decay"], std_min=ppo["latent_std_min"],
        std_max=ppo["latent_std_max"], hard_kl=ppo["hard_stop_mean_kl"],
        actor_grad_limit=ppo["max_grad_norm_actor"],
        critic_grad_limit=ppo["max_grad_norm_critic"])
