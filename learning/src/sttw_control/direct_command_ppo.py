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
    flags = {}
    for tensor in tensors:
        flags.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    # Adam step counters may be CPU tensors; never stack across devices.
    return all(bool(torch.stack(group).all()) for group in flags.values())


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
        self.std_min = torch.as_tensor(std_min, device=self.policy.log_std.device,
                                       dtype=self.policy.log_std.dtype).expand_as(self.policy.log_std).clone()
        self.std_max = torch.as_tensor(std_max, device=self.policy.log_std.device,
                                       dtype=self.policy.log_std.dtype).expand_as(self.policy.log_std).clone()
        if (not _finite_tensors(self.std_min, self.std_max)
                or bool((self.std_min <= 0).any())
                or bool((self.std_max < self.std_min).any())):
            raise ValueError("latent std bounds must be finite, positive, and ordered")
        self.hard_kl = float(hard_kl)
        self.actor_grad_limit = float(actor_grad_limit)
        self.critic_grad_limit = float(critic_grad_limit)
        self.hard_kl_stop = False
        self.nonfinite_stop = False
        self.last_update = None
        self.temporal_spec = None
        self.accepted_policy_updates = 0

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
        totals = torch.zeros(2, device=old_mu.device, dtype=torch.float64)
        counts = torch.zeros(2, device=old_mu.device, dtype=torch.int64)
        for start in range(0, old_mu.shape[0], 16384):
            end = min(start + 16384, old_mu.shape[0])
            new_mu = self.policy.act_inference(observations[start:end])
            row_kl = (torch.log(std / old_std[start:end])
                      + (old_std[start:end].square()
                         + (old_mu[start:end] - new_mu).square()) / (2 * std.square())
                      - .5).sum(dim=-1)
            if not _finite_tensors(row_kl):
                return float("nan"), {"0": float("nan"), "1": float("nan")}
            for index, value in enumerate((0., 1.)):
                mask = alpha[start:end] == value
                totals[index] += torch.where(mask, row_kl, 0.).sum()
                counts[index] += mask.sum()
        totals, counts = torch.stack((totals, counts.to(totals.dtype))).cpu().tolist()
        by_alpha = {str(i): totals[i] / counts[i] if counts[i] else float("nan") for i in (0,1)}
        count = sum(counts)
        mean = sum(totals) / count if count else float("nan")
        return (mean, by_alpha) if self.kl_group_index is not None else (mean, {"all": mean})

    def prepare_temporal_pairs(self):
        observations = self.storage.observations['policy']
        self.temporal_left = observations[:-1].reshape(-1, 345)
        self.temporal_right = observations[1:].reshape(-1, 345)
        self.temporal_valid = ~self.storage.dones[:-1].reshape(-1).bool()
        self.temporal_denominator = int(self.temporal_valid.sum())
        v5 = self.temporal_spec.get('preference_v5')
        cfg = (v5['temporal_regularizer'] if v5 else
               self.temporal_spec['actor_temporal_regularizer'])
        self.temporal_normalizers = cfg['physical_correction_normalizers']
        scales = observations.new_tensor([
            field['scale'] for field in self.temporal_spec['network']['frame_fields']])
        if v5:
            context_scales = observations.new_tensor([
                field['scale'] for field in self.temporal_spec['network']['context_fields']])
            limits = cfg['mask_both_endpoints']

        def eligible(obs):
            frame = obs[:, 300:320] * scales
            if not v5:
                return ((frame[:, 10].abs() <= cfg['eligible_raw_speed_rate_m_s2'])
                        & (frame[:, 11].abs() <= cfg['eligible_raw_steer_rate_rad_s'])
                        & (frame[:, 0].abs() < cfg['eligible_roll_rad'])
                        & (frame[:, 1].abs() < cfg['eligible_roll_rate_rad_s']))
            context = obs[:, 336:345] * context_scales
            return ((context[:, 4] < limits['chi_max'])
                    & (context[:, 1].abs() < limits['abs_heading_max_rad'])
                    & ((frame[:, 4] - frame[:, 8]).abs() < limits['abs_speed_error_max_m_s'])
                    & ((frame[:, 2] - frame[:, 9]).abs() < limits['abs_steer_error_max_rad'])
                    & (frame[:, 0].abs() < limits['abs_roll_max_rad'])
                    & (frame[:, 1].abs() < limits['abs_roll_rate_max_rad_s'])
                    & (frame[:, 10].abs() <= limits['raw_speed_rate_max_m_s2'])
                    & (frame[:, 11].abs() <= limits['raw_steer_rate_max_rad_s']))

        self.temporal_mask = (self.temporal_valid & eligible(self.temporal_left)
                              & eligible(self.temporal_right))
        if v5:
            start, end = cfg['zero_until_update'], cfg['linear_ramp_end_update']
            fraction = max(0., min((self.accepted_policy_updates - start) / (end - start), 1.))
            self.temporal_coefficient = cfg['max_weight'] * fraction
        else:
            self.temporal_coefficient = cfg['coefficient'] * min(
                self.accepted_policy_updates / cfg['ramp_policy_updates'], 1.)

    def temporal_loss(self, indices):
        # Pair preparation and randperm still run in their original order.
        if self.temporal_coefficient == 0:
            return next(self.policy.actor.parameters()).sum()*0.
        indices=indices[self.temporal_mask[indices]]
        if not len(indices) or not self.temporal_denominator:
            return next(self.policy.actor.parameters()).sum()*0.
        c=self.temporal_spec['action']
        def offset(obs):
            z=self.policy.actor(obs).tanh()
            return torch.stack((z[:,0]*torch.where(z[:,0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),z[:,1]*c['steer_scale_rad']),dim=-1)
        d=offset(self.temporal_right[indices])-offset(self.temporal_left[indices])
        norm=d.new_tensor(self.temporal_normalizers)
        return self.temporal_coefficient*(d/norm).square().sum()*self.num_mini_batches/self.temporal_denominator

    def value_only_update(self, epochs):
        original={k:v.detach().clone() for k,v in self.policy.state_dict().items() if not k.startswith('critic.')}
        total=norms=count=0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches,epochs):
            obs,_,_,_,returns,*_=batch
            loss=self.value_loss_coef*(returns-self.policy.evaluate(obs)).square().mean()
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.critic_parameters,self.critic_grad_limit)
            if not _finite_tensors(loss,norm):raise RuntimeError('nonfinite value-only update')
            if any(p.grad is not None for p in self.actor_parameters):raise RuntimeError('Actor gradient during value-only warmup')
            self.optimizer.step();total+=float(loss.detach());norms+=float(norm);count+=1
        assert all(torch.equal(v,self.policy.state_dict()[k]) for k,v in original.items())
        assert all(p not in self.optimizer.state for p in self.actor_parameters)
        self.storage.clear()
        return dict(value=total/count,critic_grad_norm=norms/count,actor_grad_norm=0.,actor_bitwise_unchanged=True)

    def update(self):
        if self.hard_kl_stop or self.nonfinite_stop:
            raise RuntimeError("direct command PPO is stopped; do not start another update")
        if self.storage.step != self.storage.num_transitions_per_env:
            raise ValueError("full rollout required before PPO update")
        if self.temporal_spec is not None:self.prepare_temporal_pairs()
        sums = {"temporal":0., "value": 0., "surrogate": 0., "entropy": 0.,
                "actor_grad_norm": 0., "critic_grad_norm": 0.}
        sums = {key: torch.zeros((),device=self.policy.log_std.device,dtype=torch.float64) for key in sums}
        attempted_grad_sums = {key: torch.zeros_like(sums["value"]) for key in ("actor","critic")}
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
                epoch_sums = {key: torch.zeros_like(value) for key,value in sums.items()}
                epoch_minibatches = 0
                bad = False
                if self.temporal_spec is not None:
                    temporal_chunks=torch.randperm(len(self.temporal_left),device=self.temporal_left.device).tensor_split(self.num_mini_batches)
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
                    temporal=self.temporal_loss(temporal_chunks[epoch_minibatches]) if self.temporal_spec is not None else surrogate_loss*0.
                    loss = (temporal + surrogate_loss + self.value_loss_coef * value_loss
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
                        self.policy.log_std.clamp_(self.std_min.log(), self.std_max.log())
                    optimizer_tensors = [item for state in self.optimizer.state.values()
                                         for item in state.values() if isinstance(item, torch.Tensor)]
                    if not _finite_tensors(*self.policy.parameters(), *optimizer_tensors):
                        bad = True
                        break
                    epoch_sums["temporal"] += temporal.detach()
                    epoch_sums["value"] += value_loss.detach()
                    epoch_sums["surrogate"] += surrogate_loss.detach()
                    epoch_sums["entropy"] += entropy.detach()
                    epoch_sums["actor_grad_norm"] += actor_norm
                    epoch_sums["critic_grad_norm"] += critic_norm
                    attempted_grad_sums["actor"] += actor_norm
                    attempted_grad_sums["critic"] += critic_norm
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
        summary = torch.stack([*sums.values(),*attempted_grad_sums.values()]).cpu().tolist()
        sums = dict(zip(sums,summary[:len(sums)]))
        attempted_grad_sums = dict(zip(attempted_grad_sums,summary[-2:]))
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
        if self.temporal_spec is not None:
            result.update(temporal_coefficient=self.temporal_coefficient,temporal_real_pairs=self.temporal_denominator,temporal_eligible_pairs=int(self.temporal_mask.sum()))
            if accepted_epochs:self.accepted_policy_updates+=1
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
    if (len(ppo["initial_latent_std"]) != 2
            or any(not math.isfinite(x) or x <= 0 for x in ppo["initial_latent_std"])):
        raise ValueError("initial latent std requires two finite positive channels")
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 2,
                         actor_hidden_dims=network["hidden_sizes"],
                         critic_hidden_dims=network["hidden_sizes"],
                         activation="elu", init_noise_std=ppo["initial_latent_std"][0],
                         noise_std_type="log", state_dependent_std=False,
                         actor_obs_normalization=False,
                         critic_obs_normalization=False).to(device)
    initialize_policy(policy)
    with torch.no_grad():
        policy.log_std.copy_(torch.tensor(ppo['initial_latent_std'],device=device).log())
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
        critic_grad_limit=ppo["max_grad_norm_critic"],
        kl_group_index=ppo.get('kl_group_index',ALPHA_INDEX))
