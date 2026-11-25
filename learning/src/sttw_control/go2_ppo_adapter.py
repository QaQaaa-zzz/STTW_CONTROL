"""STTW network and inactive-lane adapter for the Go2 checkout's RSL-RL PPO.

PPO losses, optimizer, adaptive learning rate and GAE stay upstream. Failed
STTW lanes remain inactive until the existing synchronous reset and are not
optimization samples. No quadruped symmetry mapping applies to this policy.
"""
import math
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


def verify_initial_weights(current, reference, excluded=()):
    """Require the declared key difference and exact identity of every shared tensor."""
    if set(current) != set(reference) - set(excluded) or not set(excluded) <= set(reference):
        raise ValueError('Paired initialization tensor names differ from the declared architecture')
    for name, value in current.items():
        a, b = value.detach().cpu(), reference[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
            raise ValueError(f'Paired initialization mismatch: {name}')
    return dict(arrays=len(current), elements=sum(v.numel() for v in current.values()),
                identical=not excluded, identical_shared=True, reference_only=sorted(excluded))


def init_near_zero_output(actor, gain=0.01):
    """Scale the final actor linear layer near zero so fresh training starts near Zero."""
    last = actor[-1]
    with torch.no_grad():
        last.weight.mul_(gain)
        last.bias.zero_()


class SeparatedPPO(PPO):
    """PPO with independent actor/critic optimizers, clipping, warmup, and KL guard.

    Prevents large failure-penalty value losses (~1e5) from crushing actor gradients
    under joint gradient clipping or triggering explosive adaptive LR growth.
    """

    def __init__(
        self,
        actor_critic,
        actor_lr=1e-4,
        critic_lr=3e-4,
        grad_clip_actor=1.0,
        grad_clip_critic=1.0,
        critic_warmup_updates=2,
        hard_kl=0.03,
        std_min=0.05,
        std_max=0.5,
        **kwargs,
    ):
        kwargs['learning_rate'] = actor_lr
        kwargs['schedule'] = 'fixed'
        kwargs['use_clipped_value_loss'] = False
        super().__init__(actor_critic, **kwargs)
        actor_params = list(actor_critic.actor.parameters())
        if actor_critic.log_std is not None:
            actor_params.append(actor_critic.log_std)
        self.actor_params = actor_params
        self.critic_params = list(actor_critic.critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=critic_lr)
        self.optimizer = self.actor_optimizer
        self.learning_rate = actor_lr
        self.critic_lr = critic_lr
        self.grad_clip_actor = grad_clip_actor
        self.grad_clip_critic = grad_clip_critic
        self.critic_warmup_updates = critic_warmup_updates
        self.hard_kl = hard_kl
        self.log_std_min = math.log(std_min)
        self.log_std_max = math.log(std_max)
        self.completed_updates = 0
        self.last_kl = 0.0
        self.last_actor_steps = 0

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy_loss = 0.0
        actor_active = self.completed_updates >= self.critic_warmup_updates
        actor_steps = 0
        last_kl = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            _target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
        ) in generator:
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            value_loss = (returns_batch - value_batch).pow(2).mean()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_params, self.grad_clip_critic)
            self.critic_optimizer.step()
            mean_value_loss += value_loss.item()

            if actor_active:
                self.actor_critic.act(obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-8)
                        + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                        / (2.0 * sigma_batch.square())
                        - 0.5,
                        dim=-1,
                    )
                    last_kl = kl.mean().item()
                if self.hard_kl is not None and last_kl > self.hard_kl:
                    actor_active = False
                    continue
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                entropy_batch = self.actor_critic.entropy
                ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
                adv = advantages_batch.squeeze(-1)
                surrogate = -adv * ratio
                surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                actor_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip_actor)
                self.actor_optimizer.step()
                if self.actor_critic.log_std is not None:
                    with torch.no_grad():
                        self.actor_critic.log_std.clamp_(self.log_std_min, self.log_std_max)
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss += entropy_batch.mean().item()
                actor_steps += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        if actor_steps:
            mean_surrogate_loss /= actor_steps
            mean_entropy_loss /= actor_steps
        self.completed_updates += 1
        self.last_kl = last_kl
        self.last_actor_steps = actor_steps
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, 0.0, mean_entropy_loss


class H75ActorCritic(torch.nn.Module):
    is_recurrent = False

    def __init__(self, actor, critic,log_std=None,active_channel=None):
        super().__init__()
        from .reference_action_channel import CHANNEL_INDEX
        index = None if active_channel is None else CHANNEL_INDEX[active_channel]
        self.action_slice = slice(None) if index is None else slice(index,index+1)
        self.single_channel = index is not None
        self.actor = actor
        self.critic = critic
        self.log_std=None if log_std is None else torch.nn.Parameter(log_std.clone())

    @classmethod
    def from_original_policy(cls,config,state_dict,device,active_channel=None):
        """Reuse original alpha-modulated Actor/Critic and learned log std.

        The new Go2 optimizer is separate; this is a weights-only import.
        """
        from .torch_path_policy import make_torch_path_network
        policy=cls(make_torch_path_network(351,2,config),make_torch_path_network(352,1,config),state_dict['log_std'],active_channel=active_channel)
        policy.load_state_dict(state_dict)
        return policy.to(device)

    def act(self, observations, **kwargs):
        if self.log_std is None and not self.single_channel:
            self.distribution=self.actor.distribution(observations)
        else:
            self.distribution=torch.distributions.Normal(self.act_inference(observations),self.std)
        return self.distribution.sample()

    def act_inference(self,observations):
        return self.actor(observations)[...,self.action_slice]

    @property
    def std(self):
        return (self.actor.log_std if self.log_std is None else self.log_std)[...,self.action_slice].exp()

    def evaluate(self, observations, **kwargs):
        return self.critic(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def reset(self, dones=None):
        pass


class ActiveLaneStorage(RolloutStorage):
    """Use upstream storage/GAE, exclude the existing inactive padding in PPO."""
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        mask = self.valid.flatten()
        names = ('observations', 'privileged_observations', 'actions', 'values',
                 'advantages', 'returns', 'actions_log_prob', 'mu', 'sigma')
        data = [getattr(self, name).flatten(0, 1)[mask] for name in names]
        advantage = data[5] - data[3]
        data[4] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        indices = torch.randperm(len(advantage), device=self.device).tensor_split(num_mini_batches)
        for _ in range(num_epochs):
            for index in indices:
                yield *(value[index] for value in data), (None, None), None
