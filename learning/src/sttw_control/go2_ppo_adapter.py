"""STTW network and inactive-lane adapter for the Go2 checkout's RSL-RL PPO.

PPO losses, optimizer, adaptive learning rate and GAE stay upstream. Failed
STTW lanes remain inactive until the existing synchronous reset and are not
optimization samples. No quadruped symmetry mapping applies to this policy.
"""
import torch
from rsl_rl.storage import RolloutStorage


class H75ActorCritic(torch.nn.Module):
    is_recurrent = False

    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def act(self, observations, **kwargs):
        self.distribution = self.actor.distribution(observations)
        return self.distribution.sample()

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
