"""STTW network and inactive-lane adapter for the Go2 checkout's RSL-RL PPO.

PPO losses, optimizer, adaptive learning rate and GAE stay upstream. Failed
STTW lanes remain inactive until the existing synchronous reset and are not
optimization samples. No quadruped symmetry mapping applies to this policy.
"""
import torch
from rsl_rl.storage import RolloutStorage


class H75ActorCritic(torch.nn.Module):
    is_recurrent = False

    def __init__(self, actor, critic,log_std=None):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.log_std=None if log_std is None else torch.nn.Parameter(log_std.clone())

    @classmethod
    def from_original_policy(cls,config,state_dict,device):
        """Reuse original alpha-modulated Actor/Critic and learned log std.

        The new Go2 optimizer is separate; this is a weights-only import.
        """
        from .torch_path_policy import make_torch_path_network
        policy=cls(make_torch_path_network(351,2,config),make_torch_path_network(352,1,config),state_dict['log_std'])
        policy.load_state_dict(state_dict)
        return policy.to(device)

    def act(self, observations, **kwargs):
        self.distribution = self.actor.distribution(observations) if self.log_std is None else torch.distributions.Normal(self.actor(observations),self.std)
        return self.distribution.sample()

    def act_inference(self,observations):
        return self.actor(observations)

    @property
    def std(self):
        return (self.actor.log_std if self.log_std is None else self.log_std).exp()

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
