"""Ordinary goal-informed baseline networks, opt-in and parent preserving.

These are engineering components, not a new constrained-control algorithm.
No production learner, collector run or checkpoint is changed on import.
"""
import copy
import math
import numpy as np
import torch
from .path_command_policy import export_torch_actor, make_path_actor
from .goal_task_credit import goal_task_ppo_batch


class GoalNetworkBundle(torch.nn.Module):
    """Copy a parent Actor; add a zero-initialized goal branch and separate values.

    Actor base parameters remain trainable. Frozen lower control is external.
    G/S are probabilities; C is an unconstrained scalar value prediction.
    New heads start at G=S=.5/C=0, not inherited task-value approximations.
    Both treatment and ordinary-learning controls must have identical inputs.
    """
    def __init__(self, parent_actor, config, log_std):
        super().__init__()
        # Validate the exact known Actor architecture before copying/migration.
        export_torch_actor(parent_actor, config)
        self.config = copy.deepcopy(config)
        self.actor = copy.deepcopy(parent_actor)
        anchor = next(self.actor.parameters())
        self.goal_branch = torch.nn.Linear(5, 2, device=anchor.device, dtype=anchor.dtype)
        self.values = torch.nn.Linear(357, 3, device=anchor.device, dtype=anchor.dtype)
        for layer in (self.goal_branch, self.values):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        std = torch.as_tensor(log_std, device=anchor.device, dtype=anchor.dtype).detach().clone()
        if std.shape != (2,) or not bool(torch.isfinite(std).all()):
            raise ValueError('two finite inherited raw Gaussian log standard deviations required')
        self.log_std = torch.nn.Parameter(std)

    def forward(self, observation, critic_observation, context):
        if (observation.shape[-1:] != (351,) or critic_observation.shape[-1:] != (352,) or
                context.shape[-1:] != (5,) or observation.shape[:-1] != context.shape[:-1] or
                critic_observation.shape[:-1] != context.shape[:-1]):
            raise ValueError('aligned original351/critic352/goal5 views required')
        mean = self.actor(observation) + self.goal_branch(context)
        values = self.values(torch.cat([critic_observation, context], dim=-1))
        return mean, values[..., 0].sigmoid(), values[..., 1], values[..., 2].sigmoid()

    def evaluate_actions(self, observation, critic_observation, context, actions):
        mean, g, c, s = self(observation, critic_observation, context)
        if actions.shape != mean.shape:
            raise ValueError('two raw sampled actions per selected observation required')
        logp = -.5 * (((actions - mean) * (-self.log_std).exp()).square() +
                      2 * self.log_std + math.log(2 * math.pi)).sum(dim=-1)
        return logp, g, c, s

    def sampling_parameters(self):
        """Detached owned-copy snapshot, including raw distribution parameters.

        Copy CPU buffers explicitly; a later Torch optimizer must not mutate an
        old sampling/advantage snapshot through a shared NumPy/JAX buffer.
        """
        import jax
        import jax.numpy as j
        owned = lambda x: j.array(np.array(x, copy=True), copy=True)
        dense = lambda layer: dict(kernel=owned(layer.weight.detach().cpu().numpy().T),
                                  bias=owned(layer.bias.detach().cpu().numpy()))
        return dict(actor=jax.tree.map(owned, export_torch_actor(self.actor, self.config)),
                    goal_branch=dense(self.goal_branch), values=dense(self.values),
                    log_std=owned(self.log_std.detach().cpu().numpy()))


def make_goal_collection_evaluator(config):
    """JAX single-lane callback for make_goal_live_collection_step; vmap externally."""
    import jax
    import jax.numpy as j
    actor = make_path_actor(config)

    def evaluate(params, observation, critic_observation, context):
        if observation.shape != (351,) or critic_observation.shape != (352,) or context.shape != (5,):
            raise ValueError('one original351/critic352/goal5 lane required')
        mean = actor.apply(params['actor'], observation)
        mean = mean + context @ params['goal_branch']['kernel'] + params['goal_branch']['bias']
        value = j.concatenate([critic_observation, context]) @ params['values']['kernel'] + params['values']['bias']
        return mean, jax.nn.sigmoid(value[0]), value[1], jax.nn.sigmoid(value[2])

    return evaluate


def goal_network_ppo_terms(bundle, observation, critic_observation, context, actions, *,
                           masks, collected, objective_weights, fixed_denominator,
                           cost_population_complete, clip=.2):
    """Wire actual network parameters into the existing selected-row PPO terms.

    Only finite legal/resolved rows reach the network. This performs no optimizer
    step. Caller authenticates all frozen old values/logp and the whole population.
    Weighted terms do not enforce goal priority or survival/alpha constraints.
    """
    if (observation.ndim != 2 or observation.shape[1] != 351 or
            critic_observation.shape != (observation.shape[0], 352) or
            context.shape != (observation.shape[0], 5)):
        raise ValueError('aligned original351/critic352/goal5 rows required')
    packed = torch.cat([observation, critic_observation, context], dim=-1)
    def evaluate(selected, raw_actions):
        return bundle.evaluate_actions(selected[:, :351], selected[:, 351:703], selected[:, 703:], raw_actions)
    return goal_task_ppo_batch(evaluate, packed, actions, masks=masks, collected=collected,
        objective_weights=objective_weights, fixed_denominator=fixed_denominator,
        cost_population_complete=cost_population_complete, clip=clip)
