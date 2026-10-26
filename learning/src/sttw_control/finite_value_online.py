"""Online adapter for the declared matched value/return-estimator experiment.

Actor, PPO loss, legal samples and physical credit remain in the existing
learner. This module changes only the task-value parameterization and records
the return estimator as part of collection/checkpoint provenance.
"""
import copy
import hashlib
import json
import math

import torch

from .finite_value_fit import FiniteValue
from .path_command_policy import MODULATION_SHAPES, export_torch_actor
from .preference_best import config_hash
from .survival_online_learner import OnlineSurvivalLearner


ROLES = {
    'free_GAE': ('reset_free', .99),
    'finite_GAE': ('time_nonpositive', .99),
    'free_MC': ('reset_free', 1.),
}


class _ColumnValue(FiniteValue):
    def forward(self, x):
        # Parent learner's scalar head convention is (batch, 1), even for N=1.
        return super().forward(x).unsqueeze(-1)


class FiniteCollectionCritic:
    """Same dynamic per-lane value transformation for the JAX collector."""
    def __init__(self, native):
        self.native = native

    def apply(self, parameters, observation):
        import jax.numpy as j
        import jax.nn
        raw = self.native.apply(parameters['network'], observation)
        remaining = j.clip(observation[..., -1:], 0., 1.)
        budget = -j.expm1(remaining * (500 * math.log(.997))) / .003
        return j.where(parameters['nonpositive'], -budget * jax.nn.softplus(raw),
                       raw - parameters['rate'] * budget)


class FiniteValueLearner(OnlineSurvivalLearner):
    def __init__(self, config, parent_policy, *, role, rate, **settings):
        if role not in ROLES:
            raise ValueError('explicit matched value role required')
        super().__init__(config, parent_policy, **settings)
        self.role = role
        mode, self.gae_lambda = ROLES[role]
        self.task_critic = _ColumnValue(self.task_critic, mode, rate=rate)
        # Rebind the task optimizer to the wrapper's copied network. Leaving
        # the parent's optimizer would silently update the discarded head.
        self.groups[1] = list(self.task_critic.parameters())
        self.optimizers[1] = torch.optim.Adam(self.groups[1], lr=self.settings['critic_lr'])

    def value_contract(self):
        return dict(role=self.role, mode=self.task_critic.mode,
                    rate=self.task_critic.rate, gae_lambda=self.gae_lambda,
                    gamma=.997, horizon_upper_steps=500)

    def fingerprint(self, *, sampling_only=False):
        base = super().fingerprint(sampling_only=sampling_only)
        if sampling_only:
            return base
        return hashlib.sha256((base + json.dumps(self.value_contract(), sort_keys=True)).encode()).hexdigest()

    def sampling_parameters(self):
        import jax.numpy as j
        network = self.task_critic.network
        params = {}
        for i, k in enumerate((0, 2, 4, 6)):
            params[f'Dense_{i}'] = dict(kernel=j.asarray(network[k].weight.detach().cpu().numpy().T),
                                        bias=j.asarray(network[k].bias.detach().cpu().numpy()))
        for k in MODULATION_SHAPES:
            if hasattr(network, k):
                params[k] = j.asarray(getattr(network, k).detach().cpu().numpy())
        return dict(actor=export_torch_actor(self.actor, self.config),
                    task_critic=dict(network={'params': params},
                                     nonpositive=j.asarray(self.task_critic.mode == 'time_nonpositive'),
                                     rate=j.asarray(self.task_critic.rate)),
                    physical=dict(kernel=j.asarray(self.physical_critic[0].weight.detach().cpu().numpy().T),
                                  bias=j.asarray(self.physical_critic[0].bias.detach().cpu().numpy())))

    def checkpoint(self):
        state = super().checkpoint()
        state['schema'] = 'sttw_finite_value_boundary_v1'
        state['value_contract'] = self.value_contract()
        return state

    @classmethod
    def restore(cls, config, state):
        if state['schema'] != 'sttw_finite_value_boundary_v1' or config_hash(config) != state['config_sha256']:
            raise ValueError('checkpoint schema/config mismatch')
        contract = state['value_contract']
        role = contract['role']
        if (role not in ROLES or (contract['mode'], contract['gae_lambda']) != ROLES[role]
                or contract['gamma'] != .997 or contract['horizon_upper_steps'] != 500):
            raise ValueError('checkpoint value/return contract mismatch')
        policy = {'actor.' + k: v for k, v in state['actor'].items()}
        policy.update({'critic.' + k.removeprefix('network.'): v
                       for k, v in state['task_critic'].items()})
        policy['log_std'] = state['log_std']
        obj = cls(config, policy, role=role, rate=contract['rate'], **state['settings'])
        obj.task_critic.load_state_dict(state['task_critic'], strict=True)
        obj.physical_critic.load_state_dict(state['physical_critic'], strict=True)
        for optimizer, saved in zip(obj.optimizers, state['optimizers']):
            optimizer.load_state_dict(copy.deepcopy(saved))
        obj.next_collection = state['next_collection']
        obj.actor_steps = state['actor_steps']
        obj.critic_steps = state['critic_steps']
        obj.optimizer_steps = list(state['optimizer_steps'])
        torch.set_rng_state(state['torch_rng_state'])
        return obj
