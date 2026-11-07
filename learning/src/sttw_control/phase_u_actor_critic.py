"""Original Brax Phase U network exposed through the Go2 RSL-RL PPO API.

Rollout storage holds pre-tanh actions. The environment applies tanh exactly once
before the original actuator mapping. KL uses the underlying Gaussian; log-prob
and entropy include the same stable tanh Jacobian as the source distribution.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class PhaseUActorCritic(nn.Module):
    is_recurrent=False

    def __init__(self,num_actor_obs,num_critic_obs,num_actions,bundle,**kwargs):
        super().__init__()
        if num_actor_obs not in (76,80) or (num_critic_obs,num_actions)!=(106,4):
            raise ValueError('Phase U requires 76 or 80 actor / 106 critic observations and 4 actions')
        path=Path(bundle);self.identity=json.loads((path/'identity.json').read_text())
        raw=(path/'actor.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.identity['artifact_sha256']:
            raise ValueError('Phase U export identity mismatch')
        data=dict(np.load(path/'actor.npz'))
        if num_actor_obs==80 and len(data['mean'])==76:
            data['kernel_0']=np.pad(data['kernel_0'],((0,4),(0,0)))
            data['mean']=np.r_[data['mean'],np.zeros(4,dtype=np.float32)]
            data['std']=np.r_[data['std'],np.ones(4,dtype=np.float32)]
        if num_actor_obs==80:self.identity=dict(self.identity,heading_observation=True,actor_observation_dim=80,network='80/106-256-256-256-8/1 SiLU',heading_observation_order=['platform_y_m','sin_platform_yaw','cos_platform_yaw_minus_one','platform_vy_m_s'],heading_frame='straight platform reference y=0, yaw=0; localized state required')
        for name in ('mean','std','critic_mean','critic_std'):
            self.register_buffer(name,torch.as_tensor(data[name]).clone())
        def network(prefix):
            layers=[]
            for i in range(4):
                weight=torch.as_tensor(data[f'{prefix}kernel_{i}']).T
                layer=nn.Linear(weight.shape[1],weight.shape[0])
                with torch.no_grad():
                    layer.weight.copy_(weight);layer.bias.copy_(torch.as_tensor(data[f'{prefix}bias_{i}']))
                layers.append(layer)
                if i<3:layers.append(nn.SiLU())
            return nn.Sequential(*layers)
        self.actor=network('');self.critic=network('critic_')
        self.distribution=None

    def reset(self,dones=None):
        pass

    def update_distribution(self,observations):
        logits=self.actor((observations-self.mean)/self.std)
        loc,scale=logits.chunk(2,-1)
        self.distribution=Normal(loc,torch.nn.functional.softplus(scale)+.001,validate_args=False)

    def act(self,observations,**kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    @staticmethod
    def log_jacobian(raw_action):
        return 2*(math.log(2)-raw_action-torch.nn.functional.softplus(-2*raw_action))

    def get_actions_log_prob(self,actions):
        return (self.distribution.log_prob(actions)-self.log_jacobian(actions)).sum(-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return (self.distribution.entropy()+self.log_jacobian(self.distribution.rsample())).sum(-1)

    def act_inference(self,observations):
        # Returned latent mean; the same environment tanh is used in eval/train.
        return self.actor((observations-self.mean)/self.std)[...,:4]

    def evaluate(self,critic_observations,**kwargs):
        return self.critic((critic_observations-self.critic_mean)/self.critic_std)
