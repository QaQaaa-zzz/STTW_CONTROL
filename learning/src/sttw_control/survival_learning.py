"""Opt-in dual-objective PPO algebra for authenticated complete-episode batches.

PPO and physical-tail masking are established techniques. This module provides
no data provenance, optimizer, rollout collection, or safety guarantee.
"""
import math
import torch
from .survival_credit import survival_ppo_loss


def dual_ppo_loss(*, new_log_prob, old_log_prob, task_advantage, task_values,
                  task_returns, physical_values, physical_targets,
                  old_physical_values, beta, fixed_denominator, clip=.2):
    """Loss on already selected, finite legal Actor rows only.

    Task advantages must be normalized/frozen once by the caller, not per epoch.
    Both Actor sums use the same declared episode/horizon denominator. Separate
    task and physical Critics use sample-mean MSE; they do not share Actor
    parameters. Beta zero is the task-only Actor control; both Critics may still
    be fitted for equal diagnostic compute. Entropy is handled by the learner.
    """
    if not isinstance(beta,(int,float)) or not math.isfinite(beta) or beta < 0:
        raise ValueError('finite nonnegative physical objective weight required')
    vectors=[new_log_prob,old_log_prob,task_advantage,task_values,task_returns,
             physical_values,physical_targets,old_physical_values]
    if (not all(isinstance(x,torch.Tensor) and x.ndim==1 and x.is_floating_point() for x in vectors)
            or not new_log_prob.numel() or any(x.shape!=new_log_prob.shape or x.device!=new_log_prob.device for x in vectors)
            or not all(bool(torch.isfinite(x).all()) for x in vectors)):
        raise ValueError('aligned finite legal decision vectors required before loss arithmetic')
    mask=torch.ones_like(new_log_prob,dtype=torch.bool)
    physical=survival_ppo_loss(new_log_prob,old_log_prob,physical_values,physical_targets,mask,
        old_values=old_physical_values,clip=clip,fixed_denominator=fixed_denominator)
    advantage=task_advantage.detach()
    ratio=(new_log_prob-old_log_prob.detach()).exp()
    task_actor=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).sum()/fixed_denominator
    task_value=(task_values-task_returns.detach()).square().mean()
    actor=task_actor+beta*physical['actor_loss']
    return dict(task_actor_loss=task_actor,physical_actor_loss=physical['actor_loss'],
        actor_loss=actor,task_value_loss=task_value,physical_value_loss=physical['value_loss'],
        total_loss=actor+task_value+physical['value_loss'])
