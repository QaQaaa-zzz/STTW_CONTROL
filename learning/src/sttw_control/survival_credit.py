"""Opt-in complete-episode physical credit, independent of task reward/GAE.

Actor decisions precede their corresponding physical transition. The caller
marks one action per actual policy decision, not one per lower integration tick.
Tracking-domain exit is not a physical terminal event. Held-reference rows
remain observed physical transitions but have no Actor decision/gradient.
This module does not collect rollouts, change termination, or update a policy.
Partition-masked credit is established prior work, e.g. Daneshmand et al.,
SafeExplorer (2026), https://arxiv.org/abs/2607.08925. This complete-episode
physical-objective adapter is not a reproduction of its full training method.
"""
import numpy as np
from numbers import Integral


def resolve_survival_credit(*, active_tick, actor_action_valid, contact_failure,
                            engineering_fault, horizon_reached,
                            physical_tick, window_start_tick, window_ticks,
                            collection_policy_id, learner_policy_id):
    """Resolve one finite physical episode before any policy update.

    A verified window boundary is supplied by the collector's actual clock.
    Incomplete or engineering-unknown episodes have no labels, never a failure
    target. A contact at the horizon is failure; a simultaneous fault is unknown.
    Collection/learner IDs must identify the same frozen sampling parameters;
    this check cannot independently authenticate caller-provided identities.
    """
    if (not isinstance(collection_policy_id, str) or not collection_policy_id
            or collection_policy_id != learner_policy_id):
        raise ValueError('collection policy must match the unchanged learner policy')
    fields = [np.asarray(x) for x in [active_tick, actor_action_valid,
                                     contact_failure, engineering_fault, horizon_reached]]
    if (any(x.dtype != np.bool_ or x.ndim != 1 for x in fields)
            or fields[0].size == 0 or any(x.shape != fields[0].shape for x in fields)):
        raise ValueError('equal nonempty one-dimensional Boolean event arrays required')
    active, action, contact, fault, horizon = fields
    if np.any((action | contact | fault | horizon) & ~active):
        raise ValueError('inactive/padded rows cannot contain actions or terminal events')
    boundary = np.flatnonzero(active & (contact | fault | horizon))
    stop = int(boundary[0]) if boundary.size else active.size-1
    clock = np.asarray(physical_tick)
    if (clock.shape != active.shape or not np.issubdtype(clock.dtype, np.integer)
            or not isinstance(window_start_tick, Integral) or isinstance(window_start_tick, bool)
            or not isinstance(window_ticks, Integral) or isinstance(window_ticks, bool)
            or window_start_tick < 0 or window_ticks <= 0):
        raise ValueError('integer physical clock and positive observation window required')
    prefix = np.arange(active.size) <= stop
    observed_clock = clock[active & prefix]
    if observed_clock.size and (observed_clock[0] != window_start_tick
                               or np.any(np.diff(observed_clock) != 1)
                               or observed_clock[-1] >= window_start_tick+window_ticks):
        raise ValueError('physical observation coverage is missing, duplicated or outside window')
    if boundary.size and horizon[stop] and clock[stop] != window_start_tick+window_ticks-1:
        raise ValueError('horizon event precedes the complete physical observation window')
    if not boundary.size:
        outcome, target = 'pending', np.nan
    elif fault[stop]:
        outcome, target = 'unknown', np.nan
    elif contact[stop]:
        outcome, target = 'contact', 0.
    else:
        outcome, target = 'survived', 1.
    actor_mask = active & action & prefix & np.isfinite(target)
    targets = np.full(active.size, np.nan, dtype=np.float32)
    targets[actor_mask] = target
    return dict(outcome=outcome, target=float(target), targets=targets,
                actor_mask=actor_mask, actor_indices=np.flatnonzero(actor_mask),
                terminal_index=stop if boundary.size else None,
                physical_ticks=int(np.sum(active & prefix)),
                actor_actions=int(np.sum(actor_mask)),
                observed_actor_actions=int(np.sum(active & action & prefix)),
                collection_policy_id=collection_policy_id)


def survival_ppo_loss(new_log_prob, old_log_prob, values, targets, mask, *,
                      old_values, clip=.2, fixed_denominator=None):
    """Physical-objective PPO terms on resolved legal decisions only.

    Log probabilities are joint probabilities of sampled raw Actor actions.
    `old_values` and old log probabilities come from the frozen collection
    policy; do not recompute advantages using the changing minibatch Critic.
    Values are probabilities from a separate physical value head. There is no
    task-loss coefficient here: combining objectives and
    authenticating an on-policy collection belongs to the caller.

    By default the Actor term is a legal-sample mean (legacy engineering use).
    For a complete episode batch, supply a predeclared denominator such as
    episode_count * maximum_decisions_per_episode. This preserves the scale
    of a sum of score contributions when contact/hold changes action counts.
    It must not be recomputed from the number of valid actions. Minibatch
    sampling/reweighting remains the learner's responsibility. Critic MSE
    remains a legal-sample mean, independent of Actor objective reduction.

    Index before arithmetic. Network evaluation must ALSO exclude invalid
    observations before forward, as in survival_ppo_batch: output masking alone
    cannot remove NaN weight gradients from an already evaluated invalid row.
    """
    import math
    import torch
    tensors = [new_log_prob, old_log_prob, values, targets, old_values]
    if (not all(isinstance(x, torch.Tensor) and x.is_floating_point() and x.ndim == 1 for x in tensors)
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.ndim != 1 or any(x.shape != mask.shape for x in tensors)
            or any(x.device != mask.device for x in tensors)):
        raise ValueError('equal one-dimensional floating tensors and Boolean mask required')
    if not math.isfinite(clip) or not 0 < clip < 1:
        raise ValueError('finite PPO clip strictly between zero and one required')
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError('no resolved Actor decisions in this minibatch')
    denominator = count if fixed_denominator is None else fixed_denominator
    if (isinstance(denominator, bool) or not isinstance(denominator, (int, float))
            or not math.isfinite(denominator) or denominator < count):
        raise ValueError('finite fixed denominator at least the selected sample count required')
    new, old, value, target, baseline = [x[mask] for x in tensors]
    old, target, baseline = old.detach(), target.detach(), baseline.detach()
    if not all(bool(torch.isfinite(x).all()) for x in [new, old, value, target, baseline]):
        raise ValueError('selected survival samples must be resolved and finite')
    if (bool(((target != 0) & (target != 1)).any())
            or bool(((value < 0) | (value > 1) | (baseline < 0) | (baseline > 1)).any())):
        raise ValueError('binary survival targets and probability-valued Critic required')
    ratio = (new-old).exp()
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError('nonfinite PPO likelihood ratio')
    advantage = target-baseline
    actor_loss = -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).sum()/denominator
    value_loss = (value-target).square().mean()
    return dict(actor_loss=actor_loss, value_loss=value_loss, samples=count)


def survival_ppo_batch(evaluate_selected, observations, actions, old_log_prob,
                       old_values, targets, mask, *, clip=.2, fixed_denominator=None):
    """Evaluate only legal resolved samples, then form physical PPO terms.

    `evaluate_selected(obs, actions)` returns differentiable joint log-prob and
    probability-valued physical Critic output for those rows. It must not step
    an optimizer. Collection provenance/on-policy validation remains external.
    """
    import torch
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1
            or not all(isinstance(x, torch.Tensor) and x.is_floating_point()
                       and x.ndim == 2 and x.shape[0] == mask.numel() and x.device == mask.device
                       for x in [observations, actions])
            or not all(isinstance(x, torch.Tensor) and x.is_floating_point()
                       and x.shape == mask.shape and x.device == mask.device
                       for x in [old_log_prob, old_values, targets])):
        raise ValueError('aligned observations/actions and collection vectors required')
    if not bool(mask.any()):
        raise ValueError('no resolved Actor decisions in this minibatch')
    # Recorded state/action are data, even if an upstream collector supplied a
    # reparameterized sample with an autograd graph still attached.
    obs, action = observations[mask].detach(), actions[mask].detach()
    if not bool(torch.isfinite(obs).all() & torch.isfinite(action).all()):
        raise ValueError('selected pre-action observations and actions must be finite')
    log_prob, value = evaluate_selected(obs, action)
    selected = torch.ones(obs.shape[0], dtype=torch.bool, device=mask.device)
    return survival_ppo_loss(log_prob, old_log_prob[mask], value, targets[mask],
                             selected, old_values=old_values[mask], clip=clip,
                             fixed_denominator=fixed_denominator)
