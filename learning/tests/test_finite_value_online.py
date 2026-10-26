"""Matched online value functions, collection identity and exact Adam resume."""
import copy

import numpy as np
import pytest
import torch

from sttw_control.path_command_policy import make_torch_path_network, make_path_critic
from sttw_control.finite_value_online import FiniteValueLearner, FiniteCollectionCritic

CFG = {'network': {'architecture': 'alpha_modulated_v1'}}


def source():
    torch.manual_seed(411)
    actor = make_torch_path_network(351, 2, CFG)
    critic = make_torch_path_network(352, 1, CFG)
    policy = {'actor.' + k: v for k, v in actor.state_dict().items()}
    policy.update({'critic.' + k: v for k, v in critic.state_dict().items()})
    policy['log_std'] = torch.tensor([-1., -2.])
    return policy


def learner(role):
    return FiniteValueLearner(CFG, source(), role=role, rate=.16, beta=0.,
                             seed=412, epochs=2, fixed_denominator=4)


def batch(obj):
    frozen = obj.freeze_collection()
    obs = torch.sin(torch.arange(702).float().reshape(2, 351)) * .1
    critic = torch.cat([obs, torch.tensor([[1.], [.002]])], dim=1)
    with torch.no_grad():
        mean = obj.actor(obs)
        action = mean + torch.tensor([[.03, -.02], [-.01, .02]])
        logp = torch.distributions.Normal(mean, obj.log_std.exp()).log_prob(action).sum(-1)
        task = obj.task_critic(critic).squeeze(-1)
        physical = obj.physical_critic(critic).squeeze(-1)
    return dict(collection_id=frozen['collection_id'], obs=obs, critic=critic,
                action=action, old_logp=logp, old_mean=mean, old_task=task,
                old_physical=physical, task_advantage=torch.tensor([1., -.5]),
                task_returns=task + torch.tensor([1., -.5]),
                physical_targets=torch.tensor([1., 0.]))


def test_three_roles_start_from_same_actor_hidden_features_and_value_function():
    objs = [learner(r) for r in ['free_GAE', 'finite_GAE', 'free_MC']]
    x = torch.zeros(4, 352)
    x[:, -1] = torch.tensor([0., .002, .5, 1.])
    expected = torch.tensor([-.16 * (1 - .997 ** (500 * t)) / .003
                             for t in [0., .002, .5, 1.]])
    for obj in objs:
        torch.testing.assert_close(obj.task_critic(x).squeeze(-1), expected)
        assert obj.task_critic(x[:1]).shape == (1, 1)
        assert obj.fingerprint(sampling_only=True) == objs[0].fingerprint(sampling_only=True)
        assert torch.equal(obj.task_critic.network[0].weight, objs[0].task_critic.network[0].weight)
    assert len({o.fingerprint() for o in objs}) == 3
    assert [o.gae_lambda for o in objs] == [.99, .99, 1.]


@pytest.mark.parametrize('role', ['free_GAE', 'finite_GAE', 'free_MC'])
def test_vmap_collection_values_match_torch_after_nonzero_optimizer_update(role):
    import jax
    import jax.numpy as j
    obj = learner(role)
    obj.update(batch(obj), warmup=True)
    # Full per-lane parameter layout, including a singleton/terminal query.
    x = torch.cos(torch.arange(5 * 352).float().reshape(5, 352)) * .15
    x[:, -1] = torch.tensor([0., .002, .1, .5, 1.])
    params = obj.sampling_parameters()['task_critic']
    stacked = jax.tree.map(lambda a: j.repeat(a[None], 5, axis=0), params)
    critic = FiniteCollectionCritic(make_path_critic(CFG))
    value = jax.jit(jax.vmap(lambda p, v: critic.apply(p, v)))(stacked, j.asarray(x.numpy()))
    np.testing.assert_allclose(np.asarray(value).reshape(-1),
                               obj.task_critic(x).detach().numpy().reshape(-1), rtol=2e-5, atol=1e-4)
    if role == 'finite_GAE':
        assert np.max(value) <= 0 and float(value[0, 0]) == 0.


@pytest.mark.parametrize('role', ['free_GAE', 'finite_GAE', 'free_MC'])
def test_wrapped_critic_uses_new_optimizer_and_boundary_resume_repeats_next_update(role):
    obj = learner(role)
    actor_before = copy.deepcopy(obj.actor.state_dict())
    head_before = obj.task_critic.network[6].weight.detach().clone()
    obj.update(batch(obj), warmup=True)
    assert not torch.equal(head_before, obj.task_critic.network[6].weight)
    assert all(torch.equal(v, actor_before[k]) for k, v in obj.actor.state_dict().items())
    obj.update(batch(obj), warmup=False)
    saved = obj.checkpoint()
    assert saved['schema'] == 'sttw_finite_value_boundary_v1'
    restored = FiniteValueLearner.restore(CFG, saved)
    assert restored.fingerprint() == obj.fingerprint()
    b1, b2 = batch(obj), batch(restored)
    assert b1['collection_id'] == b2['collection_id']
    obj.update(b1, warmup=False)
    restored.update(b2, warmup=False)
    assert restored.fingerprint() == obj.fingerprint()
    assert restored.optimizer_steps == obj.optimizer_steps == [4, 6, 6]


def test_changed_rate_or_return_estimator_cannot_consume_frozen_collection():
    obj = learner('free_GAE')
    b = batch(obj)
    obj.task_critic.rate = .17
    with pytest.raises(ValueError, match='changed'):
        obj.update(b, warmup=False)
    obj = learner('free_GAE')
    b = batch(obj)
    obj.gae_lambda = 1.
    with pytest.raises(ValueError, match='changed'):
        obj.update(b, warmup=False)


def test_pending_and_old_schema_are_not_restorable_as_new_boundary():
    obj = learner('finite_GAE')
    state = obj.checkpoint()
    state['schema'] = 'sttw_dual_survival_boundary_v1'
    with pytest.raises(ValueError, match='schema'):
        FiniteValueLearner.restore(CFG, state)
    obj.freeze_collection()
    with pytest.raises(ValueError, match='boundary'):
        obj.checkpoint()
