"""Frozen nominal Gaussian retention: KL direction, isolation and PPO wiring."""
import copy

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

from sttw_control.direct_command_ppo import DirectCommandPPO


def algorithm():
    obs = TensorDict({'policy': torch.zeros(2, 351), 'critic': torch.zeros(2, 352)}, batch_size=[2])
    policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['critic']}, 2,
                         actor_hidden_dims=[8], critic_hidden_dims=[8],
                         noise_std_type='log', init_noise_std=.25)
    storage = RolloutStorage('rl', 2, 2, obs, [2], device='cpu')
    algo = DirectCommandPPO(policy, storage, num_learning_epochs=1, num_mini_batches=1,
                           desired_kl=100., device='cpu', actor_lr=1e-4, critic_lr=1e-4,
                           adam_betas=(.9,.999), adam_epsilon=1e-5, weight_decay=0.,
                           std_min=.01, std_max=2., hard_kl=100., actor_grad_limit=10.,
                           critic_grad_limit=10., kl_group_index=None)
    return algo, obs


def fill(algo, obs):
    with torch.no_grad():
        for _ in range(2):
            algo.act(obs)
            algo.process_env_step(obs, torch.tensor([1., 2.]), torch.zeros(2, dtype=torch.bool), {})
        algo.compute_returns(obs)


def test_zero_kl_and_distribution_is_not_overwritten():
    algo, _ = algorithm()
    obs = torch.randn(5, 351)
    means = algo.policy.actor(obs)
    algo.configure_parent_anchor(obs, means, algo.policy.log_std, batch_size=3)
    distribution = algo.policy.distribution
    torch.testing.assert_close(algo.parent_anchor_kl(), torch.tensor(0.), atol=1e-6, rtol=0.)
    assert algo.policy.distribution is distribution


def test_closed_form_direction_gradients_and_detached_copies():
    algo, _ = algorithm()
    obs = torch.randn(4, 351, requires_grad=True)
    teacher_means = (algo.policy.actor(obs).detach() + .3).requires_grad_()
    teacher_log_std = torch.tensor([-.2, -.7], requires_grad=True)
    algo.configure_parent_anchor(obs, teacher_means, teacher_log_std, batch_size=4)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(teacher_means.detach(), teacher_log_std.detach().exp()),
        torch.distributions.Normal(algo.policy.actor(obs.detach()), algo.policy.log_std.exp())).sum(-1).mean()
    loss = algo.parent_anchor_kl()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in algo.policy.actor.parameters())
    assert bool(algo.policy.log_std.grad.abs().sum() > 0)
    assert obs.grad is teacher_means.grad is teacher_log_std.grad is None
    with torch.no_grad():
        obs.fill_(999.)
        teacher_means.fill_(999.)
        teacher_log_std.fill_(999.)
    torch.testing.assert_close(algo.parent_anchor_kl(), loss.detach())


def test_cyclic_batches_wrap_without_rng_use():
    algo, _ = algorithm()
    obs = torch.zeros(5, 351)
    means = algo.policy.actor(obs).detach() + torch.arange(5.)[:, None]
    algo.configure_parent_anchor(obs, means, algo.policy.log_std, batch_size=3)
    rng = torch.get_rng_state().clone()
    values = [float(algo.parent_anchor_kl().detach()) for _ in range(3)]
    variance = float(algo.policy.log_std[0].exp().square().detach())
    assert values == pytest.approx([5 / (3 * variance), 25 / (3 * variance), 14 / (3 * variance)])
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('bad', ['shape', 'nan', 'coefficient', 'batch_size'])
def test_invalid_anchor_rejected(bad):
    algo, _ = algorithm()
    obs, means, std = torch.zeros(3, 351), torch.zeros(3, 2), torch.zeros(2)
    kwargs = {}
    if bad == 'shape': obs = obs[:, :-1]
    if bad == 'nan': means[0, 0] = float('nan')
    if bad == 'coefficient': kwargs['coefficient'] = -1.
    if bad == 'batch_size': kwargs['batch_size'] = 0
    with pytest.raises(ValueError): algo.configure_parent_anchor(obs, means, std, **kwargs)


def test_disabled_compatibility_and_update_metric():
    algo, obs = algorithm()
    other = copy.deepcopy(algo)
    means = algo.policy.actor(obs['policy']).detach() + .1
    other.configure_parent_anchor(obs['policy'], means, algo.policy.log_std, coefficient=0.)
    state = torch.get_rng_state()
    fill(algo, obs)
    result = algo.update()
    torch.set_rng_state(state)
    fill(other, obs)
    other_result = other.update()
    assert result['parent_kl'] == other_result['parent_kl'] == 0.
    for key, tensor in algo.policy.state_dict().items():
        assert torch.equal(tensor, other.policy.state_dict()[key])
    algo.configure_parent_anchor(obs['policy'], means, algo.policy.log_std, coefficient=.05)
    fill(algo, obs)
    assert algo.update()['parent_kl'] > 0.


def test_positive_anchor_changes_actor_update_and_epoch_restore_restores_cursor():
    algo, obs = algorithm()
    other = copy.deepcopy(algo)
    means = algo.policy.actor(obs['policy']).detach() + .5
    algo.configure_parent_anchor(obs['policy'], means, algo.policy.log_std, batch_size=3)
    snapshot = algo._snapshot()
    algo.parent_anchor_kl()
    assert algo.parent_anchor_cursor == 1
    algo._restore(snapshot)
    assert algo.parent_anchor_cursor == 0
    state = torch.get_rng_state()
    fill(algo, obs)
    algo.update()
    torch.set_rng_state(state)
    fill(other, obs)
    other.update()
    assert any(not torch.equal(p, q) for p, q in zip(algo.policy.actor.parameters(), other.policy.actor.parameters()))


def test_value_only_keeps_anchor_cursor_and_actor_frozen():
    algo, obs = algorithm()
    means = algo.policy.actor(obs['policy']).detach() + .5
    algo.configure_parent_anchor(obs['policy'], means, algo.policy.log_std, batch_size=3)
    fill(algo, obs)
    result = algo.value_only_update(1)
    assert result['actor_bitwise_unchanged']
    assert algo.parent_anchor_cursor == 0
