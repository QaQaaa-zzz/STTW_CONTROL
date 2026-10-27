"""Real network/density/gradient CPU tests, no robot physics or policy campaign."""
import numpy as np
import torch
import jax
import jax.numpy as j
from sttw_control.path_command_policy import make_torch_path_network
from sttw_control.goal_networks import (GoalNetworkBundle, make_goal_collection_evaluator,
                                      goal_network_ppo_terms)
from sttw_control.survival_collection import sample_gaussian_action

CFG = {'network': {'architecture': 'alpha_modulated_v1'}}


def bundle():
    torch.manual_seed(14)
    parent = make_torch_path_network(351, 2, CFG)
    with torch.no_grad():
        for name in ('gain_2', 'shift_2', 'gain_3', 'shift_3'):
            getattr(parent, name).uniform_(-.02, .02)
    return parent, GoalNetworkBundle(parent, CFG, torch.tensor([-1., -2.]))


def views(n=3):
    obs = torch.sin(torch.arange(n * 351, dtype=torch.float32).reshape(n, 351)) * .1
    obs[:, 336] = torch.linspace(0., 1., n)
    critic = torch.cat([obs, torch.full((n, 1), .7)], -1)
    goal = torch.arange(n * 5, dtype=torch.float32).reshape(n, 5) * .03
    return obs, critic, goal


def test_warm_start_preserves_parent_action_and_does_not_modify_source():
    parent, net = bundle(); obs, critic, goal = views()
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    with torch.no_grad():
        mean, g, c, s = net(obs, critic, goal)
        torch.testing.assert_close(mean, parent(obs), rtol=0, atol=0)
        torch.testing.assert_close(g, torch.full((3,), .5)); assert not c.any()
        torch.testing.assert_close(s, torch.full((3,), .5))
        net.actor[6].bias.add_(1.)
    for k, v in parent.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_actual_per_lane_jax_values_mean_and_raw_density_match_torch():
    _, a = bundle(); _, b = bundle()
    with torch.no_grad():
        a.goal_branch.weight.fill_(.03); a.values.weight.fill_(.01)
        b.actor[6].bias.add_(.2); b.goal_branch.bias.fill_(.1); b.values.bias.copy_(torch.tensor([.3, 2., -.4]))
    obs, critic, goal = views(2)
    with jax.default_device(jax.devices('cpu')[0]):
        params = jax.tree.map(lambda *x: j.stack(x), a.sampling_parameters(), b.sampling_parameters())
        evaluator = make_goal_collection_evaluator(CFG)
        actual = jax.jit(jax.vmap(evaluator))(params, j.asarray(obs.numpy()), j.asarray(critic.numpy()), j.asarray(goal.numpy()))
        for i, net in enumerate((a, b)):
            with torch.no_grad(): expected = net(obs[i], critic[i], goal[i])
            for x, y in zip(actual, expected):
                np.testing.assert_allclose(x[i], y.numpy(), rtol=1e-5, atol=1e-6)
            raw, logp, _ = sample_gaussian_action(actual[0][i], params['log_std'][i], jax.random.PRNGKey(3))
            with torch.no_grad(): density = net.evaluate_actions(obs[i], critic[i], goal[i], torch.tensor(np.array(raw)))[0]
            np.testing.assert_allclose(logp, density.numpy(), rtol=1e-5, atol=2e-6)


def test_exported_snapshot_cannot_change_when_learner_parameters_change():
    _, net = bundle()
    with jax.default_device(jax.devices('cpu')[0]):
        p = net.sampling_parameters()
        old = jax.tree.map(lambda x: np.array(x, copy=True), p)
        with torch.no_grad():
            for x in net.parameters(): x.add_(.2)
        for x, y in zip(jax.tree.leaves(p), jax.tree.leaves(old)):
            np.testing.assert_array_equal(x, y)


def test_real_ppo_and_one_synthetic_optimizer_step_use_only_resolved_legal_rows():
    _, net = bundle(); obs, critic, goal = views(4)
    masks = {k: torch.tensor([True, True, False, False]) for k in ('goal', 'cost', 'survival')}
    with torch.no_grad():
        mean, g, c, s = net(obs, critic, goal)
        actions = mean + .1
        logp, _, _, _ = net.evaluate_actions(obs, critic, goal, actions)
    frozen = dict(old_log_prob=logp.clone(), old_goal_values=g.clone(), old_cost_values=c.clone(),
        old_physical_values=s.clone(), goal_targets=torch.ones(4), cost_targets=torch.full((4,), 2.),
        physical_targets=torch.ones(4))
    old_copy = {k: v.clone() for k, v in frozen.items()}
    for x in (obs, critic, goal, actions): x[2:] = torch.nan
    old_branch = net.goal_branch.weight.detach().clone()
    result = goal_network_ppo_terms(net, obs, critic, goal, actions,
        masks=masks, collected=frozen, objective_weights=dict(goal=1., cost=.1, survival=1.),
        fixed_denominator=4, cost_population_complete=True)
    total = result['actor_loss'] + sum(result['value_losses'].values())
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer.zero_grad(); total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    assert net.goal_branch.weight.grad.abs().sum() > 0
    optimizer.step()  # Synthetic CPU engineering only; no robot/data training run.
    assert not torch.equal(net.goal_branch.weight, old_branch)
    for k, v in frozen.items(): torch.testing.assert_close(v, old_copy[k], rtol=0, atol=0)


def test_only_goal_outcome_change_reverses_actual_goal_branch_parameter_update():
    _, yes = bundle(); _, no = bundle()
    obs, critic, goal = views(1)
    with torch.no_grad():
        mean, g, c, s = yes(obs, critic, goal)
        actions = mean + .1
        logp = yes.evaluate_actions(obs, critic, goal, actions)[0]
    for net, target in [(yes, 1.), (no, 0.)]:
        collected = dict(old_log_prob=logp, old_goal_values=g, old_cost_values=c,
            old_physical_values=s, goal_targets=torch.tensor([target]),
            cost_targets=torch.tensor([2.]), physical_targets=torch.ones(1))
        masks = dict(goal=torch.ones(1, dtype=torch.bool), cost=torch.zeros(1, dtype=torch.bool),
                     survival=torch.ones(1, dtype=torch.bool))
        loss = goal_network_ppo_terms(net, obs, critic, goal, actions,
            masks=masks, collected=collected, objective_weights=dict(goal=1., cost=0., survival=0.),
            fixed_denominator=1, cost_population_complete=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=.01)
        optimizer.zero_grad(); loss['actor_loss'].backward(); optimizer.step()
    assert (yes.goal_branch.weight * no.goal_branch.weight).sum() < 0


def test_value_head_loss_has_no_gradient_path_into_actor():
    _, net = bundle(); obs, critic, goal = views(2)
    _, g, c, s = net(obs, critic, goal)
    ((g - 1).square().mean() + (c - 2).square().mean() + (s - 1).square().mean()).backward()
    assert all(x.grad is None for x in net.actor.parameters())
    assert all(x.grad is None for x in net.goal_branch.parameters())
    assert net.values.bias.grad is not None
