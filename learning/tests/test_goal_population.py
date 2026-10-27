"""Whole-inventory assembly into actual network PPO, using synthetic records."""
import copy
import numpy as np
import pytest
import torch
from sttw_control.goal_population import assemble_goal_population
from sttw_control.goal_networks import GoalNetworkBundle, goal_network_ppo_terms
from sttw_control.path_command_policy import make_torch_path_network
from sttw_control.path_command_reward import BASE_COST_KEYS


def episode():
    cfg = dict(network={'architecture': 'alpha_modulated_v1'},
        timing=dict(lower_dt_s=.005, upper_dt_s=.02), path_reward=dict(scale=2.))
    net = GoalNetworkBundle(make_torch_path_network(351, 2, cfg), cfg, [-1., -2.])
    obs, critic, ctx = torch.ones((3, 351)) * .1, torch.ones((3, 352)) * .1, torch.ones((3, 5)) * .1
    with torch.no_grad():
        mean, g, c, s = net(obs, critic, ctx); action = mean + .1
        logp = net.evaluate_actions(obs, critic, ctx, action)[0]
    valid = np.array([True, False, True])
    d = dict(valid=valid, task_tick=np.array([0, -1, 8]), observation=obs.numpy(),
        critic_observation=critic.numpy(), goal_context=ctx.numpy(), raw_action=action.numpy(),
        log_prob=logp.numpy(), old_goal_value=g.numpy(), old_cost_value=c.numpy(), physical_value=s.numpy())
    for name in d:
        if name not in ('valid', 'task_tick'): d[name][~valid] = np.nan
    xy = np.full((12, 2), 5.); xy[5] = 0.
    p = dict(active_tick=np.ones(12, bool), actual_xy=xy, physical_tick=np.arange(12),
        actor_action_valid=np.repeat(valid, 4), task_reward_valid=np.ones(12, bool),
        effective_costs=np.ones((12, len(BASE_COST_KEYS))), tick_reward=np.full(12, -.1),
        contact_failure=np.zeros(12, bool), nonfinite=np.zeros(12, bool),
        lower_fault=np.zeros(12, bool), policy_fault=np.zeros(12, bool), horizon_reached=np.arange(12) == 11)
    return net, cfg, dict(episode_id='ep0', collection_id='same', decisions=d, physical=p, original_goal_xy=[0., 0.])


def assemble(episodes, cfg, expected=1):
    return assemble_goal_population(episodes, expected_episode_count=expected, config=cfg,
        collection_id='same', learner_id='same', horizon_s=.06, radius_m=1.)


def test_complete_inventory_targets_flow_to_actual_network_loss():
    net, cfg, e = episode(); batch = assemble([e], cfg)
    assert batch['fixed_denominator'] == 3 and batch['cost_population_complete']
    assert batch['masks']['goal'].tolist() == [True, False, False]
    assert batch['masks']['survival'].tolist() == [True, False, True]
    result = goal_network_ppo_terms(net, batch['observation'], batch['critic_observation'],
        batch['context'], batch['actions'], masks=batch['masks'], collected=batch['collected'],
        objective_weights=dict(goal=1., cost=.1, survival=1.),
        fixed_denominator=batch['fixed_denominator'], cost_population_complete=batch['cost_population_complete'])
    result['actor_loss'].backward()
    assert torch.isfinite(net.goal_branch.weight.grad).all()


def test_missing_duplicate_and_other_policy_inventory_are_rejected():
    _, cfg, e = episode()
    with pytest.raises(ValueError, match='inventory'): assemble([e], cfg, expected=2)
    with pytest.raises(ValueError, match='unique'): assemble([e, e], cfg, expected=2)
    e['collection_id'] = 'other'
    with pytest.raises(ValueError, match='frozen policy'): assemble([e], cfg)


def test_incomplete_costs_retained_in_population_block_original_cost_update():
    net, cfg, e = episode(); other = copy.deepcopy(e); other['episode_id'] = 'ep1'
    other['physical']['task_reward_valid'][3] = False
    other['physical']['effective_costs'][3] = np.nan; other['physical']['tick_reward'][3] = np.nan
    batch = assemble([e, other], cfg, expected=2)
    assert len(batch['summaries']) == 2 and not batch['cost_population_complete']
    with pytest.raises(ValueError, match='full-population'):
        goal_network_ppo_terms(net, batch['observation'], batch['critic_observation'],
            batch['context'], batch['actions'], masks=batch['masks'], collected=batch['collected'],
            objective_weights=dict(goal=1., cost=.1, survival=1.),
            fixed_denominator=batch['fixed_denominator'], cost_population_complete=batch['cost_population_complete'])


def test_scaled_cost_targets_convert_raw_units_once_and_preserve_old_head_units():
    _, cfg, e = episode()
    e['decisions']['old_cost_value'][e['decisions']['valid']] = .3
    raw = assemble([e], cfg)
    scaled = assemble_goal_population([e], expected_episode_count=1, config=cfg,
        collection_id='same', learner_id='same', horizon_s=.06, radius_m=1., cost_target_scale=100.)
    mask = raw['masks']['cost']
    torch.testing.assert_close(scaled['collected']['cost_targets'][mask],
        raw['collected']['cost_targets'][mask] / 100.)
    torch.testing.assert_close(scaled['collected']['old_cost_values'][mask], torch.tensor([.3]))
    torch.testing.assert_close(scaled['raw_cost_targets'][mask], raw['collected']['cost_targets'][mask])
    assert scaled['cost_target_scale'] == 100.
    assert scaled['cost_population_complete'] == raw['cost_population_complete']
    for name in ('goal', 'survival'):
        torch.testing.assert_close(scaled['masks'][name], raw['masks'][name])
    for name in ('goal_targets', 'physical_targets'):
        m = raw['masks']['goal' if name == 'goal_targets' else 'survival']
        torch.testing.assert_close(scaled['collected'][name][m], raw['collected'][name][m])


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf'), True])
def test_invalid_cost_units_are_rejected(scale):
    _, cfg, e = episode()
    with pytest.raises(ValueError, match='cost target scale'):
        assemble_goal_population([e], expected_episode_count=1, config=cfg,
            collection_id='same', learner_id='same', horizon_s=.06, radius_m=1., cost_target_scale=scale)
