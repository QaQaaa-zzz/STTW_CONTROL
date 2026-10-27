"""Synthetic CPU credit-to-Actor checks, not robot performance experiments."""
import numpy as np
import pytest
import torch

from sttw_control.goal_task_credit import (resolve_goal_task_credit,
    goal_task_ppo_loss, goal_task_ppo_batch, goal_cost_population_complete)


def packet(**changes):
    data = dict(active_tick=np.ones(6, bool),
        actor_action_valid=np.array([True, False, True, False, True, False]),
        contact_failure=np.zeros(6, bool), engineering_fault=np.zeros(6, bool),
        horizon_reached=np.array([False]*5+[True]), physical_tick=np.arange(6),
        window_start_tick=0, window_ticks=6, collection_policy_id='unchanged', learner_policy_id='unchanged',
        actual_xy=np.full((6, 2), 5.), goal_xy=[1., 0.], radius_m=1.,
        running_tick_cost=np.arange(1., 7.), task_cost_valid=np.ones(6, bool))
    data.update(changes)
    return data


def arrived(**changes):
    xy = np.full((6, 2), 5.); xy[3] = [1., 0.]
    return resolve_goal_task_credit(**packet(actual_xy=xy, **changes))


def torch_packet(result):
    masks = dict(goal=torch.tensor(result['goal_actor_mask']),
        cost=torch.tensor(result['cost_actor_mask']), survival=torch.tensor(result['physical']['actor_mask']))
    data = dict(old_log_prob=torch.zeros(6), old_goal_values=torch.full((6,), .5),
        goal_targets=torch.tensor(result['goal_targets']), old_cost_values=torch.zeros(6),
        cost_targets=torch.tensor(result['cost_targets']), old_physical_values=torch.full((6,), .5),
        physical_targets=torch.tensor(result['physical']['targets']))
    return masks, data


def direct_loss(result, weights):
    masks, data = torch_packet(result)
    logp = torch.zeros(6, requires_grad=True)
    loss = goal_task_ppo_loss(new_log_prob=logp, goal_values=torch.full((6,), .5),
        cost_values=torch.zeros(6), physical_values=torch.full((6,), .5),
        goal_mask=masks['goal'], cost_mask=masks['cost'], physical_mask=masks['survival'],
        **data, objective_weights=weights, fixed_denominator=6,
        cost_population_complete=goal_cost_population_complete([result]))
    return logp, loss


def test_held_goal_reaches_preceding_legal_actions_but_not_later_actions():
    r = arrived()
    np.testing.assert_equal(r['goal_actor_mask'], [True, False, True, False, False, False])
    np.testing.assert_equal(r['physical']['actor_mask'], [True, False, True, False, True, False])
    np.testing.assert_equal(r['goal_targets'][[0, 2]], [1., 1.])
    np.testing.assert_equal(r['cost_targets'][[0, 2]], [10., 7.])
    assert r['goal_entry_index'] == 3 and r['gamma_goal'] == r['gamma_cost'] == 1.


def test_goal_and_survival_have_different_credit_after_arrival():
    r = arrived()
    x, loss = direct_loss(r, dict(goal=1., cost=.1, survival=1.))
    g = torch.autograd.grad(loss['actor_terms']['goal'], x, retain_graph=True)[0]
    s = torch.autograd.grad(loss['actor_terms']['survival'], x, retain_graph=True)[0]
    c = torch.autograd.grad(loss['actor_terms']['cost'], x)[0]
    assert g[4] == c[4] == 0 and s[4] != 0
    assert c[0] > 0  # Gradient descent reduces log likelihood of costly actions.


def test_changing_only_factual_goal_entry_changes_actor_loss_and_gradient():
    yes, no = arrived(), resolve_goal_task_credit(**packet())
    assert yes['physical']['target'] == no['physical']['target'] == 1.
    x, a = direct_loss(yes, dict(goal=1., cost=0., survival=0.))
    y, b = direct_loss(no, dict(goal=1., cost=0., survival=0.))
    ga = torch.autograd.grad(a['actor_loss'], x)[0]
    gb = torch.autograd.grad(b['actor_loss'], y)[0]
    assert a['actor_loss'].item() != b['actor_loss'].item()
    assert ga[0] < 0 < gb[0]


def test_goal_then_contact_or_fault_preserves_only_known_objectives():
    contact = np.zeros(6, bool); contact[5] = True
    r = arrived(contact_failure=contact)
    assert r['goal_target'] == 1. and r['physical']['target'] == 0.
    fault = np.zeros(6, bool); fault[5] = True
    r = arrived(engineering_fault=fault)
    assert r['goal_target'] == 1. and r['physical']['outcome'] == 'unknown'
    assert not r['physical']['actor_mask'].any()


def test_same_tick_contact_prevents_goal_but_horizon_sample_may_reach_goal():
    xy = np.full((6, 2), 5.); xy[5] = [1., 0.]
    r = resolve_goal_task_credit(**packet(actual_xy=xy))
    assert r['goal_target'] == 1. and r['physical']['target'] == 1.
    contact = np.zeros(6, bool); contact[5] = True
    r = resolve_goal_task_credit(**packet(actual_xy=xy, contact_failure=contact))
    assert r['goal_target'] == r['physical']['target'] == 0.


def test_unknown_partial_and_inactive_padding_do_not_become_failed_goals():
    active = np.array([True, True, True, False, False, False])
    r = resolve_goal_task_credit(**packet(active_tick=active,
        actor_action_valid=np.array([True, False, True, False, False, False]), horizon_reached=np.zeros(6, bool)))
    assert np.isnan(r['goal_target']) and not r['goal_actor_mask'].any()
    assert not r['cost_actor_mask'].any()
    with pytest.raises(ValueError, match='policy'):
        resolve_goal_task_credit(**packet(learner_policy_id='updated'))


def test_known_early_goal_cannot_update_policy_before_physical_episode_resolves():
    active = np.array([True, True, True, True, False, False])
    r = arrived(active_tick=active,
        actor_action_valid=np.array([True, False, True, False, False, False]),
        horizon_reached=np.zeros(6, bool))
    assert r['goal_target'] == 1. and not r['physical_episode_resolved']
    assert not r['goal_actor_mask'].any() and not r['cost_actor_mask'].any()
    assert not r['physical']['actor_mask'].any()


def test_missing_task_cost_does_not_erase_goal_or_invent_cost_targets():
    valid = np.ones(6, bool); valid[1] = False
    costs = np.arange(1., 7.); costs[1] = np.nan
    r = arrived(task_cost_valid=valid, running_tick_cost=costs)
    assert r['goal_target'] == 1.
    assert not r['cost_actor_mask'][0] and r['cost_actor_mask'][2]
    assert r['cost_targets'][2] == 7.


def test_no_legal_action_still_has_outcome_but_never_an_actor_update():
    r = arrived(actor_action_valid=np.zeros(6, bool))
    assert r['goal_target'] == r['physical']['target'] == 1.
    with pytest.raises(ValueError, match='no resolved'):
        direct_loss(r, dict(goal=1., cost=.1, survival=1.))


def test_only_selected_finite_rows_reach_actual_actor_forward():
    r = arrived(); masks, data = torch_packet(r)
    union = masks['goal'] | masks['cost'] | masks['survival']
    obs = torch.full((6, 2), torch.nan); obs[union] = 1.
    actions = torch.full((6, 2), torch.nan); actions[union] = torch.tensor([1., -.5])
    actor = torch.nn.Linear(2, 2, bias=False); torch.nn.init.zeros_(actor.weight)
    mean = actor(obs[union]).detach()
    data['old_log_prob'][union] = torch.distributions.Normal(mean, 1.).log_prob(actions[union]).sum(-1)
    calls = []
    def evaluate(o, a):
        assert torch.isfinite(o).all() and torch.isfinite(a).all()
        calls.append(len(o)); mean = actor(o)
        return (torch.distributions.Normal(mean, 1.).log_prob(a).sum(-1),
                torch.full((len(o),), .5), torch.zeros(len(o)), torch.full((len(o),), .5))
    loss = goal_task_ppo_batch(evaluate, obs, actions, masks=masks, collected=data,
        objective_weights=dict(goal=1., cost=.1, survival=1.), fixed_denominator=6,
        cost_population_complete=goal_cost_population_complete([r]))
    loss['actor_loss'].backward()
    assert calls == [3] and torch.isfinite(actor.weight.grad).all()
    assert actor.weight.grad.abs().sum() > 0


def test_actual_actor_gradient_changes_with_goal_fact_at_fixed_actions_and_survival():
    actor = torch.nn.Linear(2, 2, bias=False); torch.nn.init.zeros_(actor.weight)
    gradients = []
    for result in (arrived(), resolve_goal_task_credit(**packet())):
        masks, data = torch_packet(result)
        union = masks['goal'] | masks['cost'] | masks['survival']
        obs = torch.full((6, 2), torch.nan); obs[union] = 1.
        actions = torch.full((6, 2), torch.nan); actions[union] = torch.tensor([1., -.5])
        data['old_log_prob'][union] = torch.distributions.Normal(
            torch.zeros((int(union.sum()), 2)), 1.).log_prob(actions[union]).sum(-1)
        def evaluate(o, a):
            mean = actor(o)
            return (torch.distributions.Normal(mean, 1.).log_prob(a).sum(-1),
                    torch.full((len(o),), .5), torch.zeros(len(o)), torch.full((len(o),), .5))
        actor.zero_grad()
        loss = goal_task_ppo_batch(evaluate, obs, actions, masks=masks, collected=data,
            objective_weights=dict(goal=1., cost=0., survival=0.), fixed_denominator=6,
            cost_population_complete=goal_cost_population_complete([result]))
        loss['actor_loss'].backward(); gradients.append(actor.weight.grad.clone())
    assert gradients[0][0, 0] < 0 < gradients[1][0, 0]


def test_declared_denominator_and_explicit_weights_are_required():
    r = arrived()
    with pytest.raises(ValueError, match='weights'):
        direct_loss(r, dict(goal=1., survival=1.))
    masks, data = torch_packet(r)
    with pytest.raises(ValueError, match='denominator'):
        goal_task_ppo_loss(new_log_prob=torch.zeros(6), goal_values=torch.full((6,), .5),
            cost_values=torch.zeros(6), physical_values=torch.full((6,), .5),
            goal_mask=masks['goal'], cost_mask=masks['cost'], physical_mask=masks['survival'],
            **data, objective_weights=dict(goal=1., cost=.1, survival=1.), fixed_denominator=1,
            cost_population_complete=True)


def test_partial_cost_suffix_cannot_silently_update_original_cost_objective():
    valid = np.ones(6, bool); valid[1] = False
    costs = np.arange(1., 7.); costs[1] = np.nan
    partial = arrived(task_cost_valid=valid, running_tick_cost=costs)
    assert partial['cost_actor_mask'][2]  # A local factual suffix is known.
    assert not goal_cost_population_complete([arrived(), partial])
    with pytest.raises(ValueError, match='full-population'):
        direct_loss(partial, dict(goal=1., cost=.1, survival=1.))
    # Known G/S may still be checked without claiming an original-C update.
    x, loss = direct_loss(partial, dict(goal=1., cost=0., survival=1.))
    assert torch.isfinite(torch.autograd.grad(loss['actor_loss'], x)[0]).all()


def test_partial_cost_population_is_rejected_before_network_forward():
    r = arrived(); masks, data = torch_packet(r)
    calls = []
    with pytest.raises(ValueError, match='full-population'):
        goal_task_ppo_batch(lambda *args: calls.append(True), torch.ones((6, 2)), torch.ones((6, 2)),
            masks=masks, collected=data, objective_weights=dict(goal=1., cost=.1, survival=1.),
            fixed_denominator=6, cost_population_complete=False)
    assert not calls


def clipping_packet(ratios=(1.5,), *, aggregation='scalarize_then_clip', conflict=True):
    n = len(ratios)
    new = torch.tensor(ratios).log().requires_grad_()
    mask = torch.ones(n, dtype=torch.bool); empty = torch.zeros(n, dtype=torch.bool)
    result = goal_task_ppo_loss(new_log_prob=new, old_log_prob=torch.zeros(n),
        goal_values=torch.full((n,), .5), old_goal_values=torch.full((n,), .5),
        goal_targets=torch.ones(n), goal_mask=mask,
        cost_values=torch.zeros(n), old_cost_values=torch.zeros(n),
        cost_targets=torch.full((n,), torch.nan), cost_mask=empty,
        physical_values=torch.full((n,), .5), old_physical_values=torch.full((n,), .5),
        physical_targets=torch.zeros(n) if conflict else torch.ones(n), physical_mask=mask,
        objective_weights=dict(goal=1., cost=0., survival=1.), fixed_denominator=n,
        cost_population_complete=True, actor_aggregation=aggregation)
    return new, result


@pytest.mark.parametrize('ratio', [.5, 1.5])
def test_scalarize_before_clip_cancels_opposing_goal_and_survival_advantages(ratio):
    logp, standard = clipping_packet((ratio,))
    oldlogp, historical = clipping_packet((ratio,), aggregation='componentwise_clip')
    assert standard['actor_aggregation'] == 'scalarize_then_clip'
    assert standard['actor_loss'].item() == pytest.approx(0.)
    assert historical['actor_loss'].item() == pytest.approx(.15, abs=1e-7)
    assert torch.autograd.grad(standard['actor_loss'], logp)[0].item() == pytest.approx(0.)
    assert abs(torch.autograd.grad(historical['actor_loss'], oldlogp)[0].item()) > 0


def test_same_sign_advantages_have_same_loss_and_gradient_under_both_modes():
    a, standard = clipping_packet((.5, 1., 1.5), conflict=False)
    b, historical = clipping_packet((.5, 1., 1.5), aggregation='componentwise_clip', conflict=False)
    torch.testing.assert_close(standard['actor_loss'], historical['actor_loss'])
    torch.testing.assert_close(torch.autograd.grad(standard['actor_loss'], a)[0],
                               torch.autograd.grad(historical['actor_loss'], b)[0])


def test_scalarization_respects_distinct_goal_cost_survival_masks_and_nan_padding():
    logp = torch.tensor([np.log(1.5), np.log(.5), np.nan], requires_grad=True)
    first = torch.tensor([True, False, False]); physical = torch.tensor([True, True, False])
    x = torch.tensor([.5, float('nan'), float('nan')], requires_grad=True)
    target = torch.tensor([1., float('nan'), float('nan')], requires_grad=True)
    result = goal_task_ppo_loss(new_log_prob=logp, old_log_prob=torch.tensor([0., 0., np.nan]),
        goal_values=torch.tensor([.5, .5, np.nan]), old_goal_values=x, goal_targets=target, goal_mask=first,
        cost_values=torch.zeros(3), old_cost_values=torch.zeros(3), cost_targets=torch.tensor([2., np.nan, np.nan]), cost_mask=first,
        physical_values=torch.tensor([.5, .5, np.nan]), old_physical_values=torch.tensor([.5, .5, np.nan]),
        physical_targets=torch.tensor([1., 0., np.nan]), physical_mask=physical,
        objective_weights=dict(goal=1., cost=.1, survival=1.), fixed_denominator=3, cost_population_complete=True)
    torch.testing.assert_close(result['scalarized_advantage'], torch.tensor([.8, -.5], dtype=logp.dtype))
    assert result['actor_loss'].item() == pytest.approx(-(.96 - .4)/3, abs=1e-7)
    result['actor_loss'].backward()
    assert torch.isfinite(logp.grad).all() and logp.grad[2] == 0
    assert x.grad is None and target.grad is None


def test_unsupported_aggregation_is_rejected_before_network_forward():
    r = arrived(); masks, data = torch_packet(r); calls = []
    with pytest.raises(ValueError, match='aggregation'):
        goal_task_ppo_batch(lambda *args: calls.append(True), torch.ones((6, 2)), torch.ones((6, 2)),
            masks=masks, collected=data, objective_weights=dict(goal=1., cost=.1, survival=1.),
            fixed_denominator=6, cost_population_complete=True, actor_aggregation='unknown')
    assert calls == []
