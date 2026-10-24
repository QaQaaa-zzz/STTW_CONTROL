"""Physical-tail credit must not end at the tracking-domain boundary."""
import numpy as np
import pytest

from sttw_control.survival_credit import resolve_survival_credit


def resolve(**changes):
    data = dict(active_tick=[True]*4, actor_action_valid=[True, True, False, False],
                contact_failure=[False]*4, engineering_fault=[False]*4,
                horizon_reached=[False, False, False, True],
                physical_tick=np.arange(4), window_start_tick=0, window_ticks=4,
                collection_policy_id='frozen_collection_A', learner_policy_id='frozen_collection_A')
    data.update(changes)
    return resolve_survival_credit(**data)


def test_survival_after_actor_handover_reaches_preceding_actions_only():
    credit = resolve()
    assert credit['outcome'] == 'survived'
    np.testing.assert_array_equal(credit['actor_mask'], [True, True, False, False])
    np.testing.assert_array_equal(credit['actor_indices'], [0, 1])
    np.testing.assert_array_equal(credit['targets'][:2], [1., 1.])
    assert np.isnan(credit['targets'][2:]).all()
    assert credit['physical_ticks'] == 4 and credit['actor_actions'] == 2


def test_contact_in_hold_tail_is_failure_for_preceding_actions():
    credit = resolve(contact_failure=[False, False, False, True])
    assert credit['outcome'] == 'contact'
    np.testing.assert_array_equal(credit['targets'][:2], [0., 0.])
    assert credit['actor_actions'] == 2


@pytest.mark.parametrize('events,outcome', [
    (dict(engineering_fault=[False, False, False, True]), 'unknown'),
    (dict(horizon_reached=[False]*4), 'pending'),
    (dict(contact_failure=[False, False, False, True],
          engineering_fault=[False, False, False, True]), 'unknown'),
])
def test_unresolved_or_faulted_tail_never_generates_learning_labels(events, outcome):
    credit = resolve(**events)
    assert credit['outcome'] == outcome
    assert not credit['actor_mask'].any() and credit['actor_actions'] == 0
    assert np.isnan(credit['targets']).all()


def test_policy_update_between_collection_and_credit_is_rejected():
    with pytest.raises(ValueError, match='policy'):
        resolve(learner_policy_id='updated_policy_B')


def test_padding_cannot_contain_actor_actions_or_terminal_events():
    with pytest.raises(ValueError):
        resolve(active_tick=[True, False, True, True])
    with pytest.raises(ValueError):
        resolve(active_tick=[True, True, True, False])


def test_first_terminal_closes_episode_without_credit_to_reset_actions():
    credit = resolve(contact_failure=[False, True, False, False],
                     actor_action_valid=[True, True, True, True])
    assert credit['outcome'] == 'contact'
    np.testing.assert_array_equal(credit['actor_mask'], [True, True, False, False])
    assert credit['physical_ticks'] == 2


def test_surviving_episode_with_no_legal_actor_actions_has_no_actor_samples():
    credit = resolve(actor_action_valid=[False]*4)
    assert credit['outcome'] == 'survived' and credit['actor_actions'] == 0


def test_fractional_masks_are_not_silently_treated_as_boolean_events():
    with pytest.raises(ValueError):
        resolve(active_tick=[1., .5, 1., 1.])


def test_missing_observed_physical_tick_cannot_be_called_survival():
    with pytest.raises(ValueError, match='coverage'):
        resolve(active_tick=[True, False, True, True],
                actor_action_valid=[True, False, False, False])


def test_logical_padding_is_allowed_when_physical_clock_has_no_gap():
    credit = resolve(active_tick=[True, False, True, True, True],
        actor_action_valid=[True, False, True, False, False],
        contact_failure=[False]*5, engineering_fault=[False]*5,
        horizon_reached=[False]*4+[True], physical_tick=np.array([0, 0, 1, 2, 3]))
    assert credit['outcome'] == 'survived' and credit['actor_actions'] == 2


def test_premature_horizon_or_wrong_window_origin_cannot_supply_positive_label():
    with pytest.raises(ValueError):
        resolve(window_ticks=8)
    with pytest.raises(ValueError):
        resolve(window_start_tick=1)


def test_survival_loss_updates_opposite_actions_and_excludes_nan_hold_rows():
    import torch
    from sttw_control.survival_credit import survival_ppo_loss
    logp = torch.tensor([0., 0., float('nan')], requires_grad=True)
    old_logp = torch.tensor([0., 0., float('nan')], requires_grad=True)
    values = torch.tensor([.5, .5, float('nan')], requires_grad=True)
    old_values = torch.tensor([.5, .5, float('nan')], requires_grad=True)
    targets = torch.tensor([1., 0., float('nan')], requires_grad=True)
    result = survival_ppo_loss(logp, old_logp, values, targets,
                               torch.tensor([True, True, False]), old_values=old_values)
    (result['actor_loss']+result['value_loss']).backward()
    assert result['samples'] == 2
    np.testing.assert_allclose(logp.grad, [-.25, .25, 0.])
    np.testing.assert_allclose(values.grad, [-.5, .5, 0.])
    assert old_logp.grad is None and old_values.grad is None and targets.grad is None


def test_clipping_uses_frozen_collection_value_not_current_critic():
    import torch
    from sttw_control.survival_credit import survival_ppo_loss
    logp = torch.tensor([np.log(2.)], requires_grad=True)
    result = survival_ppo_loss(logp, torch.tensor([0.]), torch.tensor([.99]),
                               torch.tensor([1.]), torch.tensor([True]),
                               old_values=torch.tensor([.5]))
    result['actor_loss'].backward()
    assert result['actor_loss'].item() == pytest.approx(-.6)
    assert logp.grad.item() == pytest.approx(0.)


@pytest.mark.parametrize('target,mask', [(float('nan'), True), (1., False)])
def test_unresolved_target_or_empty_minibatch_cannot_enter_optimizer(target, mask):
    import torch
    from sttw_control.survival_credit import survival_ppo_loss
    with pytest.raises(ValueError):
        survival_ppo_loss(torch.tensor([0.]), torch.tensor([0.]), torch.tensor([.5]),
                          torch.tensor([target]), torch.tensor([mask]), old_values=torch.tensor([.5]))


def test_invalid_hold_observations_are_removed_before_real_network_forward():
    import torch
    from sttw_control.survival_credit import survival_ppo_batch
    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.zero_()
        critic.weight.zero_()
    observations = torch.tensor([[1.], [float('nan')]], requires_grad=True)
    actions = torch.tensor([[1.], [float('nan')]], requires_grad=True)
    def evaluate(obs, action):
        distribution = torch.distributions.Normal(actor(obs), torch.ones_like(action))
        return distribution.log_prob(action).sum(-1), critic(obs).sigmoid().squeeze(-1)
    result = survival_ppo_batch(evaluate, observations, actions,
        torch.tensor([-.5-.5*np.log(2*np.pi), float('nan')]),
        torch.tensor([.5, float('nan')]), torch.tensor([1., float('nan')]),
        torch.tensor([True, False]))
    (result['actor_loss']+result['value_loss']).backward()
    assert actor.weight.grad.item() == pytest.approx(-.5)
    assert critic.weight.grad.item() == pytest.approx(-.25)
    assert observations.grad is None and actions.grad is None
