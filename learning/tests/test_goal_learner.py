"""Lifecycle bugs that could invalidate a goal-learning comparison; CPU only."""
import copy
from unittest.mock import patch
import numpy as np
import pytest
import torch
from test_goal_population import episode
from sttw_control.goal_learner import GoalLearner


def setup(epochs=2):
    net, cfg, e = episode(); e['case'] = dict(alpha=.1, speed=1.8, profile=0)
    learner = GoalLearner(net, objective_weights=dict(goal=1., cost=.1, survival=1.),
        epochs=epochs, actor_lr=1e-4, value_lr=3e-4, clip=.2, entropy_coef=0., grad_clip=1.)
    manifest = [dict(episode_id=e['episode_id'], original_goal_xy=e['original_goal_xy'], case=e['case'])]
    frozen = learner.freeze_collection(manifest, horizon_s=.06, radius_m=1.)
    e['collection_id'] = frozen['collection_id']
    return learner, e, frozen


def test_warmup_consumes_once_preserves_actor_branch_and_logstd_exactly():
    learner, e, frozen = setup()
    before = [x.detach().clone() for x in learner.actor_parameters]
    learner.prepare_population([e]); result = learner.update(warmup=True)
    assert result['optimizer_steps'] == [0, 2]
    for x, y in zip(learner.actor_parameters, before): torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match='unconsumed'): learner.update(warmup=False)
    assert learner.pending_status() is None
    again = learner.freeze_collection([dict(episode_id=e['episode_id'],
        original_goal_xy=e['original_goal_xy'], case=e['case'])], horizon_s=.06, radius_m=1.)
    assert again['collection_id'] != frozen['collection_id']
    with pytest.raises(ValueError, match='frozen policy'): learner.prepare_population([e])


def test_actual_update_changes_actor_and_holds_old_batch_despite_caller_mutation():
    learner, e, frozen = setup()
    receipt = learner.prepare_population([e])
    before = learner.bundle.goal_branch.weight.detach().clone()
    e['decisions']['raw_action'][:] = np.nan  # does not alias internally owned tensors
    receipt['summaries'][0]['goal'] = 'invented'
    result = learner.update(warmup=False)
    assert result['optimizer_steps'] == [2, 2]
    assert not torch.equal(before, learner.bundle.goal_branch.weight)


def test_exact_inventory_pairing_and_original_goal_cannot_change():
    learner, e, _ = setup()
    bad = copy.deepcopy(e); bad['episode_id'] = 'replacement'
    with pytest.raises(ValueError, match='inventory'): learner.prepare_population([bad])
    bad = copy.deepcopy(e); bad['case']['alpha'] = .9
    with pytest.raises(ValueError, match='original case'): learner.prepare_population([bad])
    bad = copy.deepcopy(e); bad['original_goal_xy'] = [2., 0.]
    with pytest.raises(ValueError, match='original case'): learner.prepare_population([bad])
    with pytest.raises(ValueError, match='unresolved'): learner.freeze_collection([], horizon_s=.06, radius_m=1.)


def test_parameter_or_settings_change_after_freeze_is_rejected():
    learner, e, _ = setup()
    with torch.no_grad(): learner.bundle.values.bias[0].add_(.1)
    with pytest.raises(ValueError, match='changed during'): learner.prepare_population([e])
    learner, e, _ = setup(); learner.prepare_population([e])
    learner.settings['objective_weights']['goal'] = 2.
    with pytest.raises(ValueError, match='changed before'): learner.update(warmup=False)


def test_pending_physics_remains_pending_and_cannot_update():
    learner, e, _ = setup()
    e['physical']['horizon_reached'][:] = False
    learner_before = learner.fingerprint()
    with pytest.raises(ValueError, match='pending'): learner.prepare_population([e])
    assert learner.pending_status()['phase'] == 'collecting'
    assert learner.pending_status()['population_diagnostics'][0]['survival'] == 'pending'
    with pytest.raises(ValueError, match='unconsumed'): learner.update(warmup=False)
    assert learner.fingerprint() == learner_before


def test_unknown_episode_is_preserved_and_stops_batch_without_recollection():
    learner, e, _ = setup()
    p = e['physical']; p['horizon_reached'][:] = False; p['policy_fault'][11] = True
    with pytest.raises(ValueError, match='unknown population'): learner.prepare_population([e])
    state = learner.pending_status()
    assert state['phase'] == 'error' and state['population_diagnostics'][0]['survival'] == 'unknown'
    assert len(state['manifest']) == 1 and learner.optimizer_steps == [0, 0]
    with pytest.raises(ValueError, match='fresh collecting'): learner.prepare_population([e])


def test_no_action_and_missing_cost_populations_cannot_silently_be_replaced():
    learner, e, _ = setup()
    e['decisions']['valid'][:] = False; e['decisions']['task_tick'][:] = -1
    e['physical']['actor_action_valid'][:] = False
    with pytest.raises(ValueError, match='no resolved'): learner.prepare_population([e])
    assert learner.pending_status()['phase'] == 'untrainable_no_actions'
    learner, e, _ = setup(); e['physical']['task_reward_valid'][3] = False
    with pytest.raises(ValueError, match='original-cost'): learner.prepare_population([e])
    assert learner.pending_status()['phase'] == 'rejected_missing_cost'


def test_fault_after_first_optimizer_step_preserves_error_and_forbids_retry():
    learner, e, _ = setup(); learner.prepare_population([e])
    with patch.object(learner.optimizers[1], 'step', side_effect=RuntimeError('synthetic optimizer fault')):
        with pytest.raises(RuntimeError, match='synthetic'): learner.update(warmup=False)
    state = learner.pending_status()
    assert state['phase'] == 'error' and state['optimizer_steps_after'] == [1, 0]
    assert 'partial' in state['step_outcome']
    with pytest.raises(ValueError, match='unconsumed'): learner.update(warmup=False)


def test_contact_is_a_known_failure_target_and_does_not_block_learning():
    learner, e, _ = setup(epochs=1)
    e['physical']['contact_failure'][11] = True
    receipt = learner.prepare_population([e])
    assert receipt['summaries'][0]['survival'] == 'contact'
    result = learner.update(warmup=False)
    assert result['optimizer_steps'] == [1, 1]


def test_survival_only_legal_actions_after_fallback_goal_allow_critic_warmup():
    learner, e, _ = setup(epochs=1)
    d = e['decisions']; d['valid'][:] = [False, False, True]; d['task_tick'][:] = [-1, -1, 8]
    for k in d:
        if k not in ('valid', 'task_tick'): d[k][:2] = np.nan
    e['physical']['actor_action_valid'] = np.repeat(d['valid'], 4)
    before = [p.detach().clone() for p in learner.actor_parameters]
    learner.prepare_population([e]); result = learner.update(warmup=True)
    assert result['optimizer_steps'] == [0, 1]
    for x, y in zip(learner.actor_parameters, before): torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_validation_failure_before_first_optimizer_call_is_distinguished():
    learner, e, _ = setup(); learner.prepare_population([e])
    with patch('sttw_control.goal_learner.goal_network_ppo_terms', side_effect=RuntimeError('pre-step fault')):
        with pytest.raises(RuntimeError, match='pre-step'): learner.update(warmup=False)
    assert learner.pending_status()['step_outcome'] == 'no_optimizer_step_started'
    assert learner.optimizer_steps == [0, 0]


@pytest.mark.parametrize('first_warmup', [True, False])
def test_completed_boundary_checkpoint_restores_adam_and_next_update_exactly(first_warmup):
    learner, e, _ = setup()
    learner.prepare_population([e]); learner.update(warmup=first_warmup)
    saved = learner.checkpoint(); restored = GoalLearner.restore(saved)
    assert restored.optimizer_steps == ([0, 2] if first_warmup else [2, 2]) and restored.collection_index == 1
    assert restored.fingerprint() == learner.fingerprint()
    old_units = learner.checkpoint(); del old_units['settings']['cost_target_scale']
    with pytest.raises(ValueError, match='explicit checkpoint C units'): GoalLearner.restore(old_units)
    # Fresh synthetic collection values from the new frozen heads; no old batch replay.
    d = e['decisions']; valid = d['valid']
    with torch.no_grad():
        views = [torch.tensor(d[k][valid]) for k in ('observation', 'critic_observation', 'goal_context')]
        raw = learner.bundle(*views)[0] + .1
        d['raw_action'][valid] = raw.numpy()
        logp, g, c, s = learner.bundle.evaluate_actions(*views, raw)
    for key, value in [('log_prob', logp), ('old_goal_value', g), ('old_cost_value', c), ('physical_value', s)]:
        d[key][valid] = value.numpy()
    for obj in (learner, restored):
        frozen = obj.freeze_collection([dict(episode_id=e['episode_id'],
            original_goal_xy=e['original_goal_xy'], case=e['case'])], horizon_s=.06, radius_m=1.)
        fresh = copy.deepcopy(e); fresh['collection_id'] = frozen['collection_id']
        obj.prepare_population([fresh]); obj.update(warmup=False)
    for a, b in zip(learner.bundle.parameters(), restored.bundle.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert learner.optimizer_steps == restored.optimizer_steps == ([2, 4] if first_warmup else [4, 4])


def test_pending_and_error_checkpoints_are_forensic_and_cannot_auto_resume():
    learner, _, _ = setup()
    checkpoint = learner.checkpoint()
    assert checkpoint['pending']['phase'] == 'collecting'
    with pytest.raises(ValueError, match='forensic'): GoalLearner.restore(checkpoint)
    checkpoint['pending']['manifest'][0]['episode_id'] = 'mutated-copy'
    assert learner.pending_status()['manifest'][0]['episode_id'] == 'ep0'


def test_aggregation_is_bound_and_v1_checkpoints_do_not_silently_change_algorithm():
    learner, e, _ = setup()
    assert learner.settings['actor_aggregation'] == 'scalarize_then_clip'
    learner.settings['actor_aggregation'] = 'componentwise_clip'
    with pytest.raises(ValueError, match='changed during'): learner.prepare_population([e])
    learner, e, _ = setup(); learner.prepare_population([e]); learner.update(warmup=True)
    saved = learner.checkpoint()
    assert saved['schema'] == 'sttw_goal_learner_v2'
    saved['schema'] = 'sttw_goal_learner_v1'
    with pytest.raises(ValueError, match='v2'): GoalLearner.restore(saved)


def test_explicit_cost_scale_is_frozen_and_restored_without_rescaling_recorded_head():
    net, cfg, e = episode(); e['case'] = dict(alpha=.1, speed=1.8, profile=0)
    learner = GoalLearner(net, objective_weights=dict(goal=1., cost=1., survival=1.),
        epochs=1, actor_lr=1e-4, value_lr=3e-4, clip=.2, entropy_coef=0., grad_clip=1.,
        cost_target_scale=100.)
    manifest = [dict(episode_id=e['episode_id'], original_goal_xy=e['original_goal_xy'], case=e['case'])]
    frozen = learner.freeze_collection(manifest, horizon_s=.06, radius_m=1.)
    assert frozen['cost_target_scale'] == 100.
    e['collection_id'] = frozen['collection_id']
    learner.settings['cost_target_scale'] = 1.
    with pytest.raises(ValueError, match='changed during'): learner.prepare_population([e])
    learner.settings['cost_target_scale'] = 100.
    learner.prepare_population([e]); learner.update(warmup=True)
    restored = GoalLearner.restore(learner.checkpoint())
    assert restored.settings['cost_target_scale'] == 100.
    assert restored.fingerprint() == learner.fingerprint()
    old_units = learner.checkpoint(); del old_units['settings']['cost_target_scale']
    with pytest.raises(ValueError, match='explicit checkpoint C units'): GoalLearner.restore(old_units)
