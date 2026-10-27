"""Actual live collector field/clock semantics, with synthetic physical records."""
import numpy as np
import pytest
from sttw_control.goal_live_credit import resolve_live_goal_credit
from sttw_control.path_command_reward import BASE_COST_KEYS


def fixture():
    d = dict(valid=np.array([True, False, True]), task_tick=np.array([0, -1, 8]))
    active = np.ones(12, bool)
    xy = np.full((12, 2), 5.); xy[5] = [0., 0.]
    components = np.ones((12, len(BASE_COST_KEYS)))
    p = dict(active_tick=active, actual_xy=xy, physical_tick=np.arange(12),
        actor_action_valid=np.repeat(d['valid'], 4), task_reward_valid=active.copy(),
        effective_costs=components, tick_reward=np.full(12, -.1),
        scored_tick_reward=np.full(12, -999.), contact_failure=np.zeros(12, bool),
        nonfinite=np.zeros(12, bool), lower_fault=np.zeros(12, bool),
        policy_fault=np.zeros(12, bool), horizon_reached=np.arange(12) == 11)
    cfg = dict(timing=dict(lower_dt_s=.005, upper_dt_s=.02),
               path_reward=dict(scale=2.))
    return d, p, cfg


def resolve(d, p, cfg):
    return resolve_live_goal_credit(d, p, goal_xy=[0., 0.], radius_m=1., config=cfg,
        collection_id='same', learner_id='same', horizon_s=.06)


def test_fallback_goal_and_running_components_reach_actual_upper_decisions():
    d, p, cfg = fixture(); r = resolve(d, p, cfg)
    np.testing.assert_equal(r['goal_mask'], [True, False, False])
    np.testing.assert_equal(r['survival_mask'], [True, False, True])
    assert r['goal_targets'][0] == 1.
    assert r['cost_targets'][0] == pytest.approx(.6)  # includes fallback, once-integrated
    assert r['credit']['cost_window_complete']


def test_scored_failure_replacement_cannot_change_running_cost_credit():
    d, p, cfg = fixture(); r = resolve(d, p, cfg)
    p['scored_tick_reward'][:] = np.nan
    rr = resolve(d, p, cfg)
    np.testing.assert_equal(r['cost_targets'], rr['cost_targets'])
    p['tick_reward'] = np.full(12, -999.)
    with pytest.raises(ValueError, match='integrated effective'):
        resolve(d, p, cfg)


def test_partial_costs_preserve_goal_but_reject_full_coverage_claim():
    d, p, cfg = fixture()
    p['task_reward_valid'][3] = False
    p['effective_costs'][3] = np.nan; p['tick_reward'][3] = np.nan
    r = resolve(d, p, cfg)
    assert r['goal_targets'][0] == 1. and not r['cost_mask'].any()
    assert not r['credit']['cost_window_complete']


def test_repeated_permission_is_not_four_actor_decisions():
    d, p, cfg = fixture()
    r = resolve(d, p, cfg)
    assert r['credit']['physical']['observed_actor_actions'] == 2
    p['actor_action_valid'][2] = False
    with pytest.raises(ValueError, match='permission'):
        resolve(d, p, cfg)


def test_clock_and_effective_channel_identity_are_checked():
    d, p, cfg = fixture(); d['task_tick'][2] = 4
    with pytest.raises(ValueError, match='clock'):
        resolve(d, p, cfg)
    d, p, cfg = fixture(); p['effective_costs'] = np.ones((12, 9))
    with pytest.raises(ValueError, match='channels'):
        resolve(d, p, cfg)
