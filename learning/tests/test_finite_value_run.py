"""Reject mismatched clocks/return estimators before any simulation."""
import pytest
from sttw_control.finite_value_run import validate_contract


def contract():
    cfg = {'training_future_phase_C': {'gamma': .997, 'gae_lambda': .99, 'episode_s': 10.},
           'timing': {'lower_dt_s': .005, 'upper_dt_s': .02}}
    plan = dict(gamma=.997, horizon_s=10., lower_dt_s=.005, upper_dt_s=.02,
                updates=64, episodes_per_arm=60, total_envs=180,
                maximum_allocated_lower_slots=23040000,
                arms={'free_GAE': 0, 'finite_GAE': 0, 'free_MC': 0},
                gae_lambda_by_arm={'free_GAE': .99, 'finite_GAE': .99, 'free_MC': 1.})
    return cfg, plan


def test_declared_budget_and_return_estimators_match_actual_180_lane_clock():
    cfg, plan = contract()
    validate_contract(cfg, plan)


@pytest.mark.parametrize('key,value', [('gamma', .99), ('horizon_s', 8.),
    ('lower_dt_s', .01), ('upper_dt_s', .04), ('total_envs', 120),
    ('maximum_allocated_lower_slots', 15360000),
    ('gae_lambda_by_arm', {'free_GAE': .99, 'finite_GAE': .99, 'free_MC': .99}),
    ('arms', {'free_GAE': 0, 'finite_GAE': 7, 'free_MC': 0})])
def test_incompatible_value_clock_or_unmatched_control_permission_rejected(key, value):
    cfg, plan = contract()
    plan[key] = value
    with pytest.raises(ValueError):
        validate_contract(cfg, plan)


def test_environment_discount_mismatch_rejected():
    cfg, plan = contract()
    cfg['training_future_phase_C']['gamma'] = .99
    with pytest.raises(ValueError):
        validate_contract(cfg, plan)


@pytest.mark.parametrize('section,key,value', [('timing','lower_dt_s',.01),
    ('timing','upper_dt_s',.04), ('training_future_phase_C','episode_s',8.)])
def test_actual_config_clock_cannot_drift_from_value_horizon(section,key,value):
    cfg, plan = contract()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_contract(cfg, plan)


def test_value_diagnostic_uses_full_real_rewards_through_fallback():
    import numpy as np
    from sttw_control.finite_value_run import collection_value_diagnostic
    decision = dict(valid=np.array([[True], [False], [True]]),
                    old_task_value=np.array([[-2.], [np.nan], [-1.]]),
                    task_tick=np.array([[1988], [-1], [1996]]))
    # Three upper rewards -1,-2,-3, with no legal Actor in middle.
    physical = dict(active_tick=np.ones((12, 1), dtype=bool),
                    scored_tick_reward=np.repeat([-1., -2., -3.], 4).reshape(-1, 1) / 4)
    result = collection_value_diagnostic(decision, physical, [0], gamma=.5)
    # Full MC at first decision is -2.75; final is -3.
    assert result['all']['count'] == 2
    assert result['all']['mc_rmse'] == pytest.approx(((.75**2 + 2**2) / 2)**.5)
    assert result['last20ms']['mc_rmse'] == 2.
    assert result['last20ms']['count'] == 1
