"""Emergency survival must not be relabeled by a recoverable task exit."""
import numpy as np
from sttw_control.failure_contract import split_failure_events


def events(**changes):
    flags = dict(contact_failure=False, engineering_fault=False,
                 tracking_domain_exit=False, legacy_roll_crossed=False, time_limit=False)
    return split_failure_events(**dict(flags, **changes))


def test_path_exit_closes_task_but_preserves_physical_continuation():
    result = events(tracking_domain_exit=True)
    assert result['task_done'] and not result['physical_done']
    assert result['risk_bootstrap'] and not result['task_bootstrap']
    assert not result['fell'] and not result['survived_window']


def test_roll_validity_crossing_is_telemetry_until_real_contact():
    result = events(legacy_roll_crossed=True)
    assert result['legacy_roll_crossed'] and not result['physical_done']
    assert not result['fell']


def test_contact_and_engineering_failure_are_different_outcomes():
    contact = events(contact_failure=True)
    fault = events(engineering_fault=True, contact_failure=True)
    assert contact['physical_done'] and contact['fell'] and not contact['unknown']
    assert fault['physical_done'] and fault['unknown'] and not fault['fell']


def test_fixed_observation_end_has_no_beyond_window_bootstrap():
    result = events(time_limit=True, tracking_domain_exit=True, legacy_roll_crossed=True)
    assert result['physical_done'] and result['survived_window']
    assert not result['risk_bootstrap'] and not result['task_bootstrap']


def test_mixed_batch_keeps_each_lane_outcome_and_bootstrap():
    result = events(tracking_domain_exit=np.array([True,False,False,False]),
                    contact_failure=np.array([False,True,False,False]),
                    engineering_fault=np.array([False,False,True,False]),
                    time_limit=np.array([False,False,False,True]))
    np.testing.assert_array_equal(result['risk_bootstrap'], [True,False,False,False])
    np.testing.assert_array_equal(result['fell'], [False,True,False,False])
    np.testing.assert_array_equal(result['unknown'], [False,False,True,False])
    np.testing.assert_array_equal(result['survived_window'], [False,False,False,True])
