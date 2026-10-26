import numpy as np
import pytest

from sttw_control.preference_phase_audit import phase_metrics, first_hit_count, endpoint_direction


def test_first_arrival_includes_crossing_transition_without_interpolation():
    assert first_hit_count([False, True, True]) == 2
    assert first_hit_count([False, False]) is None
    metric = phase_metrics([2., 0., 999.], [0., 2., 999.], [3., 4., 999.], 2, .005)
    assert metric['duration_s'] == .01
    assert metric['original_alpha_cost'] == 7
    assert metric['mean_abs_path_error_m'] == 1


def test_different_durations_need_mean_not_only_error_integral():
    a = phase_metrics(np.ones(10), np.ones(10), np.ones(10), 2, 1.)
    b = phase_metrics(np.full(10, .5), np.full(10, 2.), np.ones(10), 10, 1.)
    assert b['speed_abs_error_integral_m'] > a['speed_abs_error_integral_m']
    assert endpoint_direction(a,b)['both_expected_directions']


def test_invalid_or_nonfinite_prefix_is_not_missing_or_zero_error():
    for n in [0, 3, 1.5]:
        with pytest.raises(ValueError):phase_metrics([1., 1.], [1., 1.], [1., 1.], n, .005)
    with pytest.raises(ValueError):phase_metrics([np.nan], [1.], [1.], 1, .005)
