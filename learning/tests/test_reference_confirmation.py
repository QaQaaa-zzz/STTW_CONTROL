import pytest
from sttw_control.reference_confirmation import confirmation_cases
from sttw_control.reference_confirmation import capture_stop_reason, capture_result_action
from sttw_control.reference_confirmation import validate_path_evidence


CONTEXTS = [dict(route='left', speed=1.8, alpha=0., lookup='steer_minus', target_delta=[0., -.04]),
            dict(route='right', speed=1.5, alpha=0., lookup='speed_plus', target_delta=[.15, 0.]),
            dict(route='right', speed=1.8, alpha=1., lookup='continue_original', target_delta=[0., 0.])]


def prefixes():
    return [dict(route=c['route'], speed=c['speed'], alpha=c['alpha'],
                 state_id=str(i), snapshot=f'/state{i}.pkl', snapshot_sha256=str(i))
            for i, c in enumerate(CONTEXTS)]


def test_frozen_lookup_has_fifteen_real_cases_and_three_exact_aliases():
    rows, pairs = confirmation_cases(prefixes(), CONTEXTS, [0., -32., 32.])
    assert len(rows) == 15 and len(pairs) == 9
    assert sum(p['baseline_index'] == p['lookup_index'] for p in pairs) == 3
    for p in pairs:
        base, lookup = rows[p['baseline_index']], rows[p['lookup_index']]
        assert (base['state_id'], base['force_N']) == (lookup['state_id'], lookup['force_N'])
        assert base['candidate'] == 'continue_original'
        if lookup['route'] == 'left':
            assert lookup['candidate'] == 'steer_minus'
            assert lookup['target_delta'] == [0., -.04]


def test_unavailable_capture_is_not_replaced_or_silently_dropped_from_pairs():
    rows, pairs = confirmation_cases(prefixes()[1:], CONTEXTS, [0., -32., 32.])
    assert len(rows) == 9 and len(pairs) == 9
    missing = [p for p in pairs if p['baseline_index'] is None]
    assert len(missing) == 3
    assert all(p['lookup_index'] is None and p['availability'] == 'UNAVAILABLE' for p in missing)


def test_duplicate_capture_identity_is_rejected_before_simulation():
    with pytest.raises(ValueError, match='duplicate'):
        confirmation_cases(prefixes() + prefixes()[:1], CONTEXTS, [0., -32., 32.])


def test_engineering_fault_stops_campaign_even_with_domain_or_physical_failure():
    reason = capture_stop_reason([True], [True], [True])
    assert reason == 'ENGINEERING_FAULT'
    with pytest.raises(RuntimeError, match='engineering'):
        capture_result_action(dict(phase='error', capture_stop_reason=reason,
                                   error='Nominal prefix invalid before declared geometric phases'), 1)
    with pytest.raises(RuntimeError, match='engineering'):
        capture_result_action(dict(phase='error', error='Nominal prefix invalid before declared geometric phases'), 1)


def test_only_structured_task_or_physical_termination_is_unavailable():
    assert capture_stop_reason([False], [True], [False]) == 'PHYSICAL_OR_DOMAIN_EXIT'
    assert capture_stop_reason([False], [False], [False]) is None
    for reason in ['PHYSICAL_OR_DOMAIN_EXIT', 'PHASE_NOT_REACHED']:
        assert capture_result_action(dict(phase='error', capture_stop_reason=reason), 1) == 'UNAVAILABLE'
    assert capture_result_action(dict(phase='completed'), 0) == 'CAPTURED'


def test_path_evidence_allows_declared_micron_roundoff_but_not_geometry_change():
    import numpy as np
    expected = np.array([[0., 0.], [1., 70.]], dtype=np.float32)
    actual = expected.copy()
    actual[1, 1] += np.float32(7.6293945e-6)
    with pytest.raises(AssertionError):
        validate_path_evidence(actual, expected)
    assert validate_path_evidence(actual, expected, atol_m=1e-4) < 1e-4
    actual[1, 1] += .01
    with pytest.raises(AssertionError):
        validate_path_evidence(actual, expected, atol_m=1e-4)
