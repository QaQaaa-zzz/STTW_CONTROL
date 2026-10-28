"""Software-state recursion tests; no plant/environment or saved-run execution."""
import json
import numpy as np
import pytest
from sttw_control.coordination_replay import replay_window, initial_state


def fixture():
    n = 20
    raw = {k: np.zeros(n) for k in ('coord_prev_alpha', 'coord_latched_before',
        'pre_upper_yaw', 'coord_pre_progress', 'alpha_exec', 'rule_latched',
        'coord_degraded', 'coord_guard_fallback_selected')}
    raw.update(time=np.arange(n)*.005, active_tick=np.ones(n, bool), query_observation=np.zeros((n, 351)),
        coord_latents=np.zeros((n, 3, 2)), coord_nominal=np.zeros((n, 2)),
        nominal=np.zeros((n, 2)), coord_permission=np.ones(n, bool),
        filtered_offset=np.zeros((n, 2)), offsets=np.zeros((n, 2)),
        coord_probabilities=np.zeros((n, 3, 4)), coord_qmin=np.zeros((n, 3)),
        coord_task_mask=np.ones((n, 3), bool), coord_proxy_cost=np.zeros((n, 3)),
        target_offset=np.zeros((n, 2)), governed=np.zeros((n, 2)),
        latent_z=np.zeros((n, 2)), composed_latent=np.zeros((n, 2)))
    raw['coord_prev_alpha'][4] = .5
    raw['filtered_offset'][3] = [.1, .2]
    raw['offsets'][3] = [.3, .4]
    calls = []
    def decide(targets, curvature, nominal, speed, quality, permission, latch, prev, user):
        calls.append((prev, latch, permission))
        alpha = 1.-prev if permission else prev
        return dict(alpha=alpha, latched=not latch if permission else latch,
                    task_mask=np.ones(3, bool), proxy_cost=np.zeros(3),
                    degraded=False, guard_fallback_selected=False)
    callbacks = dict(actor=lambda obs: np.column_stack([obs[:, 336]+1, obs[:, 336]]),
        probabilities=lambda obs, z, yaw: (np.ones((3, 4)), np.zeros((3, 2))),
        targets=lambda nominal, offsets: nominal[None]+offsets,
        path=lambda tick, progress: ('changed' if tick >= 8 else 'original', .2),
        decision=decide,
        filter=lambda f, o, z, nominal: (f*.8+z*.2, o*.5+z*.5, nominal+o*.5+z*.5, z))
    return raw, dict(speed=2.6, effective_user_alpha=.5), callbacks, calls


def test_recurs_memory_and_filter_without_log_reset_and_midblock_resume():
    raw, case, cb, calls = fixture()
    raw['coord_permission'][8:12] = False
    # Corrupt all later logged state: replay must use only its initial memory.
    raw['coord_prev_alpha'][5:] = 99
    raw['coord_latched_before'][5:] = 99
    raw['filtered_offset'][4:] = 99
    raw['offsets'][4:] = 99
    full, final = replay_window(raw, case, 4, 16, cb)
    assert calls == [(.5, False, True), (.5, True, False), (.5, True, True)]
    assert [p['new_decision'] for p in full] == [True, False, False, False]*3
    assert all(p['issued'] == [0., 0.] for p in full[4:8])
    assert full[4]['filtered'] != [99., 99.]
    first, state = replay_window(raw, case, 4, 7, cb)
    second, resumed = replay_window(raw, case, 7, 16, cb, state=json.loads(json.dumps(state)))
    assert first+second == full
    assert resumed == final
    assert full[3]['path_id'] == 'original' and full[4]['path_id'] == 'changed'


def test_missing_field_clock_and_unaligned_initialization_rejected():
    raw, case, cb, _ = fixture()
    with pytest.raises(ValueError, match='boundary'):
        initial_state(raw, 5)
    state = initial_state(raw, 4)
    with pytest.raises(ValueError, match='clock'):
        replay_window(raw, case, 8, 12, cb, state=state)
    del raw['query_observation']
    with pytest.raises(ValueError, match='missing replay fields'):
        replay_window(raw, case, 4, 12, cb)


def test_decision_memory_can_change_across_upper_blocks():
    raw, case, cb, calls = fixture()
    raw['coord_prev_alpha'][4] = 0
    packets, _ = replay_window(raw, case, 4, 16, cb)
    assert [c[0] for c in calls] == [0., 1., 0.]
    assert [p['alpha'] for p in packets[::4]] == [1., 0., 1.]


def test_comparison_bounds_never_allow_discrete_or_q_gate_changes():
    from sttw_control.coordination_replay import COMPARISONS, compare_packets
    raw, case, cb, _ = fixture()
    packets, _ = replay_window(raw, case, 4, 8, cb)
    # Construct matching reference, then introduce small engineering differences.
    for name, (field, _) in COMPARISONS.items():
        values = np.asarray([p[name] for p in packets])
        raw[field] = np.zeros((20,)+values.shape[1:], dtype=values.dtype)
        raw[field][4:8] = values
    raw['composed_latent'][4, 0] += 1.5e-4
    assert not compare_packets(packets, raw, 4, 8)['failed_fields']
    raw['coord_qmin'][4, 0] = .79999
    packets[0]['qmin'] = list(packets[0]['qmin'])
    packets[0]['qmin'][0] = .80001  # numeric error tiny, qualification differs
    raw['governed'][4, 1] += 3e-5  # exceeds steering bound, independently reported
    raw['alpha_exec'][4] = .50000001  # discrete identity requires exact comparison
    result = compare_packets(packets, raw, 4, 8)
    assert 'qmin' not in result['failed_fields']
    assert 'q_gate_0.8' in result['failed_fields']
    assert 'governed' in result['failed_fields']
    assert 'alpha' in result['failed_fields']
    assert result['maximum_component_absolute_errors']['governed'][1] > 2e-5
