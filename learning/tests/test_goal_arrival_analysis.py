"""A hindsight per-event ceiling must not masquerade as one fixed decision."""
import pytest

from sttw_control.goal_arrival_analysis import reference_bank_opportunity


def inventory():
    rows = []
    for route in ['left', 'right']:
        for alpha in [0., .25, .5, .75, 1.]:
            for event in range(7):
                for role in ['task_only', 'task_floor', 'task_early', 'task_constant']:
                    hit = ((role == 'task_only' and event in [0, 1, 2, 3]) or
                           (role == 'task_early' and event in [0, 4, 5, 6]))
                    rows.append(dict(dataset='early_prospective', context=route, alpha=alpha,
                        event_id=str(event), force_N=0 if event == 0 else 32,
                        role=role, goal_radius_m=.3, goal_arrived=hit, physical_outcome='survived'))
    return rows


def test_event_oracle_and_fixed_route_alpha_choice_are_separate(tmp_path):
    result = reference_bank_opportunity(inventory(), tmp_path, .3)
    assert result['per_event_ceilings']['disturbed']['any_candidate_arrival'] == 60
    assert sum(r['disturbed_arrival'] for r in result['fixed_route_alpha']) == 30
    assert len(result['fixed_route_alpha']) == 10


def test_unknowns_and_duplicate_candidates_are_not_silent_failures(tmp_path):
    rows = inventory(); rows[0]['goal_arrived'] = None
    with pytest.raises(ValueError, match='known outcomes'):
        reference_bank_opportunity(rows, tmp_path, .3)
    rows = inventory(); rows.append(rows[0])
    with pytest.raises(ValueError, match='duplicate'):
        reference_bank_opportunity(rows, tmp_path, .3)


def test_both_force_signs_are_held_out_and_cannot_choose_their_own_rule(tmp_path):
    rows = inventory()
    for row in rows:
        event = int(row['event_id'])
        if event:
            onset = 100 * ((event + 1) // 2)
            sign = -32 if event % 2 else 32
            row['event_id'] = f't{onset}_F{sign:+g}'
            row['force_N'] = sign
        else:
            row['event_id'] = 'zero'
    first = reference_bank_opportunity(rows, tmp_path, .3)['leave_one_onset_out']
    assert first['test_count'] == 60
    for row in rows:
        if row['event_id'].startswith('t100_'):
            row['goal_arrived'] = not row['goal_arrived']
    second = reference_bank_opportunity(rows, tmp_path, .3)['leave_one_onset_out']
    selections = lambda result: [(c['route'], c['alpha'], c['selected_role'])
        for c in result['decisions'] if c['held_onset'] == 't100']
    assert selections(first) == selections(second)
