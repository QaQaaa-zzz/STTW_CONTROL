from collections import defaultdict
import copy

import pytest

from sttw_control.reference_feedback_confirmation import confirmation_cases
from sttw_control.finite_value_evaluation_report import role_labels


def inputs():
    protocol=dict(schema='reference_feedback_time_duration_confirmation_v1',seed=2667,
        roles=['parent_plain','parent_causal','free_GAE','feedback_005','feedback_010'],
        alphas=[0.,.5,1.],forces_N=[-16.,16.,-32.,32.],durations_s=[.06,.14],
        onset_upper_tick_range=[140,241])
    contexts={name:dict(speed_m_s=speed,world_direction=[0.,1.,0.]) for name,speed in [('left',1.2),('right',1.5)]}
    source=[dict(context=context,alpha=alpha,force_N=force,pulse_onset_task_tick=600)
            for context in contexts for alpha in protocol['alphas'] for force in protocol['forces_N']]
    return protocol,contexts,source


def test_new_times_duration_pairs_and_all_five_methods_are_matched():
    protocol,contexts,source=inputs()
    cases=confirmation_cases(protocol,contexts,source)
    assert len(cases)==len({r['case_id'] for r in cases})==240
    groups=defaultdict(list);duration_pairs=defaultdict(set)
    for row in cases:
        assert row['pulse_onset_task_tick']!=600
        assert 560<=row['pulse_onset_task_tick']<=960 and row['pulse_onset_task_tick']%4==0
        assert row['pulse_duration_s'] in [.06,.14]
        groups[row['context'],row['alpha'],row['event_id']].append(row)
        duration_pairs[row['context'],row['alpha'],row['force_N']].add(row['pulse_onset_task_tick'])
        assert row['mode_index']==(row['role']!='parent_plain')
        assert row['feedback_gain_rad']=={'feedback_005':.05,'feedback_010':.1}.get(row['role'],0)
    assert len(groups)==48 and all(len(group)==5 for group in groups.values())
    for group in groups.values():
        assert len({(r['pulse_onset_task_tick'],r['pulse_duration_s'],r['force_N'],r['alpha']) for r in group})==1
    assert all(len(times)==1 for times in duration_pairs.values())
    assert cases==confirmation_cases(protocol,contexts,source)


def test_missing_control_or_non_grid_duration_is_rejected():
    protocol,contexts,source=inputs()
    wrong=copy.deepcopy(protocol);wrong['roles'].pop()
    with pytest.raises(ValueError):confirmation_cases(wrong,contexts,source)
    wrong=copy.deepcopy(protocol);wrong['durations_s']=[.07,.14]
    with pytest.raises(ValueError):confirmation_cases(wrong,contexts,source)


def test_zero_gain_wrapping_does_not_mislabel_parent_as_a_different_actor():
    labels=role_labels([dict(role='parent_plain',feedback_gain_rad=0.,causal_support=False),
                        dict(role='free_GAE',feedback_gain_rad=0.,causal_support=True)])
    assert labels==dict(parent_plain='parent_plain',free_GAE='free_GAE + short support')
