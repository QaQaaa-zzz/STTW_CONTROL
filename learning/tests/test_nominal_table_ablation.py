import copy
import json
from pathlib import Path

import pytest

from sttw_control.nominal_table_ablation import nominal_profile_key,shared_cases,weak_feedback_cases,parent_feedback_cases


def inputs():
    protocol=json.loads((Path(__file__).parents[1]/'configs/nominal_table_ablation.json').read_text())
    source=[dict(case_id=f'{context}/a{alpha:g}/F{force}/feedback_nominal',role='feedback_nominal',
        alpha=alpha,context=context,force_N=force,feedback_kind=1,feedback_gain_rad=.1,mode_index=1,
        actor_role='free_GAE',causal_support=True,pulse_onset_task_tick=644,pulse_duration_s=.1)
        for context in ['left_bank2','left_bank4','right_bank2','right_bank4']
        for alpha in [0.,.5,1.] for force in [0.,-16.,16.,-32.,32.]]
    return protocol,source


def test_shared_table_changes_only_lookup_and_identity_not_actor_alpha_or_events():
    protocol,source=inputs();old=copy.deepcopy(source);result=shared_cases(source,protocol)
    assert source==old and len(result)==40
    assert len({r['case_id'] for r in result})==40 and sum(not r['force_N'] for r in result)==8
    lookup={r['case_id']:r for r in source}
    for row in result:
        original=lookup[row['case_id'].rsplit('/',1)[0]+'/feedback_nominal']
        assert row['alpha'] in [0.,1.] and row['nominal_profile_alpha']==.5
        assert row['actor_role']=='free_GAE'
        for key in original:
            if key not in ['role','case_id']:assert row[key]==original[key]
        assert nominal_profile_key(row)==row['context']+'/a0.5'


def test_default_profile_lookup_preserves_existing_own_alpha_and_midpoint_identity():
    for alpha in [0.,.5,1.]:
        assert nominal_profile_key(dict(context='left',alpha=alpha))==f'left/a{alpha:g}'
    assert nominal_profile_key(dict(context='left',alpha=.5,nominal_profile_alpha=.5))=='left/a0.5'


def test_no_midpoint_repeats_or_posthoc_profile_gain_threshold_choices():
    p,source=inputs()
    for key,value in [('executed_alphas',[0.,.5,1.]),('nominal_profile_alpha',0.),('feedback_gain_rad',.05),('goal_radius_m',.6)]:
        wrong=copy.deepcopy(p);wrong[key]=value
        with pytest.raises(ValueError):shared_cases(source,wrong)
    with pytest.raises(ValueError):shared_cases(source[:-1],p)


def test_weak_feedback_preserves_full_alpha_events_and_only_changes_gain():
    _,source=inputs()
    for row in source:
        row.update(role='feedback_010',feedback_kind=3,
                   case_id=row['case_id'].replace('feedback_nominal','feedback_010'))
    original=copy.deepcopy(source)
    protocol=json.loads((Path(__file__).parents[1]/'configs/weak_feedback_confirmation.json').read_text())
    cases=weak_feedback_cases(source,protocol)
    assert len(cases)==60 and source==original
    assert sum(not row['force_N'] for row in cases)==12
    assert set(row['alpha'] for row in cases)=={0.,.5,1.}
    for before,after in zip(source,cases):
        assert after['feedback_gain_rad']==.05 and after['role']=='feedback_005'
        for key in before:
            if key not in ['case_id','role','feedback_gain_rad']:assert after[key]==before[key]
    for key,value in [('feedback_gain_rad',.075),('executed_alphas',[0.,1.]),('goal_radius_m',.6)]:
        wrong=copy.deepcopy(protocol);wrong[key]=value
        with pytest.raises(ValueError):weak_feedback_cases(source,wrong)
    with pytest.raises(ValueError):weak_feedback_cases(source[:-1],protocol)


def test_parent_actor_control_changes_actor_identity_only_and_labels_truthfully():
    _,source=inputs()
    for row in source:
        row.update(role='feedback_010',feedback_kind=3,
                   case_id=row['case_id'].replace('feedback_nominal','feedback_010'))
    original=copy.deepcopy(source)
    protocol=json.loads((Path(__file__).parents[1]/'configs/parent_feedback_confirmation.json').read_text())
    cases=parent_feedback_cases(source,protocol)
    assert source==original and len(cases)==60
    assert sum(not row['force_N'] for row in cases)==12
    for before,after in zip(source,cases):
        assert after['role']=='parent_feedback_010' and after['actor_role']=='parent_causal'
        for key in before:
            if key not in ['case_id','role','actor_role']:assert after[key]==before[key]
    from sttw_control.finite_value_evaluation_report import role_labels
    label=role_labels(cases)['parent_feedback_010']
    assert 'parent task_only64' in label and 'free_GAE' not in label and 'short support' in label
    for key,value in [('actor_role','free_GAE'),('feedback_gain_rad',.05),('goal_radius_m',.6)]:
        wrong=copy.deepcopy(protocol);wrong[key]=value
        with pytest.raises(ValueError):parent_feedback_cases(source,wrong)
    with pytest.raises(ValueError):parent_feedback_cases(source[:-1],protocol)
