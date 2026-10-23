import numpy as np
import pytest
from sttw_control import recovery_margin_scan as scan


def data(end=20.):
    t=np.arange(.8,end,.005)
    return dict(time=t,physical_failure=np.zeros(len(t),bool),domain_exit=np.zeros(len(t),bool),
        peak_roll=np.full(len(t),.2),path_cross_track=np.zeros(len(t)),
        path_heading_error=np.zeros(len(t)),actual_forward_speed=np.full(len(t),2.45),
        goal_section_signed_distance=np.ones(len(t)),path_progress=np.full(len(t),10.))


def test_working_limit_and_task_recovery_are_distinct_labels():
    d=data();d['peak_roll'][300]=.31
    r=scan.outcome_labels(d,speed=2.45,goal_s=8.,working_roll=.302)
    assert r['task_recovered'] and r['physical_survived_horizon'] is True
    assert r['working_failure'] and not r['performance_success']


def test_early_domain_failure_is_unknown_physical_horizon_not_a_fall():
    d=data(8.5);d['domain_exit'][-1]=True
    r=scan.outcome_labels(d,speed=2.45,goal_s=8.,working_roll=.302)
    assert not r['task_recovered'] and not r['physical_failure']
    assert r['physical_survived_horizon'] is None
    assert not r['final_hold']


def test_final_hold_uses_original_deadline_and_all_three_errors():
    for key,delta in [('path_cross_track',.11),('path_heading_error',.051),('actual_forward_speed',.101)]:
        d=data();d[key][-1]+=delta
        r=scan.outcome_labels(d,speed=2.45,goal_s=8.,working_roll=.302)
        assert r['complete_horizon'] and not r['final_hold'] and not r['task_recovered']


def test_preference_screen_preserves_ten_second_half_second_hold_contract():
    d=data(10.);d['path_cross_track'][d['time']<9.5]=.2
    r=scan.outcome_labels(d,speed=2.45,goal_s=8.,working_roll=.302,horizon=10.,hold_s=.5)
    assert r['task_recovered'] and r['complete_horizon']


@pytest.mark.parametrize('horizon',[10.,20.])
def test_scan_contract_uses_original_config_deadline_and_explicit_state_roles(horizon):
    manifest={'config':{'training_future_phase_C':{'episode_s':horizon}},
              'original_task_horizon_s':horizon}
    decl={'maximum_lower_ticks':int(6*horizon/.005),
          'roles_by_bank_index':{'2':'calibration','9':'validation'}}
    c=scan.scan_contract(decl,manifest,[2,9],[0,-8,8])
    assert c['horizon_s']==horizon
    assert c['maximum_policy_intervals']==round(horizon/.02)
    assert c['roles']==['calibration','validation']


def test_scan_contract_rejects_deadline_budget_and_role_drift():
    m={'config':{'training_future_phase_C':{'episode_s':10.}},'original_task_horizon_s':10.}
    d={'maximum_lower_ticks':8001,'roles_by_bank_index':{'2':'calibration','9':'validation'}}
    with pytest.raises(ValueError,match='budget'):scan.scan_contract(d,m,[2,9],[0,8])
    d['maximum_lower_ticks']=4000;d['roles_by_bank_index'].pop('9')
    with pytest.raises(ValueError,match='roles'):scan.scan_contract(d,m,[2,9],[0])
    m['original_task_horizon_s']=20.
    with pytest.raises(ValueError,match='deadline'):scan.scan_contract(d,m,[2,9],[0])


def test_difficulty_audit_excludes_nominal_failures_and_keeps_state_roles():
    rows=[]
    for idx,role,nominal in [(2,'calibration',True),(9,'validation',False)]:
        for force,ok,risk in [(0,nominal,0.),(8,True,1.),(-8,False,2.)]:
            rows.append(dict(bank_index=idx,data_role=role,alpha=0.,force_N=force,
                             task_recovered=ok,domain_failure=not ok,
                             performance_success=ok,physical_failure=False,predicted_risk=risk))
    x=scan.difficulty_summary(rows)
    a,b=x['by_role']
    assert a['nominal_viable_states']==1 and a['pulse_cases']==2
    assert a['pulse_task_losses']==1 and a['risk_failure_auc']==1.
    assert a['force_magnitude_failure_auc']==.5
    assert b['nominal_viable_states']==0 and b['pulse_cases']==0
    assert b['risk_failure_auc'] is None


def test_difficulty_audit_does_not_invent_auc_with_no_task_negatives():
    rows=[dict(bank_index=2,data_role='validation',alpha=0.,force_N=f,
               task_recovered=True,domain_failure=False,performance_success=True,
               physical_failure=False,predicted_risk=abs(f)) for f in [0,8,-8]]
    assert scan.difficulty_summary(rows)['by_role'][0]['risk_failure_auc'] is None
