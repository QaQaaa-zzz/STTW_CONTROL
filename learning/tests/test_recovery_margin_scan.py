import numpy as np
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
