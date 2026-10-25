"""Preserve original task bands, goal section and missing held-tail semantics."""
import numpy as np
import pytest


def fixture():
    config=dict(timing={'lower_dt_s':.005},training_future_phase_C={'episode_s':.04},
        stage_B_baseline=dict(hold_s=.01,final_y_error_m=.1,final_path_heading_error_rad=.05,
                              final_speed_error_m_s=.1,working_roll_rad=.302))
    p={k:np.zeros(8,bool) for k in ['domain_exit','contact_failure','nonfinite','lower_fault','policy_fault','fallback_active']}
    p.update(active_tick=np.ones(8,bool),task_reward_valid=np.ones(8,bool),physical_tick=np.arange(8),
        time=np.arange(8)*.005,actual_forward_speed=np.full(8,1.5),path_cross_track=np.full(8,.05),
        path_heading_error=np.full(8,.02),path_progress=np.arange(8)*.1,
        goal_section_signed_distance=np.arange(8)*.1-.5,peak_roll=np.full(8,.1))
    return p,config


def test_full_task_metrics_match_original_label_and_require_progress_as_well_as_goal_plane():
    from sttw_control.survival_endpoint_task_metrics import task_metrics
    from sttw_control.recovery_margin_scan import outcome_labels
    p,cfg=fixture()
    got=task_metrics(p,speed=1.5,goal_s=.5,config=cfg)
    original=outcome_labels(dict(p,physical_failure=p['contact_failure']),speed=1.5,goal_s=.5,
                            working_roll=.302,horizon=.04,hold_s=.01)
    assert got['task_recovered']==original['task_recovered']==True
    assert got['performance_success']==original['performance_success']==True
    assert got['tail_observed_ticks']==2 and got['task_path_rmse_m']==pytest.approx(.05)
    p['path_progress'][:]=0
    assert not task_metrics(p,speed=1.5,goal_s=.5,config=cfg)['task_recovered']
    p['path_progress']=np.arange(8)*.1;p['peak_roll'][:]=.31
    row=task_metrics(p,speed=1.5,goal_s=.5,config=cfg)
    assert row['task_recovered'] and not row['performance_success']


def test_surviving_physical_hold_tail_does_not_supply_task_errors_or_task_success():
    from sttw_control.survival_endpoint_task_metrics import task_metrics
    p,cfg=fixture();p['domain_exit'][3:]=True;p['fallback_active'][4:]=True;p['task_reward_valid'][4:]=False
    for k in ['path_cross_track','path_heading_error','path_progress','goal_section_signed_distance']:p[k][4:]=np.nan
    got=task_metrics(p,speed=1.5,goal_s=.5,config=cfg)
    assert got['task_recovered'] is False and got['task_metric_status']=='observed_failure'
    assert got['task_observed_ticks']==4 and got['tail_observed_ticks']==0
    assert got['tail_path_max_m'] is None and got['tail_speed_error_max_m_s'] is None
    assert got['task_path_rmse_m']==pytest.approx(.05)


def test_partial_fault_and_clock_gaps_are_not_declared_task_failures():
    from sttw_control.survival_endpoint_task_metrics import task_metrics
    p,cfg=fixture();p['active_tick'][4:]=False;p['task_reward_valid'][4:]=False
    assert task_metrics(p,speed=1.5,goal_s=.5,config=cfg)['task_recovered'] is None
    p['lower_fault'][3]=True
    assert task_metrics(p,speed=1.5,goal_s=.5,config=cfg)['task_metric_status']=='unknown'
    p['lower_fault'][:]=False;p['physical_tick'][2]=9
    with pytest.raises(ValueError):task_metrics(p,speed=1.5,goal_s=.5,config=cfg)


def test_held_working_roll_exceedance_and_empty_task_data_keep_their_own_scopes():
    from sttw_control.survival_endpoint_task_metrics import task_metrics
    p,cfg=fixture();p['domain_exit'][3:]=True;p['task_reward_valid'][4:]=False;p['peak_roll'][4:]=.5
    row=task_metrics(p,speed=1.5,goal_s=.5,config=cfg)
    assert row['working_failure'] is False and row['physical_working_failure_observed'] is True
    p['active_tick'][:]=False;p['task_reward_valid'][:]=False
    row=task_metrics(p,speed=1.5,goal_s=.5,config=cfg)
    assert row['task_recovered'] is None and row['task_path_abs_error_time_integral_m_s'] is None


def test_force_comparisons_use_own_zero_baseline_and_keep_unknown_task_labels():
    from sttw_control.survival_endpoint_task_metrics import zero_relative_cases
    base=dict(panel='physical_stochastic',role='physical_credit',route='right',speed=1.5,alpha=.5,replicate=2)
    rows=[dict(base,case_id='zero',force_N=0.,outcome='survived',task_recovered=None),
          dict(base,case_id='force',force_N=-32.,outcome='contact',task_recovered=False)]
    got=zero_relative_cases(rows)
    assert got[0]['physical_change_vs_zero']==-1 and got[0]['task_change_vs_zero'] is None
    assert got[0]['zero_case_id']=='zero'
    with pytest.raises(ValueError):zero_relative_cases(rows[1:])
