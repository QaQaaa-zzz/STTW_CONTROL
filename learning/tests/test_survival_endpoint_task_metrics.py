"""Preserve original task bands, goal section and missing held-tail semantics."""
import numpy as np
import pytest


def test_declared_continuation_keeps_original_goal_clock_and_remaining_horizon():
    from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics
    p,_=fixture();p['physical_tick']=np.arange(2,10);p['active_tick']=np.arange(8)<6
    p['actual_xy']=np.full((8,2),4.)
    kwargs=dict(goal_xy=[1.,0.],radius_m=.3,dt=.005,horizon_s=.04)
    with pytest.raises(ValueError,match='clock prefix'):goal_arrival_metrics(p,**kwargs)
    result=goal_arrival_metrics(p,start_tick=2,**kwargs)
    assert result['goal_arrived'] is False and result['arrival_outcome']=='not_arrived'
    p['actual_xy'][2]=[1.,0.]
    assert goal_arrival_metrics(p,start_tick=2,**kwargs)['first_arrival_s']==pytest.approx(.025)


def goal_cost_fixture():
    p, _ = fixture()
    p['actual_xy'] = np.full((8, 2), 4.)
    return p


def running_metrics(p, costs, valid=None):
    from sttw_control.survival_endpoint_task_metrics import goal_running_cost_metrics
    return goal_running_cost_metrics(p, goal_xy=[1., 0.], radius_m=.3,
        dt=.005, horizon_s=.04, running_tick_cost=costs,
        cost_valid=np.ones(8, bool) if valid is None else valid)


def test_goal_cost_includes_entry_tick_without_erasing_later_cost_or_fall():
    p = goal_cost_fixture(); p['actual_xy'][2] = [1., 0.]
    p['contact_failure'][6] = True
    costs = np.column_stack([np.arange(1., 9.), np.full(8, 2.)])
    r = running_metrics(p, costs)
    assert r['goal_arrived'] and r['contact_after_arrival']
    assert r['goal_entry_physical_tick'] == 2
    assert r['pre_arrival_running_cost'] == [6., 6.]
    assert r['post_arrival_observed_running_cost'] == [30., 10.]
    np.testing.assert_equal(np.array(r['pre_arrival_running_cost']) +
                            r['post_arrival_observed_running_cost'], r['full_observed_running_cost'])
    assert r['observed_before_fatal_running_cost'] == [21., 12.]


def test_goal_on_last_horizon_sample_counts_but_simultaneous_contact_does_not():
    p = goal_cost_fixture(); p['actual_xy'][-1] = [1., 0.]
    assert running_metrics(p, np.ones(8))['first_arrival_s'] == pytest.approx(.04)
    p['contact_failure'][-1] = True
    r = running_metrics(p, np.ones(8))
    assert r['goal_arrived'] is False and r['pre_arrival_running_cost'] is None


def test_measured_goal_with_missing_task_cost_is_not_turned_into_failure():
    p = goal_cost_fixture(); p['actual_xy'][4] = [1., 0.]
    p['domain_exit'][2:] = True  # Physics continues, task cost unavailable.
    valid = np.arange(8) < 2
    costs = np.where(valid, 1., np.nan)
    r = running_metrics(p, costs, valid)
    assert r['goal_arrived'] is True
    assert r['pre_arrival_running_cost'] is None
    assert r['full_observed_running_cost'] is None


def test_missing_post_goal_cost_preserves_known_pre_goal_subtotal():
    p = goal_cost_fixture(); p['actual_xy'][1] = [1., 0.]
    valid = np.arange(8) < 4
    costs = np.where(valid, 3., np.nan)
    r = running_metrics(p, costs, valid)
    assert r['pre_arrival_running_cost'] == 6.
    assert r['post_arrival_observed_running_cost'] is None


def test_partial_or_faulted_no_goal_observation_stays_unknown_and_padding_is_unscored():
    p = goal_cost_fixture(); p['active_tick'][4:] = False
    costs = np.where(p['active_tick'], 2., np.nan)
    r = running_metrics(p, costs)
    assert r['goal_arrived'] is None and r['pre_arrival_running_cost'] is None
    assert r['full_observed_running_cost'] == 8.
    assert r['full_observed_ticks'] == 4
    p['lower_fault'][2] = True
    assert running_metrics(p, costs)['observed_before_fatal_running_cost'] == 4.


def test_cost_shape_sign_and_finite_validity_are_explicit():
    p = goal_cost_fixture()
    for costs in (np.ones(7), np.zeros((8, 0)), -np.ones(8), np.full(8, np.nan)):
        with pytest.raises(ValueError):
            running_metrics(p, costs)


def test_goal_arrival_is_spatial_not_goal_plane_or_final_hold():
    from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics
    p, _ = fixture()
    p['actual_xy'] = np.array([[0., 1.], [.5, 1.], [1., 1.], [2., 1.],
                              [2., 2.], [2., 3.], [2., 4.], [2., 5.]])
    m = lambda: goal_arrival_metrics(p, goal_xy=[1, 0], radius_m=.3, dt=.005, horizon_s=.04)
    assert m()['goal_arrived'] is False  # Passing the target plane is insufficient.
    p['actual_xy'][2] = [1., .2]
    p['path_heading_error'][:] = 2.; p['actual_forward_speed'][:] = .3
    assert m()['goal_arrived'] is True
    assert m()['first_arrival_s'] == pytest.approx(.015)


def test_goal_arrival_preserves_later_fall_but_never_counts_fallen_position():
    from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics
    p, _ = fixture(); p['actual_xy'] = np.full((8, 2), 4.)
    p['actual_xy'][4] = [1., 0.]; p['contact_failure'][5] = True
    m = lambda: goal_arrival_metrics(p, goal_xy=[1, 0], radius_m=.3, dt=.005, horizon_s=.04)
    assert m()['goal_arrived'] and m()['contact_after_arrival']
    p['contact_failure'][3] = True
    assert m()['goal_arrived'] is False and m()['arrival_outcome'] == 'contact_before_arrival'


def test_goal_arrival_fault_partial_prefix_and_clock_semantics():
    from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics
    p, _ = fixture(); p['actual_xy'] = np.full((8, 2), 4.)
    m = lambda: goal_arrival_metrics(p, goal_xy=[1, 0], radius_m=.3, dt=.005, horizon_s=.04)
    p['lower_fault'][2] = True
    assert m()['goal_arrived'] is None
    p['lower_fault'][:] = False; p['active_tick'][4:] = False
    assert m()['goal_arrived'] is None
    p['actual_xy'][1] = [1., 0.]
    assert m()['goal_arrived'] is True  # Observed arrival remains known.
    p['physical_tick'][2] = 9
    with pytest.raises(ValueError): m()


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
