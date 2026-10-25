import numpy as np
import pytest


def test_full_five_alpha_panel_pairs_events_and_has_one_zero_execution_per_event():
    from sttw_control.prospective_preference import prospective_cases
    rows=prospective_cases('left90_R1p1',speed=1.8,world_direction=[.6,.8,0.],alphas=[0.,.25,.5,.75,1.],onsets=[384,480,636])
    assert len(rows)==217 and len({r['case_id'] for r in rows})==217
    zeros=[r for r in rows if r['role']=='zero_middle']
    assert len(zeros)==7 and all(r['alpha'] is None and r['state_alpha']==0 and r['zero_mode'] for r in zeros)
    early=[r for r in rows if r['role']=='task_early'];assert len(early)==35
    for a in [0.,.25,.5,.75,1.]:
        subset=[r for r in early if r['alpha']==a]
        assert len(subset)==7
        assert {(r['pulse_onset_task_tick'],r['force_N']) for r in subset if r['force_N']}=={(t,f) for t in [384,480,636] for f in [-32.,32.]}
    assert all(r['source_batch_slot']==0 and r['stage']=='reentry' for r in rows)
    assert all(r['reference_speed_floor_m_s']==1.5 for r in rows if r['role']=='task_floor')
    assert all(r['reference_speed_floor_m_s']==1. for r in rows if r['role']!='task_floor')
    assert all(r['profile']==6 for r in rows if r['role'].endswith('early'))
    assert all(r['profile']==7 for r in rows if r['role'].endswith('constant'))
    for bad in [[384,384,636],[385,480,636],[384,480,1996]]:
        with pytest.raises(ValueError):prospective_cases('left90_R1p1',speed=1.8,world_direction=[.6,.8,0.],alphas=[0,.5,1],onsets=bad)
    with pytest.raises(ValueError):prospective_cases('left90_R1p1',speed=1.8,world_direction=[.6,.8,0.],alphas=[0,.5,.5],onsets=[384,480,636])


def reward_fixture():
    # Hand-computed endpoint channels at chi1, ev2, ey3:
    # speed endpoints3/30 capped3/5; path50 capped8;
    # primary endpoints5/3 capped4/3. Atalpha.5 total=4+4.4+3.5=11.9.
    c=dict(reward_version='path_persistent_preference_v3',scale=1.,speed_scale_m_s=1.,cross_track_scale_m=1.,path_heading_factor=0.,path_heading_scale_rad=1.,primary_weight=1.,primary_path_band_m=0.,primary_path_scale_m=1.,primary_speed_band_m_s=0.,primary_speed_scale_m_s=1.,path_secondary_ratio=.1,failure_extra=.1,failure_cost_rate_bound=10.,component_caps=dict(speed=5.,path=8.,primary=4.,roll=1.,working_roll=1.,roll_rate=1.,overspeed=1.,low_speed=1.,offset_magnitude=1.,offset_rate=1.))
    cfg=dict(path_reward=c,timing=dict(lower_dt_s=.005,upper_dt_s=.02),training_future_phase_C=dict(episode_s=.04,gamma=.99))
    data=dict(active_tick=np.ones(8,bool),chi=np.ones(8),actual_forward_speed=np.full(8,3.5),requested_speed=np.full(8,1.5),path_cross_track=np.full(8,3.),path_heading_error=np.zeros(8),effective_costs=np.zeros((8,10)),contact_failure=np.zeros(8,bool),domain_exit=np.ones(8,bool))
    return data,cfg


def test_v3_rescore_caps_endpoint_channels_before_mixing_and_ignores_domain_exit():
    from sttw_control.preference_trace_rescore import rescore_trace_alpha
    d,c=reward_fixture();r=rescore_trace_alpha(d,alpha=.5,config=c)
    np.testing.assert_allclose(r['running_reward'],np.full(8,-.0595))
    np.testing.assert_allclose(r['scored_reward'],r['running_reward'])
    assert r['scored_cost']==pytest.approx(.476)
    assert rescore_trace_alpha(d,alpha=0.,config=c)['scored_cost']==pytest.approx(.6)
    assert rescore_trace_alpha(d,alpha=1.,config=c)['scored_cost']==pytest.approx(.352)


def test_only_actual_contact_replaces_final_active_block_and_keeps_unknown_distinct():
    from sttw_control.preference_trace_rescore import rescore_trace_alpha
    d,c=reward_fixture();d['active_tick'][6:]=False;d['contact_failure'][5]=True
    r=rescore_trace_alpha(d,alpha=.5,config=c)
    np.testing.assert_allclose(r['scored_reward'],[-.0595]*4+[0,-.3,0,0])
    assert r['scored_cost']==pytest.approx(.538)
    with pytest.raises(ValueError):rescore_trace_alpha(d,alpha=1.1,config=c)


def test_common_origin_validation_rejects_permission_or_context_drift():
    import copy
    from sttw_control.prospective_preference import prospective_cases,validate_common_origin_cases
    rows=prospective_cases('left90_R1p1',speed=1.8,world_direction=[.6,.8,0.],alphas=[0,.5,1],onsets=[384,480,636])
    source=[dict(route='left90_R1p1',requested_speed_m_s=1.8,horizon_s=10.,world_direction=[.6,.8,0.])]
    validate_common_origin_cases(rows,source)
    for key,value in [('source_batch_slot',1),('state_alpha',.25),('reference_speed_floor_m_s',3.),('profile',7),('actor_role','parent'),('route','right90_R1p1')]:
        bad=copy.deepcopy(rows);bad[0][key]=value
        with pytest.raises(ValueError):validate_common_origin_cases(bad,source)
