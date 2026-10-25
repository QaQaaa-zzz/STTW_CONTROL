import copy
import numpy as np
import pytest
from sttw_control.live_endpoint_analysis import paired_endpoint_report,tracking_recovery_time,block_reward_series


ROLES=['parent_plain','parent_support','trained_plain','trained_support']


def rows():
    categories=[['safe_task']*4,['contact','safe_task_failed','safe_task','safe_task'],
                ['safe_task','safe_task','unknown','contact']]
    result=[]
    for event,cats in enumerate(categories):
        for role,category in zip(ROLES,cats):
            result.append(dict(case_id=f'{event}/{role}',context='left',alpha=.5,role=role,event_id=str(event),
                force_N=[0.,-32.,32.][event],pulse_onset_task_tick=400,pulse_duration_s=.1,
                category=category,scored_cost=None if category=='unknown' else 10.+event))
    return result


def test_pair_counts_keep_unknown_separate_and_count_gains_and_losses():
    result=paired_endpoint_report(rows())
    x=result['comparisons']['trained_plain_vs_parent_plain']
    assert x['declared_pairs']==3 and x['known_pairs']==2 and x['unknown_pairs']==1
    assert x['physical_gains']==1 and x['physical_losses']==0 and x['task_gains']==1
    y=result['comparisons']['trained_support_vs_parent_support']
    assert y['physical_gains']==0 and y['physical_losses']==1
    assert y['task_gains']==1 and y['task_losses']==1
    assert result['role_counts']['trained_plain']['unknown']==1
    assert result['role_counts']['trained_plain']['safe_task']==2


def test_pairing_rejects_duplicate_missing_or_different_actual_disturbance():
    a=rows()
    with pytest.raises(ValueError):paired_endpoint_report(a+[a[0]])
    with pytest.raises(ValueError):paired_endpoint_report(a[:-1])
    a=copy.deepcopy(a);a[-1]['pulse_onset_task_tick']=401
    with pytest.raises(ValueError):paired_endpoint_report(a)
    a=rows();a[-1]['world_direction']=[1.,0.,0.]
    with pytest.raises(ValueError):paired_endpoint_report(a)


CFG=dict(timing={'lower_dt_s':.005},stage_B_baseline=dict(hold_s=1.,final_y_error_m=.1,
    final_path_heading_error_rad=.05,final_speed_error_m_s=.1))


def trace(n=600):
    return dict(active_tick=np.ones(n,bool),physical_tick=np.arange(n),path_cross_track=np.zeros(n),
        path_heading_error=np.zeros(n),actual_forward_speed=np.full(n,1.5))


def test_recovery_requires_full_configured_hold_and_distinguishes_no_disruption():
    a=trace();a['path_cross_track'][100:200]=.2
    r=tracking_recovery_time(a,speed=1.5,onset_tick=100,config=CFG)
    assert r['status']=='recovered' and r['first_violation_s']==.5
    assert r['recovery_start_s']==1. and r['confirmed_at_s']==2.
    assert r['delay_from_onset_s']==.5
    assert tracking_recovery_time(trace(),speed=1.5,onset_tick=100,config=CFG)['status']=='not_disrupted'


def test_unfinished_hold_missing_clock_and_nonfinite_are_not_recovered():
    a=trace(300);a['path_cross_track'][100:200]=.2
    assert tracking_recovery_time(a,speed=1.5,onset_tick=100,config=CFG)['status']=='not_recovered'
    b=trace();b['physical_tick'][200]+=1
    with pytest.raises(ValueError):tracking_recovery_time(b,speed=1.5,onset_tick=100,config=CFG)


def test_partial_contact_block_keeps_terminal_replacement_separate_from_running_components():
    cfg=dict(CFG,path_reward={'scale':.1})
    a=dict(active_tick=np.array([True]*6+[False]*2),physical_tick=np.array([0,1,2,3,4,5,-1,-1]),
        effective_costs=np.ones((8,2)),scored_tick_reward=np.array([-.001]*4+[0.,-100.,0.,0.]))
    r=block_reward_series(a,config=cfg)
    np.testing.assert_allclose(r['scored_reward'],[-.004,-100.])
    np.testing.assert_allclose(r['signed_running_components'].sum(1),[-.004,-.002])
    np.testing.assert_allclose(r['terminal_replacement_reward'],[0.,-99.998])
    np.testing.assert_allclose(r['signed_running_components'].sum(1)+r['terminal_replacement_reward'],r['scored_reward'])
    b=trace();b['path_cross_track'][110]=np.nan
    with pytest.raises(ValueError):tracking_recovery_time(b,speed=1.5,onset_tick=100,config=CFG)
