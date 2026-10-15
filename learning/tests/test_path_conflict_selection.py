from copy import deepcopy
import numpy as np
import pytest
from test_path_research_selection import config as research_config
from sttw_control import path_command_selection as s


def config():
    c=research_config();c['preference_selection_schema']='path_conflict_preference_v1'
    c['preference_acceptance'].update(pressure_endpoint_path_rmse_m=.20,
        pressure_endpoint_speed_rmse_m_s=.15,pressure_min_speed_gap_m_s=.05,
        pressure_min_path_gap_m=.02,min_actual_speed_reduction_m_s=.10,
        max_actual_speed_offset_m_s=-.05)
    return c


def trace():
    p=np.r_[np.linspace(0,2,20),np.linspace(2,3,20),np.linspace(3,5,20)]
    return dict(path_progress=p,actual_forward_speed=np.ones(60)*2.6,v_user=np.ones(60)*2.8,
        path_cross_track=np.ones(60)*.1,offsets=np.c_[np.ones(60)*-.2,np.zeros(60)],
        turn_starts=np.array([0.,3.,0.,0.]),turn_ends=np.array([2.,5.,0.,0.]),
        turn_directions=np.array([1.,-1.,0.,0.]),turn_mask=np.array([True,True,False,False]))


def test_conflict_profile_is_explicit_and_frozen():
    a=s.resolve_preference_acceptance(config())
    assert a['pressure_endpoint_path_rmse_m']==.20
    assert a['endpoint_path_rmse_m']==.15
    c=config();c['preference_acceptance']['max_actual_speed_offset_m_s']=.05
    with pytest.raises(ValueError):s.resolve_preference_acceptance(c)


def test_multiturn_bins_omit_connectors_and_require_every_turn():
    d=trace();d['path_cross_track'][21:39]=999
    m=s._conflict_turn_metrics(d,config())
    assert m['complete'] and len(m['turn_bin_counts'])==2
    assert m['path_rmse_m']==pytest.approx(.1)
    assert m['mean_actual_speed_offset_m_s']==pytest.approx(-.2)
    d['path_progress'][40:]=6
    assert not s._conflict_turn_metrics(d,config())['complete']


def test_multiturn_bins_reject_nonfinite_and_post_failure_padding():
    d=trace();d['actual_forward_speed'][0]=np.nan
    with pytest.raises(ValueError):s._conflict_turn_metrics(d,config())
    d=trace();d['physical_failure']=np.zeros(60,bool);d['physical_failure'][10]=True
    with pytest.raises(ValueError):s._conflict_turn_metrics(d,config())


def test_new_dynamic_timing_keeps_legacy_profile():
    from sttw_control.path_command_dynamic import dynamic_contract
    c=config();c['dynamic_preference_checks']=dict(rate_per_s=.5,first_change_s=1.,return_s=4.,max_lower_ticks=24000)
    c['timing']['upper_dt_s']=.02
    assert dynamic_contract(c)['return_s']==4.
    c['preference_selection_schema']='path_preference_research_v1'
    with pytest.raises(ValueError):dynamic_contract(c)


def conflict_panel():
    from test_path_research_selection import research_panel
    from sttw_control.path_command_scenarios import explicit_case_declarations
    c=config();c['training_sampler']={'version':'path_conflict_exposure_v1'}
    p=research_panel();template=next(iter(p['alpha_0'].values()))
    for row in explicit_case_declarations(c)[6:]:
        for key,a in ((s.alpha_key(a),a) for a in s.CONDITIONAL_ALPHAS):
            d=deepcopy(template);n=len(d['time']);progress=np.linspace(0,30,n)
            d.update(path_progress=progress,upper_alpha=np.full(n,a),v_user=np.full(n,2.8),
                actual_forward_speed=np.full(n,2.8),path_cross_track=np.full(n,.01),
                offsets=np.c_[np.zeros(n),np.zeros(n)],turn_starts=np.array([4.,8.,0.,0.]),
                turn_ends=np.array([7.,11.,0.,0.]),turn_directions=np.array([1.,-1.,0.,0.]),
                turn_mask=np.array([1,1,0,0],bool))
            mask=((progress>=4)&(progress<=7))|((progress>=8)&(progress<=11))
            d['actual_forward_speed'][mask]=2.8-(.22-.12*a)
            d['path_cross_track'][mask]=.1+.04*a;d['offsets'][mask,0]=-.2*(1-a)
            p[key][row['name']]=d
    return c,p


def test_all_60_hold_and_each_pressure_case_are_required_legacy_retained():
    c,p=conflict_panel();result=s.summarize_preference(p,c)
    assert result['qualified_preference'] and result['declared_episodes']==60
    assert len(result['preference_cases'])==4
    assert result['legacy_strict_preference_metrics']['declared_episodes']==30
    d=p['alpha_0']['compact_RL_2.8'];d['offsets'][:,0]=0
    failed=s.summarize_preference(p,c)
    assert not failed['qualified_preference'] and failed['preference_direction_failures']==1
    assert failed['score_tuple']>result['score_tuple']


def test_variable_speed_case_does_not_add_preference_but_must_hold():
    c,p=conflict_panel();d=p['alpha_0p5']['compact_LR_variable']
    d['actual_forward_speed'][100:200]=0
    assert s.summarize_preference(p,c)['qualified_preference']
    d['path_cross_track'][-1]=.21
    assert not s.summarize_preference(p,c)['qualified_preference']
