"""Five-alpha online analysis semantics only, no controller physics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('online14_analysis',Path(__file__).resolve().parents[1]/'cli/analyze_native_goal_confirmation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def raw(alpha,q):
    alpha=np.array(alpha);n=len(alpha);obs=np.zeros((n,351));obs[:,336]=alpha
    return dict(active_tick=np.ones(n,bool),time=np.arange(n)*.005,alpha_exec=alpha,alpha_user=np.full(n,.5),coord_qmin=np.array(q),coord_task_mask=np.ones((n,5),bool),coord_permission=np.ones(n,bool),coord_latched_before=np.ones(n,bool),coord_degraded=np.max(q,axis=1)<.8,channel_alpha_pair=np.column_stack([alpha,alpha]),query_observation=obs)


def test_static_fallback_half_never_counts_as_restored():
    r=raw([.5]*8,np.full((8,5),.1));stats=m.five_alpha_online_statistics(r,dict(onset_s=.02,duration_s=.015),first_arrival_s=.025)
    full=stats['windows']['full'];assert full['fallback_user_ticks']==8 and full['qualified_user_ticks']==0
    assert full['return_to_user_events']==0 and full['qualified_return_events']==0
    assert stats['windows']['before_pulse']['observed_ticks']==4
    assert stats['windows']['before_first_original_goal']['observed_ticks']==4


def test_qualified_return_requires_risk_task_permission():
    q=np.full((8,5),.9);r=raw([.25]*4+[.5]*4,q);stats=m.five_alpha_online_statistics(r,dict(onset_s=.02,duration_s=.015))
    assert stats['windows']['full']['switch_count']==2
    assert stats['windows']['full']['qualified_return_events']==1
    assert stats['qualified_return_times_s']==[.02]
    r['coord_permission'][4:]=False
    stats=m.five_alpha_online_statistics(r,dict(onset_s=.02,duration_s=.015))
    assert stats['windows']['full']['return_to_user_events']==1
    assert stats['windows']['full']['qualified_return_events']==0


def test_low_risk_user_not_relabelled_restored_and_logged_gate_audited():
    q=np.full((4,5),.1);q[:,2]=.83;r=raw([.5]*4,q);r['coord_degraded'][:]=True
    stats=m.five_alpha_online_statistics(r,dict(onset_s=.02,duration_s=.015))
    assert stats['windows']['full']['fallback_user_ticks']==4
    r['coord_degraded'][:]=False
    with pytest.raises(AssertionError):m.five_alpha_online_statistics(r,dict(onset_s=.02,duration_s=.015))


def dwell_fixture(n=48):
    alpha=np.r_[np.full(40,.25),np.full(8,.5)][:n];q=np.full((n,5),.1);q[:4,1]=.9;r=raw(alpha,q)
    tick=(np.arange(n)//4)*4;r.update(coord_proxy_cost=np.zeros((n,5)),coord_dwell_tick=tick,coord_raw_alpha=np.r_[np.full(4,.25),np.full(n-4,.5)],coord_raw_latched=np.ones(n,bool),coord_hold_until_before=np.where(tick==0,0,np.where(tick<=40,40,80)),coord_hold_until_tick=np.where(tick<40,40,80),coord_hold_remaining_ticks=np.where(tick<40,40,80)-tick,coord_hold_release_reason=np.where(tick==40,1,0),coord_held=(tick>0)&(tick!=40),coord_alpha_changed=(tick==0)|(tick==40),rule_latched=np.ones(n,bool),coord_prev_alpha=np.where(tick==0,.5,np.where(tick<=40,.25,.5)))
    r['coord_latched_before']=tick!=0
    return r


def test_hold_survives_q_decline_and_expires_on_native_lower_ticks():
    r=dwell_fixture();a=m.audit_five_alpha_dwell(r,.2)
    assert a['independent_numpy_match']
    assert a['decisions'][1]['held'] and a['decisions'][1]['raw_alpha']==.5 and a['decisions'][1]['executed_alpha']==.25
    assert a['decisions'][10]['task_tick']==40 and a['decisions'][10]['release_reason']==1
    r['coord_hold_until_tick'][4:8]=44
    with pytest.raises(AssertionError):m.audit_five_alpha_dwell(r,.2)


def test_denied_publication_retains_deadline_and_task_mask_bypasses_hold():
    r=dwell_fixture(12);r['coord_permission'][4:8]=False;r['coord_raw_alpha'][4:8]=.25;r['coord_held'][4:8]=False;r['coord_hold_release_reason'][4:8]=3
    m.audit_five_alpha_dwell(r,.2)
    r['coord_task_mask'][8:12,1]=False;r['alpha_exec'][8:12]=.5;r['coord_held'][8:12]=False;r['coord_hold_release_reason'][8:12]=2;r['coord_alpha_changed'][8:12]=True;r['coord_hold_until_tick'][8:12]=48;r['coord_hold_remaining_ticks'][8:12]=40
    a=m.audit_five_alpha_dwell(r,.2)
    assert a['decisions'][1]['hold_until_tick']==40
    assert a['decisions'][2]['release_reason']==2 and a['decisions'][2]['hold_until_tick']==48


@pytest.mark.parametrize('user',[0.,.25,.75,1.])
def test_nominal_nonuser_half_fallback_is_not_user_return(user):
    r=raw([.5]*4+[user]*4,np.r_[np.full((4,5),.1),np.full((4,5),.9)])
    r['alpha_user'][:]=user
    stats=m.five_alpha_online_statistics(r,dict(alpha=user,user_alpha=user,force_N=0.,onset_s=2.2,duration_s=.15))
    full=stats['windows']['full']
    assert full['fallback_half_nonuser_ticks']==4
    assert full['fallback_user_ticks']==0
    assert full['qualified_return_events']==1
    assert full['nonuser_ticks']==4 and full['switch_count']==2
    assert 'nominal_full' in stats['windows'] and 'during_pulse' not in stats['windows']


def test_dwell_uses_original_user_as_initial_alpha():
    r=dwell_fixture(12);r['alpha_user'][:]=0.;r['coord_prev_alpha'][:4]=0.
    audited=m.audit_five_alpha_dwell(r,.2,user_alpha=0.)
    assert audited['independent_numpy_match']
    assert audited['decisions'][0]['executed_alpha']==.25
    assert audited['decisions'][0]['hold_until_tick']==40


def guard_fixture(user=0.,executed=0.,task=None,cost=None):
    r=raw([executed]*4,np.full((4,5),.1));r['alpha_user'][:]=user
    if task is not None:r['coord_task_mask'][:]=task
    tick=np.zeros(4,int);changed=executed!=user;deadline=40 if changed else 0;requested=bool(r['coord_task_mask'][0].any())
    r.update(coord_proxy_cost=np.tile(np.zeros(5) if cost is None else cost,(4,1)),coord_dwell_tick=tick,coord_raw_alpha=np.full(4,executed),coord_raw_latched=np.ones(4,bool),coord_latched_before=np.zeros(4,bool),coord_prev_alpha=np.full(4,user),coord_hold_until_before=tick,coord_hold_until_tick=np.full(4,deadline),coord_hold_remaining_ticks=np.full(4,deadline),coord_hold_release_reason=tick,coord_held=np.zeros(4,bool),coord_alpha_changed=np.full(4,changed),rule_latched=np.ones(4,bool),coord_guard_fallback=np.ones(4,bool),coord_guard_fallback_requested=np.full(4,requested),coord_guard_fallback_selected=np.full(4,requested),coord_guard_fallback_alpha=np.full(4,executed),coord_guard_fallback_published=np.full(4,requested),coord_restore_published=np.zeros(4,bool))
    return r


def test_guard_retains_feasible_user_under_degraded_without_restoring():
    r=guard_fixture();stats=m.five_alpha_online_statistics(r,dict(alpha=0.,force_N=0.,onset_s=2.2,duration_s=.15),dwell_s=.2,guard_fallback=True)
    w=stats['windows']['full']
    assert w['executed_guard_user_ticks']==4 and w['qualified_user_ticks']==0
    assert w['guard_user_without_qualified_restore_ticks']==4
    assert w['fallback_half_ticks']==0 and w['fallback_user_ticks']==4
    assert w['executed_fixed_half_fallback_ticks']==0
    assert stats['dwell_audit']['decisions'][0]['effective_latched']


def test_guard_selects_lowest_cost_task_candidate_or_fixedhalf_without_tasks():
    r=guard_fixture(executed=.75,task=[False,True,False,True,False],cost=[0.,2.,0.,1.,0.])
    a=m.audit_five_alpha_dwell(r,.2,user_alpha=0.,guard_fallback=True)
    assert a['execution_origin_by_tick']==['guard_other']*4
    r=guard_fixture(executed=.5,task=[False]*5)
    a=m.audit_five_alpha_dwell(r,.2,user_alpha=0.,guard_fallback=True)
    assert a['execution_origin_by_tick']==['fixed_half']*4


def test_guard_published_cannot_be_claimed_when_held_or_denied():
    r=guard_fixture();r['coord_permission'][:]=False;r['coord_degraded'][:]=False;r['coord_guard_fallback_selected'][:]=False;r['coord_guard_fallback_published'][:]=False;r['coord_raw_latched'][:]=False;r['rule_latched'][:]=False;r['coord_hold_release_reason']=np.full(4,3)
    a=m.audit_five_alpha_dwell(r,.2,user_alpha=0.,guard_fallback=True)
    assert a['execution_origin_by_tick']==['initial_user']*4
    r['coord_guard_fallback_published'][:]=True
    with pytest.raises(AssertionError):m.audit_five_alpha_dwell(r,.2,user_alpha=0.,guard_fallback=True)


def test_guard_user_half_is_not_actual_fixedhalf_fallback():
    r=guard_fixture(user=.5,executed=.5)
    s=m.five_alpha_online_statistics(r,dict(alpha=.5,force_N=0.,onset_s=2.2,duration_s=.15),dwell_s=.2,guard_fallback=True)['windows']['full']
    assert s['fallback_half_ticks']==4  # Historical degraded occupancy key.
    assert s['executed_fixed_half_fallback_ticks']==0
    assert s['executed_guard_user_ticks']==4 and s['qualified_user_ticks']==0


def test_held_fixedhalf_origin_survives_new_raw_guard_user_request():
    r=guard_fixture(executed=.5,task=[False]*5)
    r={k:np.concatenate([v,v]) for k,v in r.items()}
    r['time']=np.arange(8)*.005;r['coord_dwell_tick'][4:]=4;r['coord_task_mask'][4:]=True;r['coord_raw_alpha'][4:]=0.;r['coord_prev_alpha'][4:]=.5;r['coord_latched_before'][4:]=True;r['coord_hold_until_before'][4:]=40;r['coord_hold_remaining_ticks'][4:]=36;r['coord_held'][4:]=True;r['coord_alpha_changed'][4:]=False
    for k in ['coord_guard_fallback_requested','coord_guard_fallback_selected']:r[k][4:]=True
    r['coord_guard_fallback_alpha'][4:]=0.;r['coord_guard_fallback_published'][4:]=False
    s=m.five_alpha_online_statistics(r,dict(alpha=0.,force_N=0.,onset_s=2.2,duration_s=.15),dwell_s=.2,guard_fallback=True)['windows']['full']
    assert s['raw_guard_user_retention_ticks']==4
    assert s['executed_guard_user_ticks']==0
    assert s['executed_fixed_half_fallback_ticks']==8


def disturbed_panel_fixture():
    import json
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/h75_five_alpha_online_disturbed_users.json').read_text())
    rows=[dict(route=r['route'],alpha=u,user_alpha=u,bank_index=6,force_N=f,onset_s=cfg['onset_by_route_s'][r['route']],duration_s=.15) for r in cfg['route_contexts'] for u in [0.,.25,.5,.75,1.] for f in [-64.,64.]]
    return cfg,rows


def test_disturbed_seventy_exact_product_and_seeded_route_timing():
    cfg,rows=disturbed_panel_fixture();p=m.validate_online_panel(rows,cfg)
    assert p==dict(nominal=False,disturbed_users=True,cases=70)
    assert {r['route']:round(r['onset_s']*50) for r in rows}=={'straight_v2p6':106,'left70_R1p9':120,'right70_R1p9':90,'left130_R2p7':132,'right130_R2p7':98,'S_left_90_R1p8':133,'S_right_90_R1p8':114}
    rows[0]['onset_s']+=.02
    with pytest.raises(ValueError,match='event identity'):m.validate_online_panel(rows,cfg)


def test_disturbed_panel_rejects_missing_sibling_or_global_event_clock():
    cfg,rows=disturbed_panel_fixture();bad=rows[:-1]+[rows[0]]
    with pytest.raises(ValueError,match='product'):m.validate_online_panel(bad,cfg)
    cfg['onset_s']=2.2
    with pytest.raises(ValueError,match='global clock'):m.validate_online_panel(rows,cfg)


def test_phases_use_actual_case_onset_not_historical_two_point_two():
    case=dict(onset_s=1.8,duration_s=.15)
    masks=dict(m.online_phase_masks(np.array([1.79,1.8,1.9,1.95,2.2]),case))
    np.testing.assert_array_equal(masks['before'],[True,False,False,False,False])
    np.testing.assert_array_equal(masks['during'],[False,True,True,False,False])
    np.testing.assert_array_equal(masks['after'],[False,False,False,True,True])


@pytest.mark.parametrize('onset',[1.8,2.12,2.4,2.64,1.96,2.66,2.28])
def test_float32_event_phases_have_exact_thirty_actual_lower_ticks(onset):
    time=np.arange(2000,dtype=np.float32)*np.float32(.005)
    phases=dict(m.online_phase_masks(time,dict(onset_s=onset,duration_s=.15)))
    assert phases['during'].sum()==30
    np.testing.assert_array_equal(np.flatnonzero(phases['during']),np.arange(round(onset/.005),round(onset/.005)+30))
    r=raw([.5]*2000,np.full((2000,5),.1));r['time']=time
    stats=m.five_alpha_online_statistics(r,dict(onset_s=onset,duration_s=.15))
    assert stats['windows']['during_pulse']['observed_ticks']==30
    assert stats['windows']['before_pulse']['observed_ticks']==round(onset/.005)
    assert stats['windows']['after_pulse']['observed_ticks']==2000-round(onset/.005)-30


def soft_fixture():
    r=raw([.5]*4,np.full((4,5),.9));grid=np.array([0.,.25,.5,.75,1.],np.float32);original=.07*(grid-.5)**2;targets=np.tile([2.,0.],(4,5,1)).astype(np.float32);penalty=np.full((4,5),(4/3-1)**2,np.float32)
    r.update(coord_latents=np.zeros((4,5,2),np.float32),coord_nominal=np.tile([2.,0.],(4,1)).astype(np.float32),coord_prev_alpha=np.full(4,.5,np.float32),coord_max_abs_curvature=np.ones(4,np.float32),coord_unbounded_targets=targets.copy(),coord_targets=targets,coord_original_proxy_cost=np.tile(original,(4,1)),coord_proxy_accel=np.full((4,5),4.,np.float32),coord_curvature_excess=np.full((4,5),1/3,np.float32),coord_curvature_excess_penalty=penalty,coord_proxy_cost=np.tile(original,(4,1))+penalty,coord_valid_target_mask=np.ones((4,5),bool),coord_historical_hard_task_mask=np.zeros((4,5),bool),coord_selected_reference_valid=np.ones(4,bool),coord_candidate_reference_fault=np.zeros(4,bool),coord_native_permission=np.ones(4,bool))
    interface=dict(speed_positive_scale_m_s=2.,speed_negative_scale_m_s=1.,steer_scale_rad=.2,speed_reference_bounds_m_s=[0.,4.],steer_reference_bound_rad=.4)
    return r,interface


def test_soft_cost_validity_and_historical_hard_mask_are_independently_audited():
    r,interface=soft_fixture();c=dict(alpha=.5,speed=2.,onset_s=2.2,duration_s=.15)
    stats=m.five_alpha_online_statistics(r,c,curvature_task_mode='soft_penalty',interface=interface)
    assert stats['curvature_task_audit']['independent_numpy_match']
    assert stats['windows']['full']['historical_hard_none_ticks']==4
    assert stats['windows']['full']['valid_reference_any_ticks']==4
    r['coord_curvature_excess_penalty']*=2
    with pytest.raises(AssertionError):m.audit_soft_curvature_task(r,c,interface)


def test_soft_invalid_selected_reference_cannot_claim_permission():
    r,interface=soft_fixture();r['coord_selected_reference_valid'][:]=False
    with pytest.raises(AssertionError):m.audit_soft_curvature_task(r,dict(alpha=.5,speed=2.),interface)


def test_fixed_h75_capability_requires_allfive_alphas_without_zero_baseline():
    routes=['straight','left70','right70','left130','right130','Sleft','Sright'];grid=[0.,.25,.5,.75,1.]
    rows=[dict(route=r,alpha=a,user_alpha=a,query_alpha=a,effective_user_alpha=a,force_N=0.,bank_index=6,mode='H75') for r in routes for a in grid]
    m.validate_fixed_capability_cases(rows,dict(alphas=grid))
    rows[0]['mode']='zero'
    with pytest.raises(ValueError,match='identity'):m.validate_fixed_capability_cases(rows,dict(alphas=grid))
    with pytest.raises(ValueError,match='complete'):m.validate_fixed_capability_cases(rows[:-1],dict(alphas=grid))


def test_fixed_capability_S_extension_requires_exact_two_routes_and_goal_identity():
    grid=[0.,.25,.5,.75,1.];routes=['S_left_90_R1p8','S_right_90_R1p8']
    paths={r:dict(old_goal_s=12.,new_goal_s=15.,old_goal_xy=[12.,0.],new_goal_xy=[15.,0.]) for r in routes}
    protocol=dict(alphas=grid,capability_variant='S_goal_extension',goal_extension_m=3.,expected_cases=10,maximum_lower_ticks=20000,goal_source=dict(paths=paths))
    rows=[dict(route=r,alpha=a,user_alpha=a,query_alpha=a,effective_user_alpha=a,force_N=0.,bank_index=6,mode='H75',goal_extension_m=3.,old_goal_xy=[12.,0.],new_goal_xy=[15.,0.]) for r in routes for a in grid]
    m.validate_fixed_capability_cases(rows,protocol)
    with pytest.raises(ValueError,match='complete'):m.validate_fixed_capability_cases(rows[:-1],protocol)
    rows[0]['old_goal_xy']=[11.,0.]
    with pytest.raises(ValueError,match='goal identity'):m.validate_fixed_capability_cases(rows,protocol)


def test_low_speed_tight_turn_requires_declared_four_routes_and_all_alphas():
    import json
    from pathlib import Path
    protocol=json.loads((Path(__file__).parents[1]/'configs/h75_friction08_low_speed_tight_turn.json').read_text())
    rows=[dict(route=c['route'],alpha=a,mode='H75',force_N=0.,bank_index=6) for c in protocol['route_contexts'] for a in protocol['alphas']]
    m.validate_fixed_capability_cases(rows,protocol)
    with pytest.raises(ValueError,match='complete'):
        m.validate_fixed_capability_cases(rows[:-1],protocol)


def test_low_speed_load_panel_requires_same_forcing_for_both_alphas():
    import json
    from pathlib import Path
    protocol=json.loads((Path(__file__).parents[1]/'configs/h75_low_speed_lateral_load_screen.json').read_text())
    rows=[dict(alpha=a,force_N=f,onset_s=t,bank_index=0,mode='H75',speed=2.,duration_s=.6,user_alpha=a,query_alpha=a,effective_user_alpha=a,route='left90_R0p9_v2p0') for a in (0.,1.) for f,t in [(0.,0.)]+[(f,t) for f in protocol['forces_N'] for t in protocol['onsets_s']]]
    m.validate_fixed_capability_cases(rows,protocol)
    rows[-1]['onset_s']=2.1
    with pytest.raises(ValueError,match='complete paired'):
        m.validate_fixed_capability_cases(rows,protocol)


def test_targeted_base_review_uses_three_alphas_at_low_speed_initial_bank():
    import json
    from pathlib import Path
    protocol=json.loads((Path(__file__).parents[1]/'configs/h75_mixed_u170_targeted_base_review.json').read_text())
    rows=[dict(route=c['route'],alpha=a,mode='H75',force_N=0.,bank_index=0) for c in protocol['route_contexts'] for a in protocol['alphas']]
    m.validate_fixed_capability_cases(rows,protocol)
    rows[0]['bank_index']=6
    with pytest.raises(ValueError,match='identity'):
        m.validate_fixed_capability_cases(rows,protocol)


@pytest.mark.parametrize('low,goal,second,expected',[(True,True,'survived',1),(False,True,'survived',0),(True,False,'survived',0),(True,True,'contact',0)])
def test_load_witness_requires_survival_goal_and_both_actual_low_speed_gates(low,goal,second,expected):
    common=dict(route='turn',event='force',force_N=-12.,onset_s=2.2,duration_s=.6,bank_index=0,model_id='frozen',speed=2.,onset_speed_mean_m_s=1.5)
    rows=[dict(common,alpha=0.,case_id='a0',low_speed_gate_passed=True,endpoint=dict(physical_outcome='contact',goal_arrived=False)),dict(common,alpha=1.,case_id='a1',low_speed_gate_passed=low,endpoint=dict(physical_outcome=second,goal_arrived=goal))]
    assert m.paired_low_speed_load_outcomes(rows)['desired_witnesses']==expected
    rows[1]['force_N']=-16.
    with pytest.raises(ValueError,match='same intended'):m.paired_low_speed_load_outcomes(rows)
