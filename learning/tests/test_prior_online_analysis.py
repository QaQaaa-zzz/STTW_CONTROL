"""Offline prior-online layout and complete four-method comparison contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

CLI=Path(__file__).parents[1]/'cli'
sys.path.insert(0,str(CLI))
spec=importlib.util.spec_from_file_location('prior_analysis',CLI/'analyze_native_goal_confirmation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_prior_inventory_complete_and_user_identity():
    from sttw_control.native_goal_run import tolerance_inventory
    p=json.loads((CLI.parent/'configs/h75_user_prior_online72.json').read_text())
    rows=tolerance_inventory(p)
    m.validate_prior_inventory(rows,p)
    rows[0]['effective_user_alpha']=.5
    with pytest.raises(ValueError,match='identity'):m.validate_prior_inventory(rows,p)


@pytest.mark.parametrize("methods",[("last","guard","fixed","prior_last"),("last","guard","fixed","prior_last","task_aware")])
def test_complete_summary_keeps_full_denominators(methods):
    rows=[]
    for user in (0.,.5,1.):
        for i in range(24):
            for method in methods:
                rows.append(dict(case_id=f'{user}/{i}/{method}',effective_user_alpha=user,method=method,event=str(i),bank_index=4,route='left',horizons={'10':dict(goal_arrived=False,physical_outcome='contact')},alpha_rule=dict(final_at_user=True,nonuser_duration_s=0.,prearrival_alpha=dict(user_duration_s=1.)),matched_original_task_metrics={}))
    summary,paired=m.preference_summary(rows)
    assert summary['logical_records']==72*len(methods) and summary['cached_cases']==72*(len(methods)-1) and summary['new_cases']==72
    assert len(paired)==3*len(methods)*(len(methods)-1)//2 and all(len(p['conditions'])==24 for p in paired)
    assert summary['per_user_method']['0.0']['prior_last']['contact']==24
    with pytest.raises(ValueError):m.preference_summary(rows[:-1])


def test_task_aware_inventory_same_bank_and_user_contract():
    from sttw_control.native_goal_run import tolerance_inventory
    p=json.loads((CLI.parent/'configs/h75_task_aware_fallback72.json').read_text())
    rows=tolerance_inventory(p)
    m.validate_prior_inventory(rows,p,task_aware=True)
    assert len(rows)==72 and {r['method'] for r in rows}=={'task_aware'}
    rows[0]['method']='prior_last'
    with pytest.raises(ValueError,match='identity'):m.validate_prior_inventory(rows,p,task_aware=True)


def test_degraded_user_execution_is_not_qualified_restoration():
    import numpy as np
    raw=dict(active_tick=np.ones(8,bool),alpha_exec=np.zeros(8),coord_degraded=np.ones(8,bool),coord_publication_withheld=np.zeros(8,bool),coord_latched_before=np.ones(8,bool),coord_qmin=np.full((8,3),.82),coord_task_mask=np.ones((8,3),bool),coord_permission=np.ones(8,bool),coord_task_aware_selected=np.ones(8,bool))
    stats=m.tolerance_alpha_statistics(raw,0.)
    assert stats['final_at_user'] and not stats['final_qualified_user_restoration']
    assert not stats['final_risk_gate_satisfied'] and stats['final_user_restore_threshold']==.85
    assert stats['user_execution_under_degraded_s']==.04
    assert stats['fallback_selected_user_without_restore_s']==.04
    raw['coord_latched_before'][:]=False
    stats=m.tolerance_alpha_statistics(raw,0.)
    assert stats['final_qualified_user_restoration'] and stats['final_risk_gate_satisfied']
    assert stats['fallback_selected_user_without_restore_s']==0.
    raw['coord_permission'][-1]=False
    assert not m.tolerance_alpha_statistics(raw,0.)['final_qualified_user_restoration']
    raw['coord_permission'][-1]=True;raw['coord_task_mask'][-1,0]=False
    assert not m.tolerance_alpha_statistics(raw,0.)['final_qualified_user_restoration']


def test_frozen_confirmation_nonuniform_methods_summary_and_windows():
    import numpy as np
    rows=[];raws={};paths={};goals={}
    for user in (0.,.5,1.):
        for i in range(20):
            for method in ('task_aware','guard','fixed')+(('speedcap',) if user==1. else ()):
                case=f'{user}/{i}/{method}';n=4 if method=='speedcap' else 8
                row=dict(case_id=case,effective_user_alpha=user,alpha=user,method=method,event=str(i),bank_index=1,route='left',path_key='p',speed=2.8,active_lower_ticks=n,prefix={'ticks':n},diagnostic_execution_path_user1_cost=0.,horizons={'10':dict(goal_arrived=False,physical_outcome='contact')},alpha_rule=dict(final_at_user=True,nonuser_duration_s=0.))
                rows.append(row);raws[case]=dict(alpha_exec=np.full(n,user),actual_forward_speed=np.full(n,2.8),actual_xy=np.zeros((n,2)))
    paths['p']=dict(s=np.array([0.,1.]),goal=1.,xy=np.array([[0.,0.],[1.,0.]]));goals['p']=np.array([1.,0.])
    m.preference_common_windows(rows,raws,paths,goals)
    assert all(r['matched_window_method_count']==(4 if r['effective_user_alpha']==1. else 3) for r in rows)
    assert all(r['matched_original_task_metrics']['same_actual_window']['actual_ticks']==(4 if r['effective_user_alpha']==1. else 8) for r in rows)
    summary,paired=m.preference_summary(rows,frozen_confirmation=True)
    assert summary['new_cases']==200 and summary['cached_cases']==0
    assert len(paired)==12 and all(len(p['conditions'])==20 for p in paired)
    assert 'speedcap' not in summary['per_user_method']['0.0']


def test_unused_model_gate_is_diagnostic_only_for_baselines():
    import numpy as np
    raw=dict(active_tick=np.ones(4,bool),alpha_exec=np.ones(4),coord_degraded=np.zeros(4,bool),coord_publication_withheld=np.zeros(4,bool),coord_latched_before=np.zeros(4,bool),coord_qmin=np.ones((4,3)),coord_task_mask=np.ones((4,3),bool),coord_permission=np.ones(4,bool))
    stats=m.tolerance_alpha_statistics(raw,1.,controller_uses_risk_gate=False)
    assert stats['final_at_user'] and stats['final_qualified_user_restoration'] is None
    assert stats['diagnostic_only_final_user_at_risk_gate'] is True
    assert stats['restoration_gate_used_by_controller'] is False


def test_frozen_confirmation_inventory_and_initial_layout():
    from sttw_control.native_goal_run import tolerance_inventory
    p=json.loads((CLI.parent/'configs/h75_frozen_confirmation200.json').read_text())
    rows=tolerance_inventory(p)
    m.validate_frozen_confirmation_inventory(rows,p)
    assert len(rows)==200
    spec=importlib.util.spec_from_file_location('initial_audit_frozen',CLI/'audit_native_goal_initial.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    assert audit.confirmation_initial_layout(dict(mode='frozen_confirmation200',protocol=p))==(200,10)
    rows[0]['method']='speedcap'
    with pytest.raises(ValueError,match='identity'):m.validate_frozen_confirmation_inventory(rows,p)


def test_speedcap_publication_preserves_steering_and_independent_speed_audit():
    import numpy as np
    n=4;candidate=np.array([.2,.3]);composed=np.array([.1,.3])
    raw=dict(active_tick=np.ones(n,bool),alpha_exec=np.ones(n),alpha_user=np.ones(n),reference_target_permitted=np.ones(n,bool),coord_candidate_alpha=np.tile([0.,.5,1.],(n,1)),channel_alpha_pair=np.ones((n,2)),query_observation=np.pad(np.ones((n,1)),((0,0),(336,14))),latent_z=np.tile(composed,(n,1)),composed_latent=np.tile(composed,(n,1)),rule_latched=np.zeros(n,bool),coord_latents=np.tile(candidate,(n,3,1)),coord_permission=np.ones(n,bool))
    case=dict(mode='user1_speedcap',method='speedcap',alpha=1.)
    assert m.is_speedcap_case(case)
    m.audit_tolerance_publication(raw,case)
    with pytest.raises(AssertionError):m.audit_tolerance_publication(raw,dict(mode='user1_fixed',method='fixed',alpha=1.))
    raw['composed_latent'][:]=candidate;raw['latent_z'][:]=candidate
    m.audit_tolerance_publication(raw,dict(mode='user1_fixed',method='fixed',alpha=1.))
    assert m.is_speedcap_case(dict(mode='speedcap'))


def test_speed_query_summary_has_260_records_and_variable_user_methods():
    rows=[]
    for user in (0.,.5,1.):
        methods=('task_aware','guard','fixed','speed_query')+(('speedcap',) if user==1. else ())
        for i in range(20):
            for method in methods:
                rows.append(dict(case_id=f'{user}/{i}/{method}',effective_user_alpha=user,method=method,event=str(i),bank_index=1,route='left',horizons={'10':dict(goal_arrived=True,physical_outcome='complete')},alpha_rule=dict(final_at_user=True,nonuser_duration_s=0.,prearrival_alpha=dict(user_duration_s=1.)),matched_original_task_metrics={}))
    summary,paired=m.preference_summary(rows,speed_comparison=True)
    assert summary['new_cases']==60 and summary['cached_cases']==200 and summary['logical_records']==260
    assert len(paired)==22 and all(len(x['conditions'])==20 for x in paired)
    assert 'speedcap' not in summary['per_user_method']['0.0']
    with pytest.raises(ValueError):m.preference_summary(rows[:-1],speed_comparison=True)


@pytest.mark.parametrize('candidate,prefix',[('speed_query','full_mask'),('guard_degraded','old_new_candidate')])
def test_full_mask_original_cost_windows_are_separate_from_all_methods(tmp_path,monkeypatch,candidate,prefix):
    import numpy as np
    (tmp_path/'analysis').mkdir();(tmp_path/'evaluation').mkdir()
    (tmp_path/'plan.json').write_text(json.dumps(dict(mode='speed_query_online60')))
    (tmp_path/'analysis/results.json').write_text('[]')
    np.savez(tmp_path/'evaluation/p_path.npz',s=np.array([0.,1.]))
    rows=[];raws={};path=dict(s=np.array([0.,1.]),goal=1.,xy=np.array([[0.,0.],[1.,0.]]));paths={'p':path};goals={'p':np.array([1.,0.])}
    for method,n in (('guard',2),('fixed',8),('task_aware',8),('speedcap',8),(candidate,6)):
        rows.append(dict(case_id=method,method=method,mode='user1_'+method,effective_user_alpha=1.,alpha=1.,event='zero',bank_index=1,route='left',path_key='p',speed=2.8,active_lower_ticks=n,prefix={'ticks':n},diagnostic_execution_path_user1_cost=0.,horizons={'10':dict(physical_outcome='complete',goal_arrived=False)} ,alpha_rule={}))
        raws[method]=dict(alpha_exec=np.ones(n),actual_forward_speed=np.full(n,2.8),actual_xy=np.zeros((n,2)))
    m.preference_common_windows(rows,raws,paths,goals);m.paired_candidate_common_windows(rows,raws,paths,goals,candidate=candidate,metric_key='candidate_pair_metrics')
    assert next(r for r in rows if r['method']=='task_aware')['matched_original_task_metrics']['same_actual_window']['actual_ticks']==2
    assert next(r for r in rows if r['method']=='task_aware')['candidate_pair_metrics']['same_actual_window']['actual_ticks']==6
    monkeypatch.setattr(m,'rescore_original_task',lambda raw,*a,**k: (dict(task_step_costs=np.ones((len(raw['alpha_exec']),3))),{}))
    results=m.write_original_task_rescoring(tmp_path,rows,raws,paths,{},None,[])
    for method in ('task_aware',candidate):
        result=next(r for r in results if r['case_id']==method)
        assert result['windows']['common_pre_first_arrival']['actual_ticks']==2
        assert result['windows'][prefix+'_common_pre_first_arrival']['actual_ticks']==6
        assert result['windows'][prefix+'_common_pre_first_arrival']['three_task_cost_sum']==18.


def test_speed_query_inventory_matches_new60_config():
    from sttw_control.native_goal_run import tolerance_inventory
    p=json.loads((CLI.parent/'configs/h75_speed_query_online60.json').read_text())
    rows=tolerance_inventory(p)
    m.validate_speed_query_inventory(rows,p)
    assert len(rows)==60 and len(m.response_plot_groups(rows))==60
    rows[0]['method']='task_aware'
    with pytest.raises(ValueError,match='identity'):m.validate_speed_query_inventory(rows,p)


def geometry_duration_protocol():
    p=json.loads((CLI.parent/'configs/h75_frozen_confirmation200.json').read_text())
    p.update(variant='geometry_duration_extension_v1',expected_cases=240,maximum_lower_ticks=480000,onsets_s=[2.6],zero_onset_s=2.6,duration_s=.15,routes=[dict(route=f'S_{direction}_{angle}_R2p2',speed=2.6) for angle in (60,120) for direction in ('left','right')])
    return p


def test_frozen_geometry_variant_derived_counts_and_speed_identity():
    import itertools
    p=geometry_duration_protocol();layout=m.frozen_confirmation_layout(p)
    assert layout==dict(variant='geometry_duration_extension_v1',conditions=24,cases=240,routes=4,user_groups=72,nominal_cases=80)
    rows=[]
    for bank,route,arm,(force,onset) in itertools.product(p['bank_indices'],p['routes'],p['arms'],[(0.,p['zero_onset_s'])]+list(itertools.product(p['forces_N'],p['onsets_s']))):
        user=0. if arm.startswith('user0_') else .5 if arm.startswith('userhalf_') else 1.
        method={'taskaware':'task_aware','guard':'guard','fixed':'fixed','speedcap':'speedcap'}[arm.rsplit('_',1)[1]]
        rows.append(dict(case_id=f'{bank}/{route["route"]}/{arm}/{force}',bank_index=bank,route=route['route'],speed=route['speed'],mode=arm,alpha=user,user_alpha=user,effective_user_alpha=user,query_alpha=user,method=method,force_N=force,onset_s=onset,duration_s=p['duration_s'],initial_fingerprint=p['bank_state_fingerprints'][str(bank)],path_key=f'bank{bank}_'+route['route']))
    m.validate_frozen_confirmation_inventory(rows,p)
    rows[0]['speed']=2.8
    with pytest.raises(ValueError,match='complete frozen confirmation conditions'):m.validate_frozen_confirmation_inventory(rows,p)
    p.pop('variant')
    with pytest.raises(ValueError,match='legacy'):m.frozen_confirmation_layout(p)
    p['variant']='unregistered_extension'
    with pytest.raises(ValueError,match='unknown'):m.frozen_confirmation_layout(p)


def test_frozen_geometry_summary_and_report_cover_all24_conditions():
    p=geometry_duration_protocol();layout=m.frozen_confirmation_layout(p);rows=[]
    for user in (0.,.5,1.):
        for i in range(24):
            for method in ('task_aware','guard','fixed')+(('speedcap',) if user==1. else ()):
                rows.append(dict(case_id=f'{user}/{i}/{method}',effective_user_alpha=user,method=method,event=str(i),bank_index=1,route='left',horizons={'10':dict(goal_arrived=False,physical_outcome='contact')},alpha_rule=dict(final_at_user=False,nonuser_duration_s=1.,prearrival_alpha=dict(user_duration_s=0.)),matched_original_task_metrics={}))
    summary,paired=m.preference_summary(rows,frozen_confirmation=True,frozen_layout=layout)
    assert summary['new_cases']==240 and summary['logical_records']==240 and summary['cached_cases']==0
    assert len(paired)==12 and all(len(x['conditions'])==24 for x in paired)
    assert summary['per_user_method']['0.0']['task_aware']['contact']==24
    assert summary['protocol_variant']=='geometry_duration_extension_v1'
    intro='\n'.join(m.frozen_confirmation_intro(p,layout))
    assert '240' in intro and '0.15' in intro and '0.1s' in intro and '不称新独立初态' in intro
    assert 'S_left_120_R2p2' in intro and 'R1.8' not in intro
    with pytest.raises(ValueError):m.preference_summary(rows[:-1],frozen_confirmation=True,frozen_layout=layout)


def test_legacy_frozen_protocol_remains200_and_rejects_unknown_variant():
    p=json.loads((CLI.parent/'configs/h75_frozen_confirmation200.json').read_text());layout=m.frozen_confirmation_layout(p)
    assert layout['cases']==200 and layout['conditions']==20 and layout['user_groups']==60
    p['expected_cases']=240
    with pytest.raises(ValueError,match='declared'):m.frozen_confirmation_layout(p)


def guard_stage_protocol(stage):
    p=geometry_duration_protocol() if stage=='extension' else json.loads((CLI.parent/'configs/h75_frozen_confirmation200.json').read_text())
    p.update(stage=stage,expected_cases=72 if stage=='extension' else 60,arms=list(m.GUARD_FALLBACK_ARMS))
    return p


@pytest.mark.parametrize('stage,new_count,logical,conditions',[('extension',72,312,24),('regression',60,260,20)])
def test_guard_fallback_stage_layout_summary_and_rescue_gate(stage,new_count,logical,conditions):
    import copy
    p=guard_stage_protocol(stage);layout=m.guard_fallback_layout(p)
    assert layout['new_cases']==new_count and layout['logical_cases']==logical
    rows=[]
    for user in (0.,.5,1.):
        for i in range(conditions):
            for method in ('task_aware','guard','fixed','guard_degraded')+(('speedcap',) if user==1. else ()):
                goal=False;outcome='survived';ticks=2000
                if method=='task_aware' and i==0:outcome='contact';ticks=600
                if method=='guard_degraded' and i==0:goal=True
                if user==.5 and i==1:
                    if method=='task_aware':goal=True
                    if method=='guard_degraded':outcome='contact';ticks=650
                rows.append(dict(case_id=f'{user}/{i}/{method}',effective_user_alpha=user,method=method,event=str(i),bank_index=1,route='left',active_lower_ticks=ticks,horizons={'10':dict(goal_arrived=goal,physical_outcome=outcome)},alpha_rule=dict(final_at_user=True,nonuser_duration_s=0.,prearrival_alpha=dict(user_duration_s=1.)),matched_original_task_metrics={}))
    summary,paired=m.preference_summary(rows,guard_layout=layout)
    assert summary['new_cases']==new_count and summary['cached_cases']==logical-new_count
    assert len(paired)==22 and all(len(x['conditions'])==conditions for x in paired)
    gate=summary['new_vs_old'];assert gate['any_rescue'] and gate['total_goal_gains']==3 and gate['total_contact_rescues']==3
    assert gate['total_goal_losses']==gate['total_new_contacts']==1
    changed=copy.deepcopy(rows)
    for r in changed:
        if r['method']=='guard_degraded' and r['event']=='0':r['horizons']['10']=dict(goal_arrived=None,physical_outcome='unknown')
    assert not m.new_vs_old_outcomes(changed,conditions)['any_rescue']
    with pytest.raises(ValueError):m.preference_summary(rows[:-1],guard_layout=layout)


def test_guard_degraded_numpy_audit_keeps_risk_latch_and_honors_permission(monkeypatch):
    import numpy as np
    n=8;user=.5;probs=np.tile(np.array([.6,.3,.2])[:,None],(1,4));targets=np.tile([2.8,0.],(3,1));task=np.ones(3,bool)
    cost=.07*(np.array([0.,.5,1.])-user)**2
    raw=dict(active_tick=np.ones(n,bool),alpha_exec=np.full(n,user),query_observation=np.zeros((n,351)),coord_latents=np.zeros((n,3,2)),pre_upper_yaw=np.zeros(n),coord_probabilities=np.tile(probs,(n,1,1)),coord_qmin=np.tile(probs[:,0],(n,1)),coord_pre_progress=np.zeros(n),path_progress=np.zeros(n),coord_latched_before=np.r_[np.zeros(4,bool),np.ones(4,bool)],coord_prev_alpha=np.full(n,user),coord_nominal=np.tile([2.8,0.],(n,1)),coord_targets=np.tile(targets,(n,1,1)),coord_unbounded_targets=np.tile(targets,(n,1,1)),coord_candidate_clipped=np.zeros((n,3,2),bool),coord_max_abs_curvature=np.full(n,.1),coord_task_mask=np.tile(task,(n,1)),coord_proxy_cost=np.tile(cost,(n,1)),coord_permission=np.r_[np.ones(4,bool),np.zeros(4,bool)],coord_degraded=np.r_[np.ones(4,bool),np.zeros(4,bool)],coord_publication_withheld=np.r_[np.zeros(4,bool),np.ones(4,bool)],coord_guard_fallback=np.ones(n,bool),coord_guard_fallback_selected=np.r_[np.ones(4,bool),np.zeros(4,bool)],coord_task_aware_selected=np.zeros(n,bool),rule_latched=np.ones(n,bool))
    # Saved proxy includes both user preference and previous-action switch terms.
    monkeypatch.setattr(m,'tolerance_numpy_probabilities',lambda *a:(np.tile(probs,(2,1,1)),np.zeros((2,3,2))))
    interface=dict(speed_reference_bounds_m_s=[1.,3.],steer_reference_bound_rad=.3)
    case=dict(mode='userhalf_guardfallback',method='guard_degraded',effective_user_alpha=user,speed=2.8)
    result=m.audit_tolerance_coordination(raw,case,dict(s=np.array([0.,10.]),curvature=np.full(2,.1)),None,interface,{},arm_index=0)
    assert result['final_at_user'] and not result['final_qualified_user_restoration']
    assert result['guard_fallback_selected_user_without_restore_s']==.02
    raw['coord_guard_fallback_selected'][4]=True
    with pytest.raises(AssertionError):m.audit_tolerance_coordination(raw,case,dict(s=np.array([0.,10.]),curvature=np.full(2,.1)),None,interface,{},arm_index=0)
