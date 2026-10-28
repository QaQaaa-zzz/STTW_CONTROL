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
