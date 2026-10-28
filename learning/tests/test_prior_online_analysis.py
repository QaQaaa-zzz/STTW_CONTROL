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
