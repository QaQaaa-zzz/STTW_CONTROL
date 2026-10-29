"""Declared transfer inventory and existing-controller dispatch; zero physics."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
import numpy as np
from sttw_control.native_goal_run import tolerance_inventory, declared_confirmation_methods, select_confirmation_arm, apply_route_context_variant

ROOT=Path(__file__).parents[2]
def protocol():return json.loads((ROOT/'learning/configs/h75_frozen_transfer48.json').read_text())
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'learning/cli'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_complete_transfer_inventory_and_explicit_method_dispatch():
    p=protocol();rows=tolerance_inventory(p)
    assert len(rows)==48 and p['maximum_lower_ticks']==96000
    assert len({(r['bank_index'],r['route'],r['event']) for r in rows})==12
    assert all(r['alpha']==r['query_alpha']==r['effective_user_alpha']==1. for r in rows)
    assert sum(r['force_N']==0 for r in rows)==16
    assert declared_confirmation_methods(rows,p['arms'])==[0,1,2,3]
    p['arms'].reverse();p['method_declarations'].reverse()
    assert declared_confirmation_methods(tolerance_inventory(p),p['arms'])==[3,2,1,0]
    plan=apply_route_context_variant(dict(mode=p['mode'],protocol=p,contexts=[]))
    assert plan['contexts']==p['route_contexts']


@pytest.mark.parametrize('mutation',['unknown','user','count','budget','duplicate','missing_fingerprint'])
def test_invalid_transfer_declaration_rejected(mutation):
    p=protocol()
    if mutation=='unknown':p['method_declarations'][0]['method']='unregistered'
    elif mutation=='user':p['method_declarations'][0]['user_alpha']=.5
    elif mutation=='count':p['expected_cases']+=1
    elif mutation=='budget':p['maximum_lower_ticks']+=2000
    elif mutation=='duplicate':p['bank_indices']=[0,0]
    else:del p['bank_state_fingerprints']['6']
    with pytest.raises(ValueError):tolerance_inventory(p)


def test_initial_layout_and_analyzer_full_single_user_coverage():
    p=protocol();rows=tolerance_inventory(p);m=module('analyze_native_goal_confirmation');audit=module('audit_native_goal_initial')
    layout=m.frozen_confirmation_layout(p)
    assert layout['cases']==48 and layout['conditions']==layout['user_groups']==12
    assert audit.confirmation_initial_layout(dict(mode=p['mode'],protocol=p,cases=48,maximum_active_lower_ticks=96000))==(48,4)
    m.validate_frozen_confirmation_inventory(rows,p)
    with pytest.raises(ValueError):m.validate_frozen_confirmation_inventory(rows[:-1],p)
    for r in rows:
        r.update(horizons={'10':dict(goal_arrived=True,physical_outcome='survived')},alpha_rule=dict(final_at_user=True,nonuser_duration_s=0.,prearrival_alpha=dict(user_duration_s=5.)),matched_original_task_metrics={})
    summary,paired=m.preference_summary(rows,frozen_confirmation=True,frozen_layout=layout)
    assert summary['new_cases']==48 and summary['cached_cases']==0
    assert set(summary['per_user_method'])=={'1.0'} and len(paired)==6
    assert all(len(r['conditions'])==12 for r in paired)
    assert all(v['cases']==12 for v in summary['per_user_method']['1.0'].values())


def test_fixed_and_speedcap_dispatch_ignore_q_proposal_but_keep_permission():
    import jax
    import jax.numpy as j
    choose=jax.jit(select_confirmation_arm)
    decision=dict(alpha=j.array(0.),latched=j.array(True))
    for code in [2,3]:
        a,l=choose(code,decision,True,.5,True,1.)
        assert float(a)==1 and not bool(l)
        a,l=choose(code,decision,False,.5,True,1.)
        assert float(a)==.5 and bool(l)


def test_transfer_rescore_writes_all_method_windows_without_cached_old_pair(tmp_path,monkeypatch):
    m=module('analyze_native_goal_confirmation');p=protocol();rows=tolerance_inventory(p)
    (tmp_path/'analysis').mkdir();(tmp_path/'evaluation').mkdir()
    (tmp_path/'plan.json').write_text(json.dumps(dict(mode=p['mode'])))
    (tmp_path/'analysis/results.json').write_text('[]')
    paths={r['path_key']:{} for r in rows}
    for key in paths:(tmp_path/'evaluation'/f'{key}_path.npz').write_bytes(b'fixture')
    for r in rows:
        r.update(active_lower_ticks=4,prefix=dict(ticks=2),horizons={'10':dict(physical_outcome='survived')})
    monkeypatch.setattr(m,'rescore_original_task',lambda *args,**kwargs: (dict(task_step_costs=np.ones((4,3))),{}))
    results=m.write_original_task_rescoring(tmp_path,rows,{r['case_id']:{} for r in rows},paths,{}, {},[])
    assert len(results)==48
    assert all(r['windows']['common_actual']['actual_ticks']==4 for r in results)
    assert all(r['windows']['common_pre_first_arrival']['actual_ticks']==2 for r in results)
    assert not (tmp_path/'analysis/original_task_rescoring/old_new_candidate_paired_costs.json').exists()
