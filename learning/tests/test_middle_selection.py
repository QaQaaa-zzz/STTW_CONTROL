"""Model-selection contract and causal repeated-reconnection boundary behavior."""
import copy
import json
from pathlib import Path
import pytest
from sttw_control.native_goal_run import middle_selection_inventory, planning_chunk_steps
from sttw_control.goal_replanning import replan_requested


def protocol():return json.loads((Path(__file__).parents[1]/'configs/middle_selection.json').read_text())


def test_selection_is_three_named_single_networks_five_users_three_path_families():
    p=protocol();rows=middle_selection_inventory(p)
    assert len(rows)==336
    assert sum(c['mode']=='zero' for c in rows)==21
    for mode in ['H75','W400','u325']:
        assert sum(c['mode']==mode for c in rows)==105
        assert {c['alpha'] for c in rows if c['mode']==mode}=={0.,.25,.5,.75,1.}
    assert len({c['case_id'] for c in rows})==336
    assert {c['family'] for c in p['route_contexts']}=={0,1,2}


@pytest.mark.parametrize('mutation',['dual','missing_alpha','budget','duplicate_route'])
def test_selection_rejects_mislabeled_or_incomplete_comparison(mutation):
    p=protocol()
    if mutation=='dual':p['models']['W400']='STTW_MIDDLE_DUAL_COVERAGE_U200_A0'
    if mutation=='missing_alpha':p['alphas'].remove(.25)
    if mutation=='budget':p['maximum_lower_ticks']-=2000
    if mutation=='duplicate_route':p['route_contexts'].append(copy.deepcopy(p['route_contexts'][0]))
    with pytest.raises(ValueError):middle_selection_inventory(p)


def test_ten_and_fifty_hz_keep_exact_physical_horizon():
    assert planning_chunk_steps(10.,10.)==[5]*100
    assert planning_chunk_steps(50.,10.)==[1]*500
    assert sum(planning_chunk_steps(10.,10.))*4==2000
    with pytest.raises(ValueError):planning_chunk_steps(30.,10.)


def test_continuous_reconnection_uses_departure_then_latches_until_goal_or_failure():
    args=dict(xy=[0.,0.],goal_xy=[20.,0.],cross_track_m=0.,follower_invalid=False,already_replanned=False,contact=False,fault=False,continuous_replanning=True)
    assert not replan_requested(**args)
    assert replan_requested(**dict(args,cross_track_m=1.1))
    assert replan_requested(**dict(args,already_replanned=True))
    assert not replan_requested(**dict(args,already_replanned=True,continuous_replanning=False))
    for state in [dict(contact=True),dict(fault=True),dict(ever_arrived=True),dict(xy=[19.5,0.])]:
        assert not replan_requested(**dict(args,already_replanned=True,**state))
    with pytest.raises(ValueError):replan_requested(**dict(args,scheduled_time_s=2.8,task_time_s=2.8))


def cli(name):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'cli'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_selection_initial_audit_layout_accepts_all_physical_lanes():
    m=cli('audit_native_goal_initial');p=protocol()
    assert m.confirmation_initial_layout(dict(mode='middle_selection',protocol=p,cases=336))==(336,16)


def test_repeated_path_reward_uses_each_actual_path_without_resetting_failure_clock(monkeypatch):
    import numpy as np
    import sys,types
    m=cli('analyze_native_goal_confirmation')
    def score(raw,path,cfg,controller,keys,alpha):
        return {},dict(effective=np.full((len(raw['active_tick']),1),path['cost']))
    monkeypatch.setitem(sys.modules,'analyze_native_frozen_validation',types.SimpleNamespace(score=score))
    effective=np.array([1.,1.,2.,2.,3.,3.,3.,3.])[:,None]
    raw=dict(active_tick=np.ones(8,bool),effective_costs=effective,physical_failure=np.zeros(8,bool),scored_tick_reward=-effective[:,0]*.005)
    cfg=dict(path_reward=dict(scale=1.))
    result=m.reconstruct_execution_path_reward(raw,dict(alpha=.25),dict(cost=1.),None,8,cfg,{},['speed'],path_updates=[(2,dict(cost=2.)),(4,dict(cost=3.))])
    np.testing.assert_allclose(result['reward'],raw['scored_tick_reward'])
    with pytest.raises(ValueError):m.reconstruct_execution_path_reward(raw,dict(alpha=.25),dict(cost=1.),None,8,cfg,{},['speed'],path_updates=[(4,dict(cost=2.)),(2,dict(cost=3.))])
