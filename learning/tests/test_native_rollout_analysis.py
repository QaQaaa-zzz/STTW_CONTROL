"""First arrival must precede native stop; terminal replacement is not running cost."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec=importlib.util.spec_from_file_location('native_analysis',Path(__file__).parents[1]/'cli/analyze_native_reference_rollout.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def trace():
    n=8
    return dict(active_tick=np.ones(n,bool),time=np.arange(n)*.005,
        actual_xy=np.column_stack([np.arange(n),np.zeros(n)]),
        policy_fault=np.zeros(n,bool),lower_fault=np.zeros(n,bool),
        physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool),
        effective_costs=np.ones((n,3)),scored_tick_reward=np.full(n,-.0015),
        path_cross_track=np.zeros(n),actual_forward_speed=np.ones(n),v_user=np.ones(n))


def score(raw,goal):
    return module.summarize(raw,np.array(goal),scale=.1,dt=.005,keys=['speed','path','primary'])


def test_goal_after_stop_is_not_a_success_and_penalty_is_not_double_counted():
    raw=trace();raw['domain_exit'][-1]=True
    raw['scored_tick_reward'][4:]=[0,0,0,-10]
    row=score(raw,[7.9,0])
    assert not row['goal1m_before_native_stop']
    assert row['native_domain_exit'] and not row['native_physical_failure']
    assert row['scored_cost']==pytest.approx(10.006)
    assert row['running_cost']==pytest.approx(.012)
    assert row['terminal_replacement_adjustment']==pytest.approx(9.994)


def test_arrival_before_physical_stop_is_retained_without_hold_veto():
    raw=trace();raw['physical_failure'][-1]=True
    raw['scored_tick_reward'][4:]=[0,0,0,-10]
    row=score(raw,[3,0])
    assert row['goal1m_before_native_stop'] and row['native_physical_failure']
    assert row['first_arrival_s']==.01


def test_inactive_padding_cannot_be_silently_scored():
    raw=trace();raw['active_tick'][-1]=False
    with pytest.raises(ValueError,match='active ticks'):score(raw,[3,0])
