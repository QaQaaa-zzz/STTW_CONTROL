"""Counterexamples for task ranking, using finite saved-trace-shaped arrays only."""
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.path_command_selection import (cases,summarize,summarize_task,
    task_score_key,persist_task_best)
from sttw_control.path_command_policy import SCHEMA

CFG=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Feedback_V1.json').read_text())

def panel(ey=.01,ev=.01,goal=True):
    n=4000
    def trace():
        return dict(time=np.arange(n)*.005,physical_failure=np.zeros(n,bool),
            policy_fault=np.zeros(n,bool),lower_fault=np.zeros(n,bool),domain_exit=np.zeros(n,bool),
            peak_roll=np.full(n,.1),actual_forward_speed=np.full(n,2.+ev),v_user=np.full(n,2.),
            path_cross_track=np.full(n,ey),path_heading_error=np.zeros(n),chi=np.ones(n),
            goal_section_signed_distance=np.full(n,1. if goal else -1.),
            path_progress=np.full(n,20.),goal_progress=np.full(n,10.))
    return {name:trace() for name,_,_ in cases(CFG)}

def test_task_completion_beats_subtolerance_tiny_primary_improvement(tmp_path):
    incomplete=summarize_task(panel(ey=.01,goal=False),0,CFG)
    complete=summarize_task(panel(ey=.0100001),0,CFG)
    assert task_score_key(complete)<task_score_key(incomplete)
    assert complete['qualified_task'] and complete['qualified_preference'] is None
    assert not incomplete['qualified_task']
    # Historical continuous ordering remains available and is intentionally different.
    assert summarize(panel(ey=.01,goal=False),0,CFG)['score_tuple']<summarize(panel(ey=.0100001),0,CFG)['score_tuple']
    legacy=tmp_path/'best_model.json';legacy.write_text('{"untouched":true}')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'first learner')
    actor=tmp_path/'actor.pkl';actor.write_bytes(b'first actor')
    identity={'schema':SCHEMA,'lower_actor_sha256':'local300','update':100}
    persist_task_best(tmp_path,checkpoint,actor,incomplete,identity)
    checkpoint.write_bytes(b'second learner');actor.write_bytes(b'second actor')
    best=persist_task_best(tmp_path,checkpoint,actor,complete,{**identity,'update':150})
    assert best['update']==150 and best['qualified_task']
    assert legacy.read_text()=='{"untouched":true}'
    with pytest.raises(ValueError,match='identity'):
        persist_task_best(tmp_path,checkpoint,actor,complete,{**identity,'lower_actor_sha256':'400'})
    Path(best['actor']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='SHA'):
        persist_task_best(tmp_path,checkpoint,actor,complete,identity)

def test_ordinary_primary_is_common_speed_and_path_target():
    traces=panel(ey=.02,ev=.09)
    for d in traces.values():d['chi'][:]=0
    a=summarize_task(traces,0,CFG);b=summarize_task(traces,1,CFG)
    assert a['primary_score']==pytest.approx(.9)
    assert a['score_tuple']==b['score_tuple']
    for d in traces.values():d['actual_forward_speed'][:]=2.11
    assert summarize_task(traces,0,CFG)['primary_band_failures']==6

def test_finite_domain_is_rankable_failure_but_missing_frames_or_nan_are_not():
    traces=panel();name='left90_R2_2.6'
    traces[name]={k:v[:19].copy() for k,v in traces[name].items()}
    traces[name]['domain_exit'][-1]=True
    result=summarize_task(traces,0,CFG)
    assert result['tracking_domain_failures']==1 and result['physical_failures']==0
    assert result['task_failures']==1 and not result['qualified_task']
    assert summarize(traces,0,CFG)['score_tuple'][0]==0  # legacy physical-only ordering preserved
    traces[name]['physical_failure'][-1]=True
    assert summarize_task(traces,0,CFG)['task_failures']==1
    traces[name]['domain_exit'][-1]=False;traces[name]['physical_failure'][-1]=False
    with pytest.raises(ValueError,match='partial'):summarize_task(traces,0,CFG)
    traces[name]['domain_exit'][-1]=True;traces[name]['path_cross_track'][-1]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):summarize_task(traces,0,CFG)
    traces=panel();traces[name]['time'][10]+=.005
    with pytest.raises(ValueError,match='frames'):summarize_task(traces,0,CFG)
