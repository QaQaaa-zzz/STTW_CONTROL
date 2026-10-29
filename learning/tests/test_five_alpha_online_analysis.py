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
