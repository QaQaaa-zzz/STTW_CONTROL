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
