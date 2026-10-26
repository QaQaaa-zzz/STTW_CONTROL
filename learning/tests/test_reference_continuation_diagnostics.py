import numpy as np
import pytest

from sttw_control.reference_continuation_diagnostics import trace_diagnostics


def data(gate, legal=None):
    n=len(gate)
    return dict(active_tick=np.ones(n,bool), physical_tick=np.arange(n),
                relay_active=np.array(gate,bool), actor_action_valid=np.ones(n,bool) if legal is None else np.array(legal,bool),
                phi_dot=np.arange(n)*.1), dict(path_error=np.arange(n)*.2, heading_error=np.arange(n)*.3)


def test_release_requires_observed_following_tick_and_keeps_all_bouts():
    raw,series=data([False,True,False,True])
    d=trace_diagnostics(raw,series,onset_tick=2,dt=.005,horizon_ticks=4)
    assert d['support_bouts']==2 and not d['release_observed']
    assert d['pre_pulse']['time_s']==.01
    assert d['first_support_preceding_pose']['end_of_tick']==0
    assert d['last_support_pose']['later_actor_legal_fraction'] is None
    raw,series=data([False,True,False,False], [True,True,False,True])
    d=trace_diagnostics(raw,series,onset_tick=2,dt=.005,horizon_ticks=4)
    assert d['release_observed'] and d['last_support_pose']['later_actor_legal_fraction']==.5
    assert d['legal_support_s']==.005


def test_never_triggered_is_not_a_release_and_partial_is_not_complete():
    raw,series=data([False,False])
    d=trace_diagnostics(raw,series,onset_tick=5,dt=.005,horizon_ticks=4)
    assert not d['complete_horizon'] and d['pre_pulse'] is None
    assert d['last_support_pose'] is None and not d['release_observed']
    assert d['post_pulse_actor_legal_fraction'] is None
    raw['physical_tick']=np.array([0,2])
    with pytest.raises(ValueError,match='clock'):trace_diagnostics(raw,series,onset_tick=1,dt=.005,horizon_ticks=4)
