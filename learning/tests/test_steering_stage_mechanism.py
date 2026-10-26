import copy
from types import SimpleNamespace
import numpy as np
import jax.numpy as j
import pytest

from sttw_control.steering_stage_mechanism import stage_cases,make_stage_step,expected_mask,STAGE_ROLES


def test_complete_event_inventory_keeps_all_nominal_and_failed_events():
    original=[dict(role='goal_steer_feedback',case_id=f'c{i}/goal_steer_feedback',
        pulse_onset_task_tick=600,force_N=0 if i%5==0 else -32,source_keep_channels=[False,True],alpha=.5) for i in range(180)]
    before=copy.deepcopy(original);cases=stage_cases(original)
    assert original==before and len(cases)==360 and len({c['case_id'] for c in cases})==360
    assert sum(c['force_N']==0 for c in cases)==72
    assert {c['role'] for c in cases}==set(STAGE_ROLES)
    with pytest.raises(ValueError):stage_cases(original[:-1])


def test_switch_at_exact_upper_boundary_does_not_reset_carry(monkeypatch):
    import sttw_control.goal_preview_baseline as goal
    captured=[]
    def fake(env,actor,**kwargs):
        def advance(carry,parameters,*args):
            captured.append((carry,parameters))
            return carry,dict()
        return advance
    monkeypatch.setattr(goal,'make_goal_step',fake)
    step=make_stage_step(None,None,horizon_s=10.,spec={})
    for late in [False,True]:
        for tick in [596,600,604]:
            carry=SimpleNamespace(state=SimpleNamespace(tick=j.int32(tick)),history=object(),filtered=object())
            parameters=dict(actor=dict(native={'p':j.array([1.])},keep_channels=j.array([False,True])),
                            stage_late_neural=j.bool_(late),switch_task_tick=j.int32(600))
            end,logs=step(carry,parameters,None,None,None,None)
            assert end is carry and captured[-1][0] is carry
            np.testing.assert_array_equal(captured[-1][1]['actor']['keep_channels'],[False,(not late) if tick<600 else late])
            np.testing.assert_array_equal(parameters['actor']['keep_channels'],[False,True])
            assert bool(logs['stage_before'][0])==(tick<600)


def test_independent_stage_log_mask_detects_wrong_switch():
    raw=dict(active_tick=np.ones(8,bool),physical_tick=np.arange(8),
             stage_before=np.arange(8)<4,stage_switch_task_tick=np.full(8,4))
    mask=expected_mask(raw,dict(pulse_onset_task_tick=4,stage_late_neural=True))
    np.testing.assert_array_equal(mask[:,0],False)
    np.testing.assert_array_equal(mask[:,1],np.arange(8)>=4)
    bad=copy.deepcopy(raw);bad['stage_before'][4]=True
    with pytest.raises(AssertionError):expected_mask(bad,dict(pulse_onset_task_tick=4,stage_late_neural=True))
