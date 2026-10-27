"""Declared experimental population and recording provenance; no physics."""
import copy
import numpy as np
import pytest
from sttw_control.goal_training_run import training_cases,episodes_from_logs


def reference():
    return [dict(context=ctx,profile=profile,speed=speed,alpha=.5,force_N=force,
                 pulse_onset_task_tick=480,pulse_duration_s=.1,world_direction=[1.,0.,0.])
            for ctx,speed in [('left',1.8),('right',1.5)]
            for force in (0.,-32.,32.) for profile in (0,7)]


def test_complete_stratification_and_fresh_hidden_onsets_without_editing_reference():
    ref=reference();before=copy.deepcopy(ref)
    one=training_cases(ref,0,2026102727);two=training_cases(ref,1,2026102727)
    assert ref==before and one==training_cases(ref,0,2026102727)
    assert len(one)==36 and len({c['episode_id'] for c in one+two})==72
    for ctx in ['left','right']:
        for alpha in (0.,.5,1.):
            for force in (0.,-32.,32.):
                assert len([c for c in one if (c['context'],c['alpha'],c['force_N'])==(ctx,alpha,force)])==2
    assert all(c['profile']==7 and 400<=c['pulse_onset_task_tick']<=600 for c in one+two)
    assert [c['pulse_onset_task_tick'] for c in one]!=[c['pulse_onset_task_tick'] for c in two]


def test_packet_adapter_preserves_no_action_and_fault_cases_original_goal_and_case():
    cases=training_cases(reference(),0,1)
    d=dict(valid=np.zeros((2,36),bool),raw_action=np.full((2,36,2),np.nan))
    p=dict(active_tick=np.ones((8,36),bool),policy_fault=np.zeros((8,36),bool))
    p['policy_fault'][0,5]=True
    goals=dict(left=[2.,3.],right=[4.,5.])
    packets=episodes_from_logs(d,p,cases,goals,'frozen')
    assert len(packets)==36 and not packets[0]['decisions']['valid'].any()
    assert packets[5]['physical']['policy_fault'][0]
    assert packets[-1]['original_goal_xy']==[4.,5.]
    cases[0]['alpha']=.9
    assert packets[0]['case']['alpha']==0.
    assert packets[0]['collection_id']=='frozen'
    with pytest.raises(ValueError,match='whole declared'):
        episodes_from_logs({k:v[:,:35] for k,v in d.items()},p,cases,goals,'frozen')
