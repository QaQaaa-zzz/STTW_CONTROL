from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
from sttw_control.entry_recovery_factorial import actor_b_at_tick,factorial_inventory,make_factorial_step,factorial_contrasts


def test_two_by_two_selects_whole_frozen_actor_at_exact_handoff():
    for mode,expected in enumerate([[False,False,False],[False,True,True],[True,False,False],[True,True,True]]):
        assert [bool(actor_b_at_tick(mode,t,8)) for t in [4,8,12]]==expected


def test_declared_inventory_covers_all_controls_and_exact_eight_cache_candidates():
    direction=[0.,1.,0.];cases=factorial_inventory(direction)
    assert len(cases)==40 and len({x['case_id'] for x in cases})==40
    assert Counter(x['mode'] for x in cases)==dict(AA=10,AB=10,BA=10,BB=10)
    assert sum(x['cache_candidate'] for x in cases)==8
    assert all(x['profile']==0 and x['source_batch_slot']==0 for x in cases)
    for alpha,early in [(.25,388),(.5,380)]:
        group=[x for x in cases if x['alpha']==alpha]
        assert sorted({(x['force_N'],x['pulse_onset_task_tick']) for x in group})==sorted([(0.,early),(-32.,early),(32.,early),(-32.,580),(32.,580)])
    assert direction==[0.,1.,0.]
    with pytest.raises(ValueError):factorial_inventory([0.,2.,0.])


def test_step_preserves_entire_carry_and_does_not_publish_inactive_actor(monkeypatch):
    import jax.numpy as j
    import sttw_control.anticipatory_reference as kernel
    seen={}
    def factory(env,actor,*,horizon_s,allow_reentry):
        assert allow_reentry and horizon_s==10.
        def advance(carry,params,path,speed,pulse,profile,lower_bound,zero_mode):
            seen.update(carry=carry,params=params,pulse=pulse,profile=profile,lower_bound=lower_bound,zero_mode=zero_mode)
            return carry,dict(active_tick=j.ones(4,bool),actor_action_valid=j.array([False]*4))
        return advance
    monkeypatch.setattr(kernel,'make_reference_step',factory)
    carry=SimpleNamespace(state=SimpleNamespace(tick=j.int32(8)),history=object())
    env=SimpleNamespace(config={'residual_interface':{'speed_reference_bounds_m_s':[1.,3.5]}})
    step=make_factorial_step(env,object(),horizon_s=10.)
    end,logs=step(carry,{'A':{'w':j.array([1.,2.])},'B':{'w':j.array([-3.,4.])}},object(),1.5,j.zeros(5),1,8)
    assert end is carry and seen['carry'] is carry
    np.testing.assert_array_equal(seen['params']['w'],[-3.,4.])
    assert int(seen['profile'])==0 and float(seen['lower_bound'])==1. and not bool(seen['zero_mode'])
    assert bool(logs['desired_B'].all()) and not bool(logs['selected_B'].any())


def test_factorial_counts_do_not_call_four_combinations_independent_recoveries():
    rows=[dict(case_id=m,mode=m,alpha=.25,event_id='early',category=c,scored_cost=v)
          for m,c,v in [('AA','contact',100.),('AB','safe_task_failed',20.),('BA','contact',90.),('BB','safe_task',10.)]]
    result=factorial_contrasts(rows)
    assert result['AB_vs_AA']['physical_gains']==1 and result['AB_vs_AA']['task_gains']==0
    assert result['BB_vs_BA']['physical_gains']==1 and result['BB_vs_BA']['task_gains']==1
    assert result['BB_vs_AB']['physical_gains']==0 and result['BB_vs_AB']['task_gains']==1
    with pytest.raises(ValueError):factorial_contrasts(rows[:-1])


def test_cache_metadata_rejects_duplicates_partial_runs_and_condition_drift():
    from sttw_control.entry_recovery_run import validate_prior_records
    declared=[dict(case_id='x',force_N=-32.),dict(case_id='y',force_N=0.)]
    results=[dict(r,physical_outcome='survived') for r in declared]
    index=[dict(case_id=r['case_id'],file=r['case_id']+'.npz') for r in declared]
    status=dict(phase='completed',completed_cases=2)
    assert set(validate_prior_records(status,declared,results,index,expected_count=2))=={'x','y'}
    with pytest.raises(ValueError):validate_prior_records(dict(status,phase='running'),declared,results,index,expected_count=2)
    with pytest.raises(ValueError):validate_prior_records(status,declared,results[:1]*2,index,expected_count=2)
    with pytest.raises(ValueError):validate_prior_records(status,declared,results,index[:1]*2,expected_count=2)
    with pytest.raises(ValueError):validate_prior_records(status,declared,[dict(results[0],force_N=32.),results[1]],index,expected_count=2)
