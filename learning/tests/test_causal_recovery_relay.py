from types import SimpleNamespace
import numpy as np
import pytest
import jax.numpy as j
from sttw_control.causal_recovery_relay import relay_choice,relay_inventory,make_relay_step,CausalRelayState
from sttw_control.reference_reserve_gate import initial_gate


def test_relay_choices_keep_constant_and_matched_rules_distinct():
    assert tuple(map(int,relay_choice(0,False)))==(0,0)
    assert tuple(map(int,relay_choice(0,True)))==(1,0)
    assert tuple(map(int,relay_choice(1,True)))==(0,7)
    assert tuple(map(int,relay_choice(1,False)))==(0,0)
    assert tuple(map(int,relay_choice(2,False)))==(0,7)


def test_relay_inventory_preserves_all_ten_events_for_each_control():
    cases=relay_inventory([0.,1.,0.])
    assert len(cases)==30 and len({c['case_id'] for c in cases})==30
    assert {c['mode'] for c in cases}=={'causal_B','causal_support','constant_support'}
    for mode in ['causal_B','causal_support','constant_support']:
        group=[c for c in cases if c['mode']==mode]
        assert len(group)==10 and sum(c['force_N']==0 for c in group)==2
    assert all('handoff_tick' not in c and not c['cache_candidate'] for c in cases)


def test_sensor_based_switch_preserves_history_and_releases_to_A(monkeypatch):
    import sttw_control.anticipatory_reference as kernel
    seen=[]
    def factory(env,actor,*,horizon_s,allow_reentry):
        assert allow_reentry
        def advance(carry,params,path,speed,pulse,profile,lower_bound,zero_mode):
            seen.append((carry,params,profile,pulse))
            return carry,dict(active_tick=j.ones(4,bool),actor_action_valid=j.ones(4,bool))
        return advance
    monkeypatch.setattr(kernel,'make_reference_step',factory)
    sensor={'rate':1.}
    def observe(data):return (j.array([0.,sensor['rate'],0.,0.,0.,0.,0.]),None,None)
    env=SimpleNamespace(physics=SimpleNamespace(observe=observe),config={'residual_interface':{'speed_reference_bounds_m_s':[1.,3.5]},'timing':{'upper_dt_s':.02}})
    physical=SimpleNamespace(data=object(),failed=j.bool_(False))
    state=SimpleNamespace(physical=physical,fault=j.bool_(False),tick=j.int32(100))
    state.replace=lambda **kw:SimpleNamespace(**dict(vars(state),**kw))
    carry=CausalRelayState(state=state,gate=object(),clean_blocks=j.int32(16),relay=initial_gate())
    params={'A':{'w':j.array([1.,2.])},'B':{'w':j.array([-3.,4.])}}
    step=make_relay_step(env,object(),horizon_s=10.)
    end,logs=step(carry,params,object(),1.5,j.zeros(5),0)
    np.testing.assert_array_equal(seen[-1][1]['w'],[-3.,4.]);assert bool(logs['relay_trigger'][0])
    assert end.state.physical is physical and end.gate is carry.gate and int(end.clean_blocks)==16
    sensor['rate']=0.
    for _ in range(32):end,logs=step(end,params,object(),1.5,j.zeros(5),0)
    np.testing.assert_array_equal(seen[-1][1]['w'],[1.,2.]);assert not bool(logs['relay_active'][0])


def test_relay_contrasts_count_rescue_separately_from_task():
    from sttw_control.causal_recovery_relay import relay_contrasts
    rows=[dict(case_id=m,mode=m,alpha=.5,event_id='early',category=c,scored_cost=v)
        for m,c,v in [('AA','contact',4000.),('BB','safe_task_failed',80.),('causal_B','safe_task',20.),('causal_support','safe_task_failed',50.),('constant_support','safe_task_failed',60.)]]
    r=relay_contrasts(rows)
    assert r['causal_B_vs_AA']['physical_gains']==1
    assert r['causal_B_vs_causal_support']['physical_gains']==0
    assert r['causal_B_vs_causal_support']['task_gains']==1
    with pytest.raises(ValueError):relay_contrasts(rows[:-1])
