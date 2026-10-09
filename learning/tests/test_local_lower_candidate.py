import json,math
from pathlib import Path
import numpy as np
import jax,jax.numpy as jp
from sttw_control.lower_command import reward_terms,command_rows,publish_local_command,local_roll_reference
from sttw_control.lower_command_validation import summarize_local
CFG=Path(__file__).parents[1]/'configs/lower_local_candidate.json'

def test_reward_bound_and_terminal_replacement():
    s=json.loads(CFG.read_text());r=reward_terms(.2,.08,.28,jp.array([.5,-.5]),jp.zeros(2),False,s,roll_rate=1.,remaining_ticks=10,peak_roll=.31)
    assert float(r['working_roll'])<0
    f=reward_terms(1e6,1e6,.8,jp.ones(2),-jp.ones(2),True,s,roll_rate=100.,remaining_ticks=10,peak_roll=.8)
    bound=sum(s['reward']['caps'].values());expected=-5.-.1*.005*bound*sum(s['gamma']**k for k in range(10))
    np.testing.assert_allclose(f['reward'],expected,rtol=2e-6)
    assert all(float(f[k])==0 for k in s['reward']['caps'])
    assert math.isclose(bound,s['reward']['failure_cost_rate_bound'])

def test_published_sequences_screen_every_slew_tick():
    s=json.loads(CFG.read_text())
    def one(eid):
        rows,slew=command_rows(eid,jp.int32(0),jp.asarray(2.3),s)
        def step(raw,t):
            nxt,rates,target=publish_local_command(raw,rows,t,slew,s)
            return nxt,jp.array([nxt[0],nxt[1],local_roll_reference(nxt[0],nxt[1])])
        return jax.lax.scan(step,jp.array([2.3,0.]),jp.arange(2400))[1]
    data=np.asarray(jax.jit(jax.vmap(one))(jp.arange(12)))
    assert np.max(np.abs(data[:,:,2]))<=.26001
    assert data[:,:,0].min()>=1.69999 and data[:,:,0].max()<=2.80001
    assert np.max(np.abs(data[:,:,1]))<=.28001

def test_physical_best_requires_complete_protocol_and_does_not_use_reward():
    s=json.loads(CFG.read_text());n=2400;t=np.arange(n)*.005
    good=dict(time=t,limited_command=np.tile([2.3,0.],(n,1)),actual_forward_speed=np.full(n,2.3),actual_delta=np.zeros(n),phi=np.zeros(n),peak_roll=np.zeros(n),physical_failure=np.zeros(n,bool),raw_rates=np.zeros((n,2)),final_command_clipped=np.zeros(n,bool),actual_delta_rate=np.zeros(n))
    traces={k:good.copy() for k in s['validation']['case_ids']}
    result=summarize_local(traces,s);assert result['qualified']
    traces[next(iter(traces))]={**good,'actual_delta':np.full(n,.08)}
    assert not summarize_local(traces,s)['qualified']

def test_local_hard_kl_halves_lr_then_stops_after_three():
    from types import SimpleNamespace
    from sttw_control.lower_command_training import handle_candidate_ppo_result
    a=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{'lr':1e-4}]),hard_kl_stop=True)
    cfg=json.loads(CFG.read_text());count=0
    for expected in (1,2,3):
        count,stop=handle_candidate_ppo_result(a,{'hard_kl_stop':True,'nonfinite_stop':False},count,cfg)
        assert count==expected and not a.hard_kl_stop
        assert (stop is not None)==(expected==3)
    assert a.optimizer.param_groups[0]['lr']==1.25e-5

def test_validation_rejects_fault_and_duplicate_clock():
    import pytest
    s=json.loads(CFG.read_text());n=2400
    good=dict(time=np.arange(n)*.005,limited_command=np.tile([2.3,0.],(n,1)),actual_forward_speed=np.full(n,2.3),actual_delta=np.zeros(n),phi=np.zeros(n),peak_roll=np.zeros(n),physical_failure=np.zeros(n,bool),raw_rates=np.zeros((n,2)),final_command_clipped=np.zeros(n,bool),actual_delta_rate=np.zeros(n))
    for change in [{'lower_fault':np.ones(n,bool)},{'nonfinite':np.ones(n,bool)},{'time':np.full(n,11.995)},{'actual_delta_rate':np.full(n,np.nan)}]:
        traces={k:{**good,**change} for k in s['validation']['case_ids']}
        with pytest.raises(ValueError):summarize_local(traces,s)
