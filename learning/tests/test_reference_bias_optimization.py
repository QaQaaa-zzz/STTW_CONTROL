"""Reference authority, exact baseline retention, full-carry and search failures."""
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as j
import pytest
from sttw_control.path_command_policy import map_latent
from sttw_control.reference_bias_optimization import (
    add_physical_bias, bias_targets, make_bias_step, minimize_nominal_cost)

CFG={'residual_interface':dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.2)}


def test_zero_bias_retains_native_exactly_even_at_saturation_and_faults():
    for native in [[-50.,30.],[.3,-.2],[0.,0.]]:
        native=j.array(native);out,_,_=jax.jit(lambda n:add_physical_bias(n,j.zeros(2),CFG))(native)
        np.testing.assert_array_equal(out,native)
    for bad in [j.inf,j.nan]:
        out,_,_=add_physical_bias(j.array([bad,.3]),j.array([.1,.02]),CFG)
        assert not np.isfinite(out[0])
    out,_,_=add_physical_bias(j.array([.2,.3]),j.array([j.nan,.02]),CFG)
    assert np.isnan(out[0])


def test_bias_projects_to_original_asymmetric_authority_and_no_zero_channel_roundtrip():
    native=j.array([.35,-.25])
    out,_,_=add_physical_bias(native,j.array([-.2,0.]),CFG)
    assert float(out[1])==float(native[1])
    np.testing.assert_allclose(map_latent(out,CFG),np.asarray(map_latent(native,CFG))+[-.2,0],atol=1e-7)
    for bias in [[1.,1.],[-2.,-1.]]:
        out,_,_=add_physical_bias(native,j.array(bias),CFG);physical=np.asarray(map_latent(out,CFG))
        assert -1<=physical[0]<=.25 and abs(physical[1])<=.2
    targets=bias_targets([1,-1,-1,1,.5,.5])
    np.testing.assert_array_equal(targets[[0,4]],np.zeros((2,2)))
    np.testing.assert_allclose(targets[1:4],[[.25,-.05],[-.25,.05],[.125,.025]])
    with pytest.raises(ValueError):bias_targets([np.nan]*6)
    with pytest.raises(ValueError):bias_targets([2]*6)


def test_bias_step_forwards_entire_carry_once_and_preserves_native_parameters(monkeypatch):
    import sttw_control.goal_preview_baseline as module
    seen=[]
    class Actor:
        def apply(self,p,obs):return p['weights']+obs[:2]
    actor=Actor();state=SimpleNamespace(projection=SimpleNamespace(progress=j.float32(2.)))
    carry=SimpleNamespace(state=state,history=object(),eso=object(),fifo=object(),clock=object())
    params=dict(actor=dict(native=dict(weights=j.array([.2,-.3])),keep_channels=j.array([True,True])),
                bias_positions=j.array([0.,1.,2.,3.,4.]),bias_targets=j.array(bias_targets([0,0,.4,.6,0,0])))
    def fake_factory(env,wrapped,**kwargs):
        def advance(c,p,path,speed,pulse,mode):
            assert c is carry
            seen.append((c,p))
            z=wrapped.apply(p['actor']['native'],j.array([.1,.2]))
            return c,dict(original_actor_latent=j.tile(z,(4,1)),actor_action_valid=j.ones(4,bool))
        return advance
    monkeypatch.setattr(module,'make_goal_step',fake_factory)
    env=SimpleNamespace(config=CFG,observation=lambda *a:(j.array([.1,.2]),None,None))
    end,logs=make_bias_step(env,actor,horizon_s=10.,spec={})(carry,params,None,1.1,'hidden event passed only to physical kernel',1)
    assert end is carry and len(seen)==1
    assert seen[0][1]['actor']['native']['native'] is params['actor']['native']
    assert set(params['actor']['native'])=={'weights'}
    np.testing.assert_allclose(logs['original_actor_latent'][0],[.3,-.1],atol=1e-7)
    np.testing.assert_allclose(logs['reference_bias_target'][0],[.1,.03],atol=1e-7)


def test_cem_charges_all64_candidates_retains_feasible_incumbent_and_rejects_unknown():
    calls=[]
    def objective(x):
        calls.append(x.copy())
        return np.sum((x-.2)**2,axis=1)
    got=minimize_nominal_cost(objective,seed=123)
    assert len(calls)==4 and sum(len(x) for x in calls)==got['scored_candidates']==64
    np.testing.assert_array_equal(calls[0][0],np.zeros(6))
    assert got['cost']<=6*.2**2
    for i in range(1,4):np.testing.assert_array_equal(calls[i][0],got['history'][i-1]['incumbent'])
    def infeasible_elsewhere(x):
        s=np.full(len(x),np.inf);s[0]=2.;return s
    kept=minimize_nominal_cost(infeasible_elsewhere,seed=1)
    np.testing.assert_array_equal(kept['vector'],np.zeros(6))
    with pytest.raises(ValueError,match='unknown'):
        minimize_nominal_cost(lambda x:np.full(len(x),np.nan),seed=1)
    with pytest.raises(ValueError,match='infeasible'):
        minimize_nominal_cost(lambda x:np.full(len(x),np.inf),seed=1)


def test_matched_random_shares_first_batch_budget_but_does_not_update_distribution():
    callbacks=[]
    for method in ['cem','random']:
        calls=[]
        def f(x):
            calls.append(x.copy());return np.sum((x-.5)**2,axis=1)
        got=minimize_nominal_cost(f,seed=5,method=method)
        assert got['scored_candidates']==64
        callbacks.append(calls)
    np.testing.assert_array_equal(callbacks[0][0],callbacks[1][0])
    assert not np.array_equal(callbacks[0][1][1:],callbacks[1][1][1:])
    with pytest.raises(ValueError):minimize_nominal_cost(lambda x:np.ones(16),seed=1,method='new_sweep')


def test_realization_receipt_describes_executed_zero_and_nonzero_channels():
    native=j.array([-50.,30.])
    for bias in [j.zeros(2),j.array([0.,-.1]),j.array([.1,0.])]:
        action,realized,clipped=add_physical_bias(native,bias,CFG)
        np.testing.assert_array_equal(realized,map_latent(action,CFG))
        np.testing.assert_array_equal(np.asarray(clipped)[np.asarray(bias)==0],False)
