"""Reward contracts independent of simulator/GPU availability."""
from dataclasses import replace
import numpy as np
import pytest
import jax
import jax.numpy as jp
from sttw_control.tracking_reward import (TrackingConfig, initial_return, transition,
                                         preference_weights, tolerances, within_final,
                                         huber_tail, return_observation)


def tick(state=None, config=TrackingConfig(), xp=np, **overrides):
    values=dict(roll=0., roll_rate=0., speed_error=0., lateral_error=0.,
                heading_error=0., action=xp.zeros(2), alpha=.5, dt=.005,
                alive_rate=1., failure_penalty=100., failed=False,
                enabled=True, config=config, xp=xp)
    values.update(overrides)
    return transition(initial_return(xp=xp) if state is None else state, **values)


@pytest.mark.parametrize('alpha', [0., .25, .5, .75, 1.])
def test_constant_sum_and_monotonic_preferences(alpha):
    wv,wp=preference_weights(alpha,TrackingConfig(),xp=np)
    assert wv>0 and wp>0
    assert np.isclose(wv+wp,1.)
    np.testing.assert_allclose([wv,wp],preference_weights(1-alpha,TrackingConfig(),xp=np)[::-1])
    if alpha==0:assert np.isclose(wp/wv,10.)
    if alpha==1:assert np.isclose(wv/wp,10.)


def test_ideal_step_reward_has_same_scale_for_all_alphas():
    for alpha in [0.,.25,.5,.75,1.]:
        _,parts=tick(alpha=alpha)
        assert np.isclose(sum(parts.values()),.005*(1+4))
        assert parts['recovery']==0


def test_priority_changes_speed_path_tradeoff_not_attitude_or_authority():
    _,low=tick(alpha=0.,roll=.4,roll_rate=.2,speed_error=.2,lateral_error=.2)
    _,high=tick(alpha=1.,roll=.4,roll_rate=.2,speed_error=.2,lateral_error=.2)
    assert low['speed_tail']>high['speed_tail']
    assert low['path_tail']<high['path_tail']
    assert low['attitude']==high['attitude'] and low['roll_rate']==high['roll_rate']
    np.testing.assert_allclose(tolerances(0.,TrackingConfig(),xp=np),[.5,.1])
    np.testing.assert_allclose(tolerances(1.,TrackingConfig(),xp=np),[.2,.4])


def test_geometric_offset_penalized_even_without_heading_error():
    _,on_path=tick(lateral_error=0.,heading_error=0.)
    _,parallel=tick(lateral_error=.3,heading_error=0.)
    assert parallel['path_tracking']<on_path['path_tracking']
    assert parallel['path_tail']<0 and parallel['path_budget']<0


def test_wide_gaussian_distinguishes_large_errors_and_tail_is_unbounded():
    _,a=tick(speed_error=.5)
    _,b=tick(speed_error=1.)
    assert a['speed_tracking']>b['speed_tracking']>0
    assert sum(tick(speed_error=4.)[1].values())<sum(b.values())
    np.testing.assert_allclose(huber_tail(np.array([0.,.5,1.,2.,-2.]),xp=np),[0,.25,1,3,3])


def test_signed_reward_not_positive_clipped_and_failure_replaces_all_terms():
    _,parts=tick(speed_error=10.,lateral_error=4.)
    assert sum(parts.values())<0
    _,parts=tick(failed=True,speed_error=np.nan,action=np.array([np.inf,np.nan]))
    assert sum(parts.values())==-100
    assert all(v==0 for k,v in parts.items() if k!='failure')


def test_rate_terms_scale_with_dt_but_terminal_does_not():
    _,parts=tick(roll=.4,speed_error=.4,lateral_error=.15,enabled=False)
    _,double=tick(roll=.4,speed_error=.4,lateral_error=.15,enabled=False,dt=.01)
    for key in parts:np.testing.assert_allclose(double[key],2*parts[key])
    assert sum(tick(dt=.01,failed=True)[1].values())==-100


def test_no_recovery_bonus_without_prior_departure():
    s=initial_return(xp=np)
    for _ in range(150):
        s,parts=tick(s)
        assert parts['recovery']==0 and not s.ever_left
    assert s.hold==TrackingConfig().hold_seconds


def test_continuous_hold_and_once_only_credit():
    s,_=tick(lateral_error=.2)
    assert s.pending and s.ever_left
    for _ in range(50):s,parts=tick(s)
    assert parts['recovery']==0
    s,_=tick(s,heading_error=.3)
    assert s.hold==0
    for _ in range(99):s,parts=tick(s)
    assert parts['recovery']==0
    s,parts=tick(s)
    assert parts['recovery']==2 and s.credited and not s.pending
    s,_=tick(s,lateral_error=.3)
    total=0
    for _ in range(105):s,parts=tick(s);total+=parts['recovery']
    assert total==0


def test_return_clock_counts_departure_and_late_recovery_keeps_missed_deadline():
    s=initial_return(xp=np)
    for _ in range(100):s,_=tick(s,lateral_error=.3)
    assert s.pending and np.isclose(s.elapsed,.5) and not s.deadline_missed
    for _ in range(501):s,_=tick(s,lateral_error=.3)
    assert s.deadline_missed
    for _ in range(100):s,_=tick(s)
    assert s.deadline_missed and not s.pending


def test_initial_settling_does_not_arm_return():
    s,parts=tick(lateral_error=.3,enabled=False)
    assert not s.ever_left and not s.pending
    assert parts['return_time']==0


def test_action_delta_uses_residual_history_not_baseline_command():
    s=initial_return(xp=np)._replace(previous_action=np.array([.2,-.3]))
    _,same=tick(s,action=np.array([.2,-.3]))
    _,other=tick(s,action=np.array([.3,-.1]))
    assert same['action_delta']==0
    assert np.isclose(other['action_delta'],-.005*.02*(.1**2+.2**2))
    assert return_observation(s,TrackingConfig(),xp=np).shape==(8,)


def test_final_requirements_do_not_change_with_alpha():
    for alpha in [0.,.5,1.]:
        s,_=tick(alpha=alpha,lateral_error=.2)
        assert s.pending
    assert not within_final(0.,0.,.01,.2,0.,TrackingConfig(),xp=np)


def test_numpy_jit_and_batched_jax_agree():
    values=dict(roll=.2,roll_rate=.1,speed_error=.32,lateral_error=-.23,
                heading_error=.1,action=np.array([.1,-.2]),alpha=.8)
    ns,np_parts=tick(**values)
    def f(alpha):return tick(xp=jp,**dict(values,alpha=alpha,action=jp.array(values['action'])))
    js,jp_parts=jax.jit(f)(jp.asarray(.8))
    for a,b in zip(jax.tree.leaves(ns),jax.tree.leaves(js)):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-7)
    for key in np_parts:np.testing.assert_allclose(np_parts[key],jp_parts[key],rtol=1e-5,atol=1e-7)
    _,batched=jax.jit(jax.vmap(f))(jp.array([0.,.5,1.]))
    assert batched['speed_tracking'].shape==(3,)
    assert np.isfinite(jax.grad(lambda a:sum(f(a)[1].values()))(.5))


@pytest.mark.parametrize('change',[{'priority_ratio':0.}, {'speed_wide':0.},
    {'return_seconds':.1}, {'hold_seconds':float('nan')}, {'action_delta_weight':-1.},
    {'speed_relaxed':.1}, {'lateral_wide':.01}])
def test_invalid_reward_parameters_rejected(change):
    with pytest.raises(ValueError):replace(TrackingConfig(),**change)


def test_float32_deadline_matches_numpy_at_exact_control_tick():
    def run(state,_):
        state,_=tick(state,xp=jp,lateral_error=.3)
        return state,(state.elapsed,state.deadline_missed)
    _,(elapsed,missed)=jax.jit(lambda:jax.lax.scan(run,initial_return(),None,length=602))()
    assert not bool(missed[599]) and bool(missed[600])
    assert float(elapsed[599])==pytest.approx(3.,abs=1e-6)
    state=initial_return(xp=np)
    for i in range(602):
        state,_=tick(state,lateral_error=.3)
        assert bool(state.deadline_missed)==bool(missed[i])
