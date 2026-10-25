"""Hand-computed physical-time credit through policy-inactive gaps."""
import numpy as np
import pytest
from sttw_control.live_task_credit import live_task_gae
from sttw_control.survival_task_credit import complete_task_gae


def test_rewards_and_discounts_cross_real_three_step_gap():
    r=live_task_gae(np.array([1.,2.,3.,4.]),np.array([.5,np.nan,np.nan,.75]),
        np.array([True,False,False,True]),np.ones(4,dtype=bool),
        np.array([False,False,False,True]),gamma=.9,gae_lambda=.8)
    np.testing.assert_allclose(r['discounted_transition_rewards'][[0,3]],[5.23,4.])
    np.testing.assert_array_equal(r['duration_steps'][[0,3]],[3,1])
    np.testing.assert_allclose(r['continuation_discounts'][[0,3]],[.729,0.])
    # delta0=5.23+.9^3*.75-.5, delta3=4-.75.
    np.testing.assert_allclose(r['advantages'][[0,3]],[6.489806,3.25])
    assert np.isnan(r['advantages'][1:3]).all()


def test_lambda_one_matches_full_discounted_return_across_gap():
    r=live_task_gae(np.array([1.,2.,3.,4.]),np.array([.5,np.nan,np.nan,.75]),
        np.array([True,False,False,True]),np.ones(4,dtype=bool),
        np.array([False,False,False,True]),gamma=.9,gae_lambda=1.)
    np.testing.assert_allclose(r['returns'][[0,3]],[8.146,4.])


def test_rewards_after_last_policy_action_reach_that_action():
    r=live_task_gae(np.array([1.,2.,3.,np.nan]),np.array([.7,np.nan,np.nan,np.nan]),
        np.array([True,False,False,False]),np.array([True,True,True,False]),
        np.array([False,False,True,False]),gamma=.5,gae_lambda=.2)
    np.testing.assert_allclose(r['returns'][0],2.75)
    assert r['duration_steps'][0]==3 and r['continuation_discounts'][0]==0
    assert np.isnan(r['returns'][1:]).all()


def test_contiguous_actions_match_existing_standard_gae():
    rewards=np.array([1.,-2.,4.,np.nan]);values=np.array([.5,.75,-.3,np.nan])
    mask=np.array([True,True,True,False]);done=np.array([False,False,True,False])
    a=live_task_gae(rewards,values,mask,mask,done,gamma=.93,gae_lambda=.87)
    b=complete_task_gae(rewards,values,mask,done,gamma=.93,gae_lambda=.87)
    np.testing.assert_allclose(a['advantages'],b['advantages'],equal_nan=True)
    np.testing.assert_allclose(a['returns'],b['returns'],equal_nan=True)


def test_no_action_prefix_is_reported_not_assigned_to_a_future_action():
    r=live_task_gae(np.array([10.,2.,4.]),np.array([np.nan,.3,np.nan]),
        np.array([False,True,False]),np.ones(3,dtype=bool),
        np.array([False,False,True]),gamma=.5,gae_lambda=.9)
    assert r['unassigned_prefix_steps']==1
    assert r['unassigned_discounted_prefix_reward']==10.
    assert r['returns'][1]==4.


@pytest.mark.parametrize('gamma',[0.,.91,1.])
def test_lambda_one_equals_independent_full_timeline_monte_carlo(gamma):
    rng=np.random.default_rng(1229)
    for density in [.1,.4,1.]:
        rewards=rng.normal(size=53);rewards[49]=-1000.;rewards[50:]=np.nan
        active=np.arange(53)<50;legal=(rng.random(53)<density)&active;legal[0]=True
        values=np.where(legal,rng.normal(size=53),np.nan)
        terminal=np.arange(53)==49
        full=np.full(53,np.nan);tail=0.
        for tick in range(49,-1,-1):
            tail=float(rewards[tick])+gamma*tail;full[tick]=tail
        r=live_task_gae(rewards,values,legal,active,terminal,gamma=gamma,gae_lambda=1.)
        np.testing.assert_allclose(r['returns'][legal],full[legal],rtol=1e-12,atol=1e-10)


@pytest.mark.parametrize('actions,active,terminal',[
    ([True,False,True],[True,False,True],[False,False,True]),
    ([True,False,False],[True,True,True],[False,False,False]),
    ([True,True,True],[True,True,True],[False,True,False]),
    ([False,False,False],[True,True,True],[False,False,True]),
    ([True,False,True],[True,True,False],[False,True,False]),
    ([True,True,True],[True,True,True],[False,True,True]),
])
def test_unknown_cross_reset_or_illegal_samples_rejected(actions,active,terminal):
    with pytest.raises(ValueError):
        live_task_gae(np.ones(3),np.zeros(3),np.array(actions),np.array(active),np.array(terminal),gamma=.9,gae_lambda=.9)


def test_nonfinite_fallback_reward_is_not_silently_dropped():
    with pytest.raises(ValueError):
        live_task_gae(np.array([1.,np.nan,2.]),np.array([.3,np.nan,.4]),
            np.array([True,False,True]),np.ones(3,dtype=bool),np.array([False,False,True]),gamma=.9,gae_lambda=.9)
