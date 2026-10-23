"""Sampling optimizations preserve reset selection and diagnostic semantics."""
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from sttw_control.path_command_training import Episode
from sttw_control import short_recovery as recovery
from sttw_control import tolerance_training as training


def test_batched_reset_skips_work_without_done_and_preserves_fault_lanes():
    calls=[]
    def reset(eid, counter):
        jax.debug.callback(lambda x:calls.append(int(x)), eid)
        key=jax.random.fold_in(jax.random.PRNGKey(73),counter)
        return Episode(jax.random.uniform(key,())+eid, eid, eid,counter,j.zeros(6),j.zeros(1))
    def observe(ep):return ep.state[None],(ep.state*2)[None],j.bool_(False)
    batch=SimpleNamespace(reset=reset,observation=observe)
    ended=Episode(j.array([4.,5.,6.]),j.arange(3),j.arange(3),j.array([7,8,9]),j.zeros((3,6)),j.zeros((3,1)))
    final=jax.vmap(observe)(ended)
    run=jax.jit(lambda flags:recovery.reset_finished_batch(batch,ended,final,flags))
    flags=dict(done=j.array([False,False,False]),engineering_fault=j.array([False,False,False]))
    a,ob=run(flags);jax.block_until_ready((a,ob))
    assert calls==[], 'nonterminal batch must not execute reset'
    np.testing.assert_array_equal(a.state,ended.state)
    b,ob=run(dict(done=j.array([True,True,False]),engineering_fault=j.array([False,True,False])))
    jax.block_until_ready((b,ob))
    expected=reset(ended.env_id[0],ended.episode[0]+1)
    np.testing.assert_array_equal(b.state[0],expected.state)
    np.testing.assert_array_equal(b.state[1:],ended.state[1:])
    np.testing.assert_array_equal(b.episode,[8,8,9])
    np.testing.assert_array_equal(ob[0][:,0],b.state)
    np.testing.assert_array_equal(ob[1][:,0],2*b.state)


def test_device_rollout_totals_match_host_sums_maxima_and_joint_exposure():
    acc=training.empty_rollout_totals({'count':j.array(0.),'window_peak_roll':j.zeros(2)})
    flags=dict(physical_failure=j.array([True,False,False]),tracking_domain_failure=j.array([False,True,False]))
    stats=dict(active_ticks=j.array([4,3,2]),diagnostics={'count':j.array(9.),'window_peak_roll':j.array([.2,.4])},
        recovery=dict(family=j.array([0,1,1]),force_index=j.array([0,2,2]),active_ticks=j.array([4,3,2])),
        pulse=dict(fired_now=j.array([False,True,True]),force=j.array([0.,-6.,8.]),fallback_now=j.array([False,True,False]),
                   completed=j.array([True,False,True]),enabled=j.array([False,True,True]),
                   completed_impulse=j.array([[0.,0.,0.],[0.,0.,0.],[.8,0.,0.]]),
                   completed_expected_impulse=j.array([[0.,0.,0.],[0.,0.,0.],[.801,0.,0.]])))
    update=jax.jit(training.accumulate_rollout_totals)
    for _ in range(2):acc=update(acc,j.array([-1.,-2.,-3.]),flags,stats)
    a=jax.device_get(acc)
    assert a['ticks']==18 and a['physical']==2 and a['domain']==2
    assert a['reward_sum']==-12 and a['reward_count']==6
    np.testing.assert_allclose(a['diagnostics']['window_peak_roll'],[.2,.4])
    assert a['diagnostics']['count']==18
    assert a['joint_exposure'][0,0]==8 and a['joint_exposure'][1,2]==10
    np.testing.assert_array_equal(a['pulse']['force_histogram'],[0,2,0,0,0,0,0,2])
    assert a['pulse']['fired']==4 and a['pulse']['completed']==4 and a['pulse']['enabled_completed']==2
    np.testing.assert_allclose(a['pulse']['max_completed_impulse_error_Ns'],.001,atol=2e-8)
