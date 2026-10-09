import jax
import jax.numpy as jp
import numpy as np
from sttw_control.smooth_evaluation import expand_cases,masked_step


def test_case_method_expansion_keeps_horizons_and_b0_identity():
    cases=[('straight_hold',np.zeros((16,3)),10),('fast_turn',np.ones((16,3)),16)]
    rows,alphas,bypass,horizons=expand_cases(cases,['alpha0','alpha1','B0'],[0,1,0])
    assert rows.shape==(6,16,3)
    np.testing.assert_array_equal(alphas,[0,1,0,0,1,0])
    np.testing.assert_array_equal(bypass,[False,False,True,False,False,True])
    np.testing.assert_array_equal(horizons,[2000,2000,2000,3200,3200,3200])


def test_padding_and_failed_tail_do_not_advance_or_score_again():
    def step(state,z):return state+1,{'active_tick':jp.ones(4,bool),'reward':jp.ones(4)*-100}
    zero={'active_tick':jp.zeros(4,bool),'reward':jp.zeros(4)}
    fn=jax.jit(jax.vmap(lambda state,active:masked_step(step,state,jp.zeros(2),active,zero)))
    states,logs=fn(jp.array([9,10,11]),jp.array([True,False,False]))
    np.testing.assert_array_equal(states,[10,10,11])
    assert np.asarray(logs['active_tick']).sum()==4
    np.testing.assert_array_equal(logs['reward'][1:],0)


def test_padding_template_uses_full_policy_log_not_tick_log():
    from sttw_control.smooth_evaluation import policy_log_template
    class Env:
        zero_log={'active_tick':jp.bool_(False)}
        def policy_step(self,state,z,bypass):
            return state,0.,False,{'active_tick':jp.ones(4,bool),'scored_tick_reward':jp.ones(4),'motion':jp.ones((4,2))},None
    log=policy_log_template(Env(),jp.zeros(3))
    assert set(log)=={'active_tick','scored_tick_reward','motion'}
    assert log['motion'].shape==(4,2)
    assert not np.asarray(log['active_tick']).any()
