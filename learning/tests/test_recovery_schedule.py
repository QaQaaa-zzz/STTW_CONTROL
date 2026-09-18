from dataclasses import asdict
import jax
import jax.numpy as jp
import numpy as np
import pytest
from sttw_control.timed_reference import TimedReferenceConfig, schedule, reference_trace


def test_recovery_sequence_has_bounded_random_conflict_and_causal_exit():
    c=TimedReferenceConfig(speed_max=3.,yaw_rate_max=1.8,recovery_probability=1.,recovery_start=4.)
    rows=np.asarray(jax.jit(jax.vmap(lambda k:schedule(k,c,2.1)))(jax.random.split(jax.random.PRNGKey(9),128)))
    assert rows.shape==(128,5,3)
    assert np.all(np.diff(rows[:,:,0],axis=1)>0)
    assert np.all((rows[:,1,0]>=.5)&(rows[:,1,0]<=1))
    assert np.all((rows[:,1:,1]>=1.7)&(rows[:,1:,1]<=3.))
    np.testing.assert_allclose(rows[:,1:4,1:],np.repeat(rows[:,1:2,1:],3,axis=1))
    np.testing.assert_allclose(rows[:,-1,0],4.)
    np.testing.assert_allclose(rows[:,-1,2],0.)
    assert rows[:,1,2].min() < -1.7 and rows[:,1,2].max()>1.7
    for seed in [3,8]:
        tr=reference_trace(c,.005,10.,2.1,seed=seed)
        assert np.max(np.abs(tr['reference_command'][1401:,1]))<1e-6
        assert np.max(np.abs(np.diff(tr['reference_command'][:,1])))<=.00300001


def test_default_schedule_preserves_four_rows_and_fixed_override():
    c=TimedReferenceConfig()
    assert schedule(jax.random.PRNGKey(2),c,2.1).shape==(4,3)
    c=TimedReferenceConfig(recovery_probability=1.,fixed=((0.,2.1,0.),(1.,2.5,1.8)))
    np.testing.assert_allclose(schedule(jax.random.PRNGKey(2),c,2.1),c.fixed)


def test_mixture_preserves_stress_rows_exactly_and_is_reset_random():
    keys=jax.random.split(jax.random.PRNGKey(17),1000)
    c=TimedReferenceConfig(recovery_probability=.8)
    rows=np.asarray(jax.jit(jax.vmap(lambda k:schedule(k,c,2.1)))(keys))
    original=np.asarray(jax.jit(jax.vmap(lambda k:schedule(k,TimedReferenceConfig(),2.1)))(keys))
    recover=rows[:,-1,2]==0
    assert 750<recover.sum()<850
    np.testing.assert_array_equal(rows[~recover,:4],original[~recover])
    np.testing.assert_array_equal(rows[~recover,-1,1:],original[~recover,-1,1:])


@pytest.mark.parametrize('kw',[{'recovery_probability':1.1},{'recovery_probability':-.1},
    {'recovery_start':.5,'recovery_probability':.8},{'conflict_start_window':(1.,.5)}])
def test_bad_recovery_config_rejected(kw):
    with pytest.raises(ValueError):TimedReferenceConfig(**kw)
