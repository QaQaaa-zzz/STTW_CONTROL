from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
from flax import struct

@struct.dataclass
class Physical:
    physical_tick: object
    data: object

@struct.dataclass
class State:
    tick: object
    alpha: object
    physical: object

@struct.dataclass
class Route:
    family: object
    speed: object
    path: object


def test_conditional_recovery_reset_preserves_full_state_alpha_clock_and_force_draw(monkeypatch):
    from sttw_control import margin_state_sampling as sampler
    from sttw_control.short_recovery import PreferenceRecoveryBatch
    bank=dict(valid=np.ones(4,bool),states=State(j.array([200,400,600,700]),j.array([0.,.25,.25,1.]),Physical(j.array([1200,1400,1600,1700]),j.array([.1,.2,.3,.4]))),routes=Route(j.array([0,1,2,1]),j.array([2.4,2.5,2.6,2.7]),j.arange(4)))
    monkeypatch.setattr(sampler,'bank_risks',lambda *a,**k:j.array([[1.,2.],[1.,3.],[3.,1.],[1.,2.]]))
    pose=lambda x:j.array([0.,0.,x])
    env=SimpleNamespace(cc=None,physics=SimpleNamespace(helpers=SimpleNamespace(pose=pose)))
    base=SimpleNamespace(horizon=10.,config={'path_reward':{}},env=env)
    kwargs=dict(seed=317,forces=[-8.,8.],effort_budget=.03,gamma=.8)
    uniform=PreferenceRecoveryBatch(base,bank,arm='uniform',**kwargs)
    full=PreferenceRecoveryBatch(base,bank,arm='uniform',duration_s=10.,**kwargs)
    from sttw_control.short_recovery import segment_flags
    # The new cap reaches the original deadline; it does not invent a later task.
    assert uniform.maximum_ticks==800 and full.maximum_ticks==2000
    assert bool(segment_flags({'done':j.array(False)},900,uniform.maximum_ticks)['segment_timeout'])
    assert not bool(segment_flags({'done':j.array(False)},900,full.maximum_ticks)['segment_timeout'])
    original_done=segment_flags({'done':j.array(True)},1500,full.maximum_ticks)
    assert bool(original_done['done']) and not bool(original_done['segment_timeout'])
    for got,original in zip(jax.tree.leaves(full.reset(j.int32(0),j.int32(0))),jax.tree.leaves(uniform.reset(j.int32(0),j.int32(0)))):
        np.testing.assert_array_equal(got,original)
    guided=PreferenceRecoveryBatch(base,bank,arm='guided',**kwargs)
    shifted=PreferenceRecoveryBatch(base,bank,arm='guided',target=2.5,bandwidth=.5,**kwargs)
    unchanged_uniform=PreferenceRecoveryBatch(base,bank,arm='uniform',target=2.5,bandwidth=.5,**kwargs)
    assert not np.allclose(shifted.probabilities,guided.probabilities)
    np.testing.assert_array_equal(unchanged_uniform.probabilities,uniform.probabilities)
    ids=j.arange(128,dtype=j.int32);zero=j.zeros(128,j.int32)
    a=jax.jit(jax.vmap(uniform.reset))(ids,zero);b=jax.jit(jax.vmap(shifted.reset))(ids,zero)
    np.testing.assert_array_equal(a.state.alpha,b.state.alpha)
    np.testing.assert_array_equal(a.force,b.force)
    assert np.any(np.asarray(a.bank_index)!=np.asarray(b.bank_index))
    for episode in (a,b):
        for got,original in zip(jax.tree.leaves(episode.state),jax.tree.leaves(bank['states'])):
            np.testing.assert_array_equal(got,np.asarray(original)[episode.bank_index])
        np.testing.assert_array_equal(episode.origin_tick,episode.state.tick)
        np.testing.assert_allclose(episode.pulse[:,0],episode.state.physical.physical_tick*.005)
        np.testing.assert_allclose(episode.pulse[:,1]-episode.pulse[:,0],.1,atol=5e-7)


def test_conditional_and_historical_batches_share_existing_advance_mechanics():
    from sttw_control.short_recovery import PreferenceRecoveryBatch,ShortRecoveryBatch
    assert PreferenceRecoveryBatch.advance_batch is ShortRecoveryBatch.advance_batch
    assert PreferenceRecoveryBatch.advance_one is ShortRecoveryBatch.advance_one


def test_declared_snippet_validates_original_deadline_and_control_alignment():
    import pytest
    from sttw_control.short_recovery import declared_snippet_seconds
    config={'training_future_phase_C':{'episode_s':10.},'timing':{'upper_dt_s':.02}}
    for value in (4.,10.):
        assert declared_snippet_seconds({'snippet_s':value,'original_task_horizon_s':10.},config)==value
    for value in (0.,10.1,4.001,float('nan')):
        with pytest.raises(ValueError):declared_snippet_seconds({'snippet_s':value,'original_task_horizon_s':10.},config)
    with pytest.raises(ValueError):declared_snippet_seconds({'snippet_s':4.,'original_task_horizon_s':20.},config)
