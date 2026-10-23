import jax
import jax.numpy as j
import numpy as np
from sttw_control.tolerance_pulse import schedule, activate, force_probabilities, FORCES


def test_directional_weights_and_uniform_support():
    p, fallback = force_probabilities(2., 8., True, guided=True)
    assert not bool(fallback)
    assert p[4] > p[7] and p[0] > p[3]
    assert np.all(np.asarray(p) >= .5/8)
    np.testing.assert_allclose(np.sum(p), 1.)
    q, _ = force_probabilities(8., 2., True, guided=True)
    np.testing.assert_allclose(p, q[::-1])


def test_invalid_fallback_and_uniform_control():
    for plus, minus in [(float('nan'), 2.), (2., float('inf')), (-1., 2.)]:
        p, fallback = force_probabilities(plus, minus, True, guided=True)
        np.testing.assert_allclose(p, np.ones(8)/8)
        assert bool(fallback)
    p, fallback = force_probabilities(2., 8., False, guided=False)
    np.testing.assert_allclose(p, np.ones(8)/8)
    assert not bool(fallback)


def test_schedule_reproducible_and_control_aligned():
    batch=jax.jit(jax.vmap(lambda eid:schedule(910401,eid,j.int32(0))))(j.arange(2048))
    assert np.all(np.asarray(batch.onset_tick)%4==0)
    assert int(batch.onset_tick.min())>=160 and int(batch.onset_tick.max())<=700
    assert .45 < float(batch.enabled.mean()) < .55
    np.testing.assert_array_equal(schedule(3,4,5).key,schedule(3,4,5).key)
    assert not np.array_equal(schedule(3,4,5).key,schedule(3,4,6).key)


def test_no_early_force_and_direction_frozen_after_firing():
    state=schedule(3,4,5).replace(enabled=j.bool_(True),onset_tick=j.int32(160))
    def step(s,t,h):return activate(s,tick=t,absolute_time=10.+t*.005,heading=h,plus=3.,minus=5.,valid=True,guided=True)
    before=step(state,156,0.)
    assert not bool(before.fired) and np.all(np.asarray(before.pulse)==0)
    fired=jax.jit(step)(before,160,j.pi/2)
    assert bool(fired.fired) and float(fired.force) in np.asarray(FORCES)
    np.testing.assert_allclose(fired.pulse[:2],[10.8,10.9],atol=1e-6)
    np.testing.assert_allclose(fired.pulse[2:],[-float(fired.force),0.,0.],atol=1e-6)
    np.testing.assert_array_equal(step(fired,164,0.).pulse,fired.pulse)
    disabled=state.replace(enabled=j.bool_(False))
    assert not bool(step(disabled,160,0.).fired)


import numpy as np
import jax.numpy as j
from sttw_control.teleop_env import TeleopEnv

def test_zero_pulse_preserves_cpu_transition_and_state():
    env=TeleopEnv(backend='cpu');s=env.initial();q=np.array(s.data.qpos)
    a,la=env._step(s,s.raw,True)
    b,lb=env._step(s,s.raw,True,force_pulse=j.array([0.,.01,0.,0.,0.]))
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
    np.testing.assert_array_equal(s.data.qpos,q)
    np.testing.assert_array_equal(lb['applied_force_impulse'],[0.,0.,0.])

def test_partial_substep_pulse_impulse_and_physical_effect():
    env=TeleopEnv(backend='cpu');s=env.initial()
    a,_=env._step(s,s.raw,True)
    b,log=env._step(s,s.raw,True,force_pulse=j.array([.0001,.0049,0.,2.,0.]))
    np.testing.assert_allclose(log['applied_force_impulse'],[0.,.0096,0.],rtol=2e-5,atol=1e-9)
    assert np.linalg.norm(b.data.qvel-a.data.qvel)>1e-6


def test_wrapper_forwards_pulse_and_resets_episode_state(monkeypatch):
    from types import SimpleNamespace
    from flax import struct
    from sttw_control.tolerance_pulse import PulseBatch
    import sttw_control.tolerance_shaping as shaping
    import sttw_control.path_command_statistics as statistics
    monkeypatch.setattr(shaping, 'measurement_reserve', lambda *a, **k:dict(plus_N=j.asarray(4.),minus_N=j.asarray(6.),valid=j.bool_(True)))
    monkeypatch.setattr(statistics, 'reset_statistics', lambda *a: {})
    monkeypatch.setattr(statistics, 'merge_statistics', lambda a,b: a)
    @struct.dataclass
    class Physical:
        data: object
        controller: object
        physical_tick: object
    @struct.dataclass
    class State:
        tick: object
        physical: object
    @struct.dataclass
    class Core:
        state: object
        env_id: object
        episode: object
        route: object
    class Base:
        horizon=20.
        def __init__(self):
            self.fault=False
            self.env=SimpleNamespace(cc=SimpleNamespace(dt=.005),physics=SimpleNamespace(observe=lambda d:(j.zeros(7),j.zeros(3),j.asarray(2.))))
        def reset(self,eid,episode):return Core(State(j.int32(0),Physical(j.asarray(0.),j.asarray(0.),j.int32(2000))),eid,episode,j.asarray(0.))
        def observation(self,ep):return (j.array([ep.state.tick]),j.zeros(1),j.bool_(False))
        def advance_one(self,ep,z,*,auto_reset,force_pulse):
            assert not auto_reset
            flags=dict(done=j.bool_(True),engineering_fault=j.bool_(self.fault))
            # Fake one control interval, deliberately a partial .02s pulse before termination.
            stats=dict(diagnostics={},applied_force_impulse=.02*force_pulse[2:])
            return ep,self.observation(ep),j.asarray(0.),flags,self.observation(ep),stats
    base=Base();batch=PulseBatch(base,guided=True,pulse_seed=4,reserve_config={})
    ep=batch.reset(j.int32(0),j.int32(2));ep=ep.replace(core=ep.core.replace(state=ep.state.replace(tick=j.int32(160))),pulse_state=ep.pulse_state.replace(enabled=j.bool_(True),onset_tick=j.int32(160)))
    nxt,_,_,_,_,stats=batch.advance_one(ep,j.zeros(2))
    assert int(nxt.episode)==3 and int(nxt.state.tick)==0
    assert not bool(nxt.pulse_state.fired)
    np.testing.assert_array_equal(nxt.impulse,j.zeros(3))
    assert bool(stats['pulse']['fired_now'])
    np.testing.assert_allclose(stats['pulse']['completed_impulse'],.2*stats['pulse']['completed_expected_impulse'],atol=1e-7)
    assert np.linalg.norm(stats['pulse']['completed_impulse'])>0
    base.fault=True
    failed,_,_,_,_,_=batch.advance_one(ep,j.zeros(2))
    assert int(failed.episode)==2 and bool(failed.pulse_state.fired)
