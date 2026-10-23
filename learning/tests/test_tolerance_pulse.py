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
