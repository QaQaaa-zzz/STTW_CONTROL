"""Continuation observes live lower state without inventing upper observations."""
import jax
import jax.numpy as j
import numpy as np
from flax import struct
from sttw_control.physical_continuation import lower_risk_observation, make_hold_last_tick


@struct.dataclass
class State:
    physical: object
    lower: object
    tick: object
    fault: object
    yaw_rate: object
    history: object
    domain_exit: object
    alpha: object


@struct.dataclass
class Physical:
    data: object
    controller: object
    actuator: object
    governor: object
    physical_tick: object
    yaw_unwrapped: object
    failed: object


def initial():
    return State(Physical(j.array([.2,.3]),j.array([.4]),j.array([.5]),
        j.array([1.5,.18]),j.int32(200),j.float32(.1),j.bool_(False)),
        j.array([.6]),j.int32(100),j.bool_(False),j.float32(0.),
        j.array([99.]),j.bool_(True),j.float32(.5))


def test_risk_view_uses_live_measurement_and_includes_hidden_lower_state():
    state = initial()
    obs = lower_risk_observation(state,measurement=state.physical.data,
        pose=j.array([1.,2.,.1]),true_speed=j.float32(1.4),
        held_reference=j.array([1.5,.18]),remaining_s=j.float32(3.))
    np.testing.assert_array_equal(obs['measurement'],state.physical.data)
    assert 'upper_history' not in obs
    assert obs['upper_action_valid'] == False
    np.testing.assert_array_equal(obs['lower_state'],state.lower)
    assert obs['valid']


def test_bad_measurement_is_unknown_not_usable_risk_sample():
    obs = lower_risk_observation(initial(),measurement=j.array([j.nan]),
        pose=j.zeros(3),true_speed=j.float32(1.),held_reference=j.zeros(2),remaining_s=j.float32(3.))
    assert not obs['valid']


def test_continuation_retains_history_and_updates_only_live_lower_state():
    def tick(physical,lower,pulse,time,reference):
        return physical.replace(data=physical.data+1,physical_tick=physical.physical_tick+1,
            yaw_unwrapped=physical.yaw_unwrapped+.005),lower+1,dict(lower_fault=j.bool_(False),nonfinite=j.bool_(False))
    advance = make_hold_last_tick(tick,dt=.005)
    state = initial()
    end,_ = jax.jit(advance)(state,j.zeros(5),j.float32(0.),j.array([1.5,.18]))
    assert end.tick == state.tick+1
    assert end.physical.physical_tick == state.physical.physical_tick+1
    np.testing.assert_array_equal(end.history,state.history)
    np.testing.assert_array_equal(end.lower,state.lower+1)
    assert end.alpha == state.alpha and end.domain_exit
    np.testing.assert_allclose(end.yaw_rate,1.,rtol=1e-5)


def test_view_masks_faults_and_exhausted_horizon():
    for state,time in [(initial().replace(fault=j.bool_(True)),1.),(initial(),0.)]:
        obs = lower_risk_observation(state,measurement=j.zeros(2),pose=j.zeros(3),
            true_speed=j.float32(1.),held_reference=j.zeros(2),remaining_s=j.float32(time))
        assert not obs['valid']
