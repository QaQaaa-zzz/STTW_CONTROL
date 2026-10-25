"""The new live-following contract must never revive stopped physical state."""
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import struct

from sttw_control.live_path_continuation import make_live_path_step


@struct.dataclass
class Physical:
    failed: object
    controller_memory: object


@struct.dataclass
class State:
    physical: object
    lower: object
    tick: object
    fault: object
    domain_exit: object
    offsets: object
    filtered: object
    history: object


def initial(**overrides):
    values=dict(physical=Physical(j.bool_(False),j.float32(17)),lower=j.float32(23),
                tick=j.int32(0),fault=j.bool_(False),domain_exit=j.bool_(True),
                offsets=j.array([.2,.1]),filtered=j.array([.15,.05]),history=j.int32(7))
    values.update(overrides)
    return State(**values)


class Env:
    cc=SimpleNamespace(dt=.005)
    config=dict(timing=dict(lower_dt_s=.005,upper_dt_s=.02),
        path_reward=dict(scale=1.,offset_rate_weight=0.,offset_rate_scales=[1.,1.],
                         component_caps=dict(offset_rate=10.),failure_extra=100.,failure_cost_rate_bound=5.),
        training_future_phase_C=dict(gamma=.997))

    def __init__(self,contact_at=999,fault_at=999):
        self.contact_at,self.fault_at=contact_at,fault_at
        self.zero_log=dict(active_tick=j.bool_(False),tick_reward=j.float32(0.),
            domain_exit=j.bool_(False),target_seen=j.zeros(2),offset_before=j.zeros(2),
            raw_components=dict(offset_rate=j.float32(0.)),
            effective_components=dict(offset_rate=j.float32(0.)),
            component_capped=dict(offset_rate=j.bool_(False)))

    def tick(self,s,z,path,speed,force_pulse=None,horizon=None):
        # A state-transition fixture, not a simulator or a reward quality test.
        n=s.tick+1
        end=s.replace(tick=n,physical=s.physical.replace(failed=n>=self.contact_at),
            fault=s.fault|(n>=self.fault_at),offsets=s.offsets*.9,filtered=s.filtered*.95,
            lower=s.lower+1,domain_exit=j.bool_(True))
        log=dict(self.zero_log,active_tick=j.bool_(True),tick_reward=j.float32(-1.),
                 domain_exit=j.bool_(True),target_seen=z,offset_before=s.offsets)
        return end,log

    def record_frame(self,s):
        return s.replace(history=s.history+1)


def run(env,s,horizon=.04):
    step=make_live_path_step(env,horizon_s=horizon)
    return jax.jit(step)(s,None,j.float32(1.5),None)


def test_domain_exit_continues_original_tick_and_refreshes_real_history():
    s=initial();end,log=run(Env(),s)
    assert int(end.tick)==4 and bool(end.domain_exit)
    assert int(end.history)==8 and float(end.lower)==27
    assert float(end.physical.controller_memory)==17
    np.testing.assert_array_equal(log['target_seen'],np.zeros((4,2)))
    np.testing.assert_allclose(log['offset_before'][0],s.offsets)
    np.testing.assert_allclose(end.offsets,np.asarray(s.offsets)*.9**4,rtol=1e-6)
    np.testing.assert_allclose(end.filtered,np.asarray(s.filtered)*.95**4,rtol=1e-6)
    np.testing.assert_array_equal(log['active_tick'],[True]*4)
    np.testing.assert_allclose(log['scored_tick_reward'],[-1.]*4)


@pytest.mark.parametrize('reason',['contact','fault','horizon'])
def test_terminal_input_is_unchanged_and_does_not_fabricate_history(reason):
    s=initial()
    if reason=='contact':s=s.replace(physical=s.physical.replace(failed=j.bool_(True)))
    elif reason=='fault':s=s.replace(fault=j.bool_(True))
    else:s=s.replace(tick=j.int32(8))
    end,logs=run(Env(),s)
    for a,b in zip(jax.tree.leaves(end),jax.tree.leaves(s)):
        np.testing.assert_array_equal(a,b)
    assert not np.any(logs['active_tick'])
    np.testing.assert_array_equal(logs['scored_tick_reward'],np.zeros(4))


@pytest.mark.parametrize('reason',['contact','fault','horizon'])
def test_mid_interval_stop_masks_padding_and_preserves_failure(reason):
    env=Env(contact_at=2 if reason=='contact' else 999,fault_at=2 if reason=='fault' else 999)
    s=initial(tick=j.int32(6) if reason=='horizon' else j.int32(0))
    end,logs=run(env,s)
    assert int(end.tick)==(8 if reason=='horizon' else 2)
    assert int(end.history)==8
    np.testing.assert_array_equal(logs['active_tick'],[True,True,False,False])
    np.testing.assert_array_equal(logs['scored_tick_reward'][2:],np.zeros(2))
    if reason=='contact':
        assert bool(end.physical.failed) and float(logs['scored_tick_reward'][1])<-100
    elif reason=='fault':
        assert bool(end.fault)
    else:
        assert not bool(end.physical.failed)
        np.testing.assert_allclose(logs['scored_tick_reward'][:2],[-1.,-1.])


@pytest.mark.parametrize('horizon',[0.,-.1,float('nan'),.013])
def test_invalid_or_unaligned_clock_is_rejected(horizon):
    with pytest.raises(ValueError):make_live_path_step(Env(),horizon_s=horizon)
