"""Adapter checks: protected observations, raw density and real-time reentry."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import jax
import jax.numpy as j
from flax import struct
from sttw_control.survival_live_collection import LiveSurvivalCollectionState,make_live_survival_collection_step
from sttw_control.reference_reentry import actor_permission
from sttw_control.causal_reference_baseline import initial_gate


@struct.dataclass
class Physical:
    failed: object


@struct.dataclass
class Follower:
    invalid: object


@struct.dataclass
class State:
    tick: object
    fault: object
    domain_exit: object
    physical: object
    follower: object


class Actor:
    def __init__(self):self.inputs=[]
    def apply(self,params,obs):
        self.inputs.append(np.asarray(obs))
        assert np.isfinite(obs).all()
        return j.array([.1,.2])


class Critic:
    def apply(self,params,obs):return obs[:1]


def fake_reference(env,actor,*,horizon_s,allow_reentry):
    assert allow_reentry
    def step(carry,action,path,speed,pulse,profile,floor,zero):
        s=carry.state;active=~(s.physical.failed|s.fault|(s.tick>=2000))
        legal=actor_permission(s.domain_exit,s.follower.invalid,carry.clean_blocks,zero,~active)
        proposed=actor.apply(action,j.zeros(351))
        issued=j.where(legal,proposed.at[0].add(j.where(profile==7,.2,0.)),j.zeros(2))
        return carry.replace(state=s.replace(tick=s.tick+j.where(active,4,0))),dict(
            active_tick=j.repeat(active,4),actor_action_valid=j.repeat(active&legal,4),
            issued_latent=j.tile(issued,(4,1)))
    return step


def setup(*,invalid=False,ever=False,clean=0,tick=0):
    def observation(s,path,speed,horizon):
        obs=j.full(351,j.where(s.follower.invalid,j.nan,s.tick+1),dtype=j.float32)
        return obs,j.concatenate([obs,j.array([(horizon-s.tick*.005)/horizon])]),s.follower.invalid
    env=SimpleNamespace(cc=SimpleNamespace(dt=.005),config={'residual_interface':{'speed_reference_bounds_m_s':[1.,3.5]}},observation=observation)
    state=State(j.int32(tick),j.bool_(False),j.bool_(ever),Physical(j.bool_(False)),Follower(j.bool_(invalid)))
    carry=LiveSurvivalCollectionState(state,jax.random.PRNGKey(73),initial_gate(),j.int32(clean))
    params=dict(actor={},task_critic={},physical=dict(kernel=j.zeros((352,1)),bias=j.zeros(1)))
    actor=Actor()
    with patch('sttw_control.anticipatory_reference.make_reference_step',fake_reference):
        one=make_live_survival_collection_step(env,actor,horizon_s=10.,task_critic=Critic())
    return carry,params,actor,one


def test_invalid_observation_never_reaches_network_or_becomes_an_action():
    carry,params,actor,one=setup(invalid=True,ever=True)
    end,(decision,logs)=one(carry,params,j.zeros(2),None,1.5,j.zeros(5),j.int32(7))
    assert not bool(decision['valid'])
    assert np.isnan(decision['observation']).all() and np.isnan(decision['raw_action']).all()
    np.testing.assert_array_equal(actor.inputs[0],np.zeros(351))
    np.testing.assert_array_equal(end.key,carry.key)
    assert int(end.state.tick)==4 and logs['fallback_active'].all()


def test_log_density_is_of_raw_sample_before_deterministic_support():
    carry,params,actor,one=setup()
    end,(d,logs)=one(carry,params,j.zeros(2),None,1.5,j.zeros(5),j.int32(7))
    assert bool(d['valid'])
    raw=np.asarray(d['raw_action']);mean=np.asarray(d['mean'])
    expected=-.5*np.sum((raw-mean)**2+np.log(2*np.pi))
    np.testing.assert_allclose(d['log_prob'],expected,atol=1e-6)
    assert not np.array_equal(raw,logs['issued_latent'][0])
    assert not np.array_equal(end.key,carry.key)


def test_reentry_records_current_observation_and_clock_after_clean_wait():
    carry,params,actor,one=setup(ever=True,clean=0,tick=40)
    _,(blocked,_)=one(carry,params,j.zeros(2),None,1.5,j.zeros(5),j.int32(0))
    assert not bool(blocked['valid'])
    ready=carry.replace(clean_blocks=j.int32(16),state=carry.state.replace(tick=j.int32(104)))
    _,(d,_)=one(ready,params,j.zeros(2),None,1.5,j.zeros(5),j.int32(0))
    assert bool(d['valid']) and int(d['task_tick'])==104
    np.testing.assert_array_equal(d['observation'],np.full(351,105.))
    np.testing.assert_allclose(d['critic_observation'][-1],.948,atol=1e-7)
