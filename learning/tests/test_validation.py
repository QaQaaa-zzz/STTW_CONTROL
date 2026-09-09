"""Exercise the actual batched validator with a deterministic trajectory fixture."""
from types import SimpleNamespace
import jax.numpy as jp
import numpy as np
from flax import struct
from sttw_control.training import TrainingConfig
from sttw_control.validation import make_validator


@struct.dataclass
class State:
    obs: object
    event: object
    tick: object
    done: object
    terminated: object
    pose: object
    data: object


class FixtureEnv:
    """A path excursion during forcing, recovery, then a second excursion."""
    def __init__(self,config,backend=None):
        self.config=config;self.horizon=8
        self.bundle=SimpleNamespace(chassis=0)

    def reset(self,key):
        return State(jp.zeros(1),jp.zeros(5),jp.array(0),jp.array(False),
                     jp.array(False),jp.zeros(2),Data(jp.zeros(3),jp.eye(3)[None]))

    def step(self,state,action):
        tick=state.tick+1
        disturbed=state.event[2]!=0
        radial=jp.where(disturbed,jp.array([0.,0.,.4,0.,0.,.2,0.,0.,0.])[tick],0.)
        failed=disturbed & (action[0]>.5) & (tick>=7)
        # A dataclass with attribute access, also registered as a JAX tree.
        data=Data(jp.zeros(3),jp.eye(3)[None])
        return state.replace(tick=tick,done=tick>=8,terminated=failed,
                             pose=jp.array([radial,0.]),data=data)

    def path_features(self,pose):return pose


@struct.dataclass
class Data:
    qvel: object
    xmat: object


class Actor:
    def apply(self,p,obs):return jp.full((obs.shape[0],1),p)


def test_validator_final_hold_restarts_and_peak_includes_forcing(monkeypatch):
    from dataclasses import dataclass
    @dataclass
    class Config:
        horizon_seconds: float=8.
        speed_reference: float=0.
        controller: object=None
    monkeypatch.setattr('sttw_control.validation.RecoveryEnv',FixtureEnv)
    env=FixtureEnv(Config(controller=SimpleNamespace(dt=1.)))
    cfg=TrainingConfig(validation_seeds=(1,),validation_hold_seconds=2.,
                       validation_post_seconds=5.,validation_events=(
                           {'start':1.,'duration':2.,'steer_rate':1.},))
    validate=make_validator(env,Actor(),jp.ones(1),cfg)
    result=validate({'actor':0.})
    assert bool(result['post_event_hold_complete'][0])
    # Last invalid state is tick 5; tick 7 completes the two-tick hold.
    np.testing.assert_allclose(result['settling_seconds'],[4.])
    np.testing.assert_allclose(result['post_event_peak'],[.4])
    failed=validate({'actor':1.})
    assert not bool(failed['post_event_hold_complete'][0])
    np.testing.assert_allclose(failed['settling_seconds'],[5.])
