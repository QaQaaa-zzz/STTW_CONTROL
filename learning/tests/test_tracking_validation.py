"""Exercise new batched final-hold validation with deterministic state transitions."""
from dataclasses import replace
from types import SimpleNamespace
import jax.numpy as jp
from flax import struct
import numpy as np
import pytest
from sttw_control.tracking_reward import TrackingConfig,initial_return,transition
from sttw_control.tracking_validation import make_tracking_validator


@struct.dataclass
class Data:
    qvel: object
    xmat: object


@struct.dataclass
class State:
    obs: object
    event: object
    tick: object
    done: object
    terminated: object
    pose: object
    data: object
    measurement: object
    priority_alpha: object
    tracking_state: object
    reward: object
    reference_pose: object = None
    reference_command: object = None
    yaw_rate: object = None


class Env:
    horizon=10
    bundle=SimpleNamespace(chassis=0)
    def __init__(self):
        self.config=SimpleNamespace(controller=SimpleNamespace(dt=.1),horizon_seconds=1.,
            timed_reference=None,
            tracking=TrackingConfig(start_seconds=0.,hold_seconds=.2,return_seconds=.5),
            priority=SimpleNamespace(validation_alphas=(0.,.5,1.)))
    def reset(self,key):
        return State(jp.zeros(1),jp.zeros(6),jp.int32(0),jp.bool_(False),jp.bool_(False),jp.zeros(3),
            Data(jp.array([2.1,0.,0.]),jp.eye(3)[None]),jp.zeros(7),jp.asarray(.5),initial_return(),jp.asarray(0.))
    def set_priority(self,s,a):return s.replace(priority_alpha=a)
    def speed_command(self,tick):return jp.asarray(2.1)
    def path_features(self,pose):return pose
    def step(self,s,a):
        tick=s.tick+1
        active=(s.event[2]!=0.) | (s.event[3]!=0.) | (s.event[5]!=0.)
        lateral=jp.where(active & (tick>=2)&(tick<=3),.2,0.)
        # Large policy output causes a late second excursion despite earlier recovery.
        lateral=jp.where((a[0]>.5)&(tick>=9),.4,lateral)
        failed=(a[0]>1.5)&(tick>=9)
        rs,parts=transition(s.tracking_state,roll=0.,roll_rate=0.,speed_error=0.,
            lateral_error=lateral,heading_error=0.,action=jp.zeros(2),alpha=s.priority_alpha,
            dt=.1,alive_rate=1.,failure_penalty=100.,failed=failed,
            enabled=True,config=self.config.tracking)
        return s.replace(tick=tick,done=(tick>=10)|failed,terminated=failed,
                         pose=jp.array([lateral,0.,0.]),tracking_state=rs,reward=sum(parts.values()))


class Actor:
    def apply(self,p,obs):return jp.full((obs.shape[0],2),p)


def test_validator_never_substitutes_ever_recovered_for_terminal_hold():
    env=Env();cfg=SimpleNamespace(validation_seeds=(3,),validation_events=({'start':.1,'duration':.2,'force':1.},))
    validate=make_tracking_validator(env,Actor(),jp.ones(1),cfg)
    good=validate({'actor':0.})
    assert np.all(good['terminal_tracking_hold']) and np.all(good['recovered_after_excursion']), {k:np.asarray(v).tolist() for k,v in good.items()}
    assert not np.any(good['maintained_without_excursion'])
    late=validate({'actor':1.})
    assert not np.any(late['terminal_tracking_hold'])
    assert not np.any(late['recovered_after_excursion'])
    failed=validate({'actor':2.})
    assert np.all(failed['failed']) and not np.any(failed['terminal_tracking_hold'])


def test_validator_nominal_no_excursion_is_maintenance_not_recovery():
    cfg=SimpleNamespace(validation_seeds=(3,),validation_events=None)
    result=make_tracking_validator(Env(),Actor(),jp.ones(1),cfg)({'actor':0.})
    assert np.all(result['maintained_without_excursion'])
    assert not np.any(result['recovered_after_excursion'])


def test_validator_refuses_to_extend_episode_to_hide_missing_return_window():
    cfg=SimpleNamespace(validation_seeds=(3,),validation_events=({'start':.7,'duration':.2,'force':1.},))
    with pytest.raises(ValueError,match='return window'):
        make_tracking_validator(Env(),Actor(),jp.ones(1),cfg)


class TimedEnv(Env):
    def __init__(self):
        super().__init__()
        self.config.timed_reference = object()
        self.config.tracking = replace(self.config.tracking, timed=True)

    def reset(self, key):
        return super().reset(key).replace(reference_pose=jp.zeros(3),
            reference_command=jp.array([2.1, 0.]), yaw_rate=jp.asarray(0.))

    def step(self, s, action):
        active = (s.event[2] != 0.) | (s.event[3] != 0.) | (s.event[5] != 0.)
        error = jp.where(active, .3, .05)
        tick = s.tick + 1
        return s.replace(tick=tick, done=tick>=self.horizon,
            pose=jp.array([error, 0., 0.]), yaw_rate=error)


def test_timed_validator_reports_alpha_bounds_and_separate_nominal_statistics():
    config = SimpleNamespace(validation_seeds=(3,),
        validation_events=({'start': .1, 'duration': .2, 'force': 1.},))
    result = make_tracking_validator(TimedEnv(), Actor(), jp.ones(1), config)({'actor': 0.})
    np.testing.assert_allclose(result['longitudinal_rmse'], .3)
    np.testing.assert_allclose(result['yaw_rate_rmse'], .3)
    np.testing.assert_allclose(result['nominal_longitudinal_rmse'], .05)
    np.testing.assert_allclose(result['nominal_yaw_rate_rmse'], .05)
    np.testing.assert_allclose(result['longitudinal_tolerance_exceed_fraction'], [1., 0., 0.])
    np.testing.assert_allclose(result['yaw_rate_tolerance_exceed_fraction'], [0., 1., 1.])
    np.testing.assert_array_equal(result['nominal_longitudinal_tolerance_exceed_fraction'], 0.)
    np.testing.assert_array_equal(result['nominal_yaw_rate_tolerance_exceed_fraction'], 0.)
