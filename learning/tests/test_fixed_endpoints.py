from dataclasses import replace
import numpy as np
import jax.numpy as jp
import pytest
from sttw_control.controller import ControllerConfig,initial_controller,controller_step
from sttw_control.env import load_config,RecoveryEnv
from sttw_control.observation import observation_fields
from sttw_control.tracking_reward import TrackingConfig,initial_return,transition


def test_fixed_roll_target_changes_controller_and_keeps_dynamic_default():
    c=ControllerConfig(fixed_roll_reference=.12)
    s=initial_controller(c)
    _,a=controller_step(s,jp.array([2.,.1,0.,.05,0.,.08]),False,c)
    _,b=controller_step(s,jp.array([2.,.1,0.,.05,0.,.16]),False,c)
    assert float(a.reference_roll)==pytest.approx(.12)
    assert float(b.reference_roll)==pytest.approx(.12)
    _,d=controller_step(s,jp.array([2.,.1,0.,.05,0.,.08]),False,ControllerConfig())
    assert float(d.reference_roll)!=pytest.approx(.12)


def test_fixed_alpha_can_be_hidden_but_random_alpha_cannot():
    c=load_config('learning/configs/soft_budget_ecbc1.json')
    obs=replace(c.observation,include_priority=False)
    fixed=replace(c,priority=replace(c.priority,randomize_alpha=False,fixed_alpha=0.),observation=obs,preparation_seconds=0.)
    assert 'speed_priority' not in observation_fields(fixed.observation)
    env=RecoveryEnv(fixed);state=env.reset(3)
    assert state.obs.shape==(300,)
    assert float(state.priority_alpha)==0.
    with pytest.raises(ValueError):replace(fixed,priority=replace(fixed.priority,randomize_alpha=True))


def test_fixed_roll_reward_and_final_band_use_target():
    c=TrackingConfig(fixed_roll_reference=.12,roll_target_weight=8.,roll_target_scale=.1,roll_target_tolerance=.04)
    def calc(roll):
        return transition(initial_return(xp=np),roll=roll,roll_rate=0.,speed_error=0.,lateral_error=0.,heading_error=0.,action=np.zeros(2),alpha=0.,dt=.005,alive_rate=1.,failure_penalty=100.,failed=False,enabled=True,config=c,xp=np)
    on,good=calc(.12);off,bad=calc(0.)
    assert good['roll_target']==0. and bad['roll_target']<0.
    assert not on.pending and off.pending


def test_random_turn_requests_allow_same_direction_bounds_and_slew():
    import jax
    from sttw_control.timed_reference import TimedReferenceConfig,schedule,command_at
    c=TimedReferenceConfig(yaw_rate_min=.45,yaw_rate_max=.85)
    rows=np.asarray(schedule(jax.random.PRNGKey(7),c,2.3))
    assert np.all((rows[1:,2]>=.45)&(rows[1:,2]<=.85))
    prev=jp.array([2.3,0.])
    nxt=command_at(500,prev,jp.asarray(rows),.005,c)
    assert abs(float(nxt[0]-prev[0]))<=c.speed_slew*.005+1e-6
    assert abs(float(nxt[1]-prev[1]))<=c.yaw_slew*.005+1e-6


def test_endpoint_rewards_keep_path_and_encourage_different_choices():
    from sttw_control.tracking_reward import directional_speed_path_rates
    from sttw_control.training import normalization
    from sttw_control.network import make_policy_identity
    from dataclasses import asdict
    for alpha in [0,1]:
        cfg=load_config(f'learning/configs/fixed_endpoint_alpha{alpha}.json');c=cfg.tracking
        def score(ev,ey,ep):return sum(directional_speed_path_rates(ev,ey,ep,alpha,0.,c,xp=np).values())
        assert score(0.,.04,0.)<score(0.,0.,0.)
        slow=score(-.4,.03,.02);fast=score(0.,.15,.08)
        assert (slow>fast) if alpha==0 else (fast>slow)
        assert score(-1.5,0.,0.)<score(-.4,0.,0.)
        identity=make_policy_identity({},asdict(cfg),10)
        assert 'speed_priority' not in identity['observation_fields']
        assert len(normalization(cfg)[0])==300


def test_hidden_alpha_evaluation_can_lock_only_its_declared_preference():
    env=RecoveryEnv(load_config('learning/configs/fixed_endpoint_alpha0.json'));state=env.reset(4)
    locked=env.set_priority(state,0.)
    np.testing.assert_array_equal(locked.obs,state.obs)
    assert bool(locked.priority_locked)
    with pytest.raises(ValueError,match='fixed'):env.set_priority(state,1.)


def test_fixed_roll_controller_has_consistent_zero_error_equilibrium():
    from sttw_control.controller import _system
    c=ControllerConfig(fixed_roll_reference=.12)
    state=initial_controller(c)
    *_,ratio,gains=_system(2.3,state.gains,c)
    steer=ratio*.12
    _,out=controller_step(state,jp.array([2.3,steer,0.,.12,0.,0.]),False,c)
    assert abs(float(out.steer_rate))<1e-6
