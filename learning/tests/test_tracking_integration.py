"""CPU integration and legacy zero-residual parity; needs project dependencies."""
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('flax')
pytest.importorskip('optax')
pytest.importorskip('mujoco')
from sttw_control.env import RecoveryEnv, load_config
from sttw_control.network import make_policy_identity, ResidualActor
from sttw_control.training import normalization
from sttw_control.observation import observation_fields
from sttw_control.tracking_reward import reward_components

TASK=Path(__file__).parents[1]/'configs/path_recovery.json'


def test_config_rejects_hidden_alpha_or_command_task():
    c=load_config(TASK)
    with pytest.raises(ValueError):replace(c,observation=replace(c.observation,include_priority=False))
    with pytest.raises(ValueError):replace(c,priority=replace(c.priority,risk_gate=True))
    with pytest.raises(ValueError):replace(c,tracking_reward=None)


def test_cpu_zero_residual_preserves_same_physics_and_baseline():
    c=load_config(TASK)
    old=replace(c,tracking_reward=None,observation=replace(c.observation,include_tracking=False))
    env,legacy=RecoveryEnv(c),RecoveryEnv(old)
    a,b=env.reset(7001),legacy.reset(7001)
    assert a.obs.shape==(250,)
    mean,std=normalization(c)
    assert mean.shape==std.shape==a.obs.shape
    for _ in range(20):
        a,b=env.step(a,np.zeros(2)),legacy.step(b,np.zeros(2))
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
        np.testing.assert_array_equal(a.base,b.base)
        np.testing.assert_array_equal(a.actuator.previous,b.actuator.previous)
        assert float(a.reward)==pytest.approx(sum(float(v) for v in a.reward_parts.values()),abs=1e-6)


def test_alpha_is_visible_but_cannot_change_actuator_authority_for_same_action():
    env=RecoveryEnv(load_config(TASK));s=env.reset(7002)
    i=observation_fields(env.config.observation).index('speed_priority')
    states=[env.set_priority(s,x) for x in [0.,.5,1.]]
    for a,x in zip(states,[0.,.5,1.]):assert float(a.history.frames[-1,i])==x
    action=np.array([.1,-.1])
    ns=[env.step(a,action) for a in states]
    for n in ns[1:]:np.testing.assert_array_equal(n.data.qpos,ns[0].data.qpos)


def test_cpu_actual_transition_reward_reconstruction_and_reset():
    env=RecoveryEnv(load_config(TASK));s=env.reset(7003)
    action=np.array([.03,-.04],dtype=np.float32)
    n=env.step(s,action);c=env.config
    path=np.asarray(env.path_features(n.pose));speed=float(np.dot(n.data.qvel[:3],n.data.xmat[env.bundle.chassis].reshape(3,3)[:,0]))
    parts=reward_components(float(n.measurement[0]),float(n.measurement[1]),speed-c.speed_reference,
        path[0],path[1],action,np.asarray(s.tracking.previous_action),float(s.priority_alpha),
        n.tracking,c.tracking_reward,c.controller.dt,c.alive_reward_rate,c.failure_penalty,bool(n.terminated))
    assert sum(parts.values())==pytest.approx(float(n.reward),rel=3e-5,abs=1e-6)
    fresh=env.reset(7003)
    assert np.all(np.asarray(fresh.tracking.previous_action)==0)
    assert not bool(fresh.tracking.pending)
