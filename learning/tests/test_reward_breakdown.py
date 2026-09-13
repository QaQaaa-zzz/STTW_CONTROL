from dataclasses import asdict
import json
import numpy as np
import pytest
from sttw_control.env import TaskConfig
from sttw_control.path import CircleConfig
from sttw_control.priority import PriorityConfig
from sttw_control.observation import ObservationConfig


def fixture_trace(path, *, failed=False):
    path.mkdir(parents=True)
    c=TaskConfig(circle=CircleConfig(),priority=PriorityConfig(risk_gate=False),observation=ObservationConfig(include_path=True,include_priority=True),alive_reward_rate=5.)
    n=103;dt=c.controller.dt
    qpos=np.zeros((n,7));qpos[:,3]=1
    qvel=np.zeros((n,6));qvel[:,0]=2
    # Point at the bottom of the circle, tangent heading zero.
    pose=np.tile([c.circle.center_x,c.circle.center_y-c.circle.radius,0.],(n,1))
    reward=np.full(n,5*dt);reward[0]=0;reward[100]+=5
    terminated=np.zeros(n,bool)
    if failed:terminated[50:]=True;n=51;reward[50]=-c.failure_penalty
    trace=dict(time=np.arange(n)*dt,qpos=qpos[:n],qvel=qvel[:n],pose=pose[:n],measurement=np.zeros((n,7)),reference_roll=np.zeros(n),motion_command=np.tile([0.,2.],(n,1)),priority_alpha=np.full(n,.5),action=np.zeros((n,2)),event=np.tile([0,0,1,0,0],(n,1)),reward=reward[:n],terminated=terminated[:n])
    np.savez(path/'trace.npz',**trace)
    (path/'declaration.json').write_text(json.dumps({'config':asdict(c)}))
    (path/'event.json').write_text(json.dumps({'start_seconds':0,'end_seconds':0}))
    return trace


def test_reconstruction_checks_alive_bonus_and_failure(tmp_path):
    from sttw_control.reward_breakdown import reconstruct
    p=tmp_path/'normal';fixture_trace(p)
    x=reconstruct(p)
    assert x['error']<1e-6 and x['reward']['bonus'].sum()==5
    p=tmp_path/'failed';fixture_trace(p,failed=True);x=reconstruct(p)
    assert x['reward']['alive'][-1]==0 and x['reward']['failure'][-1]==-10
    assert x['reward']['bonus'].sum()==0


def test_reconstruction_rejects_changed_logged_reward(tmp_path):
    from sttw_control.reward_breakdown import reconstruct
    z=fixture_trace(tmp_path/'bad');z['reward'][10]+=1
    np.savez(tmp_path/'bad/trace.npz',**z)
    with pytest.raises(ValueError,match='reward mismatch'):reconstruct(tmp_path/'bad')


def test_speed_diagnostic_uses_measured_wheel_rate_not_true_speed(tmp_path):
    from sttw_control.reward_breakdown import reconstruct
    p=tmp_path/'speed';z=fixture_trace(p)
    z['measurement'][:,5]=25.
    np.savez(p/'trace.npz',**z)
    x=reconstruct(p)
    np.testing.assert_allclose(x['speed_true'],2.)
    np.testing.assert_allclose(x['speed_target'],2.)
    np.testing.assert_allclose(x['speed_estimate'],2.5)
