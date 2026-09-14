from dataclasses import replace,asdict
import numpy as np
import jax.numpy as jp
from sttw_control.env import RecoveryEnv,load_config
from sttw_control.network import make_policy_identity
from sttw_control.evaluation import evaluate
from sttw_control.reward_breakdown import reconstruct


def test_command_bias_changes_command_without_external_torque():
    c=replace(load_config('learning/configs/bend_recovery.json'),random_events=None,disturbance_start=0.,disturbance_duration=.01,disturbance_rear_torque=2/3,rear_disturbance_mode='command',horizon_seconds=.03)
    env=RecoveryEnv(c);s=env.reset(4);base,_=env.prepare_action(s,jp.zeros(2))
    np.testing.assert_allclose(base[1]-s.base[1],-2/3,atol=2e-6)
    n=env.step(s,jp.zeros(2));assert np.all(n.data.qfrc_applied==0)
    for _ in range(2):n=env.step(n,jp.zeros(2))
    base,_=env.prepare_action(n,jp.zeros(2));np.testing.assert_allclose(base,n.base)
    assert make_policy_identity(env.bundle.identity,asdict(c),10)!=make_policy_identity(env.bundle.identity,asdict(replace(c,rear_disturbance_mode='torque')),10)


def test_unconditioned_reward_reconstructs(tmp_path):
    base=load_config('learning/configs/bend_recovery.json')
    c=replace(base,priority=None,observation=replace(base.observation,include_priority=False),random_events=None,horizon_seconds=.03)
    env=RecoveryEnv(c);evaluate(env,tmp_path/'trace',seed=4)
    result=reconstruct(tmp_path/'trace');assert result['error']<3e-5


def test_reverse_axle_torque_assists_forward_rotation_and_clears():
    base=replace(load_config('learning/configs/bend_recovery.json'),random_events=None,disturbance_start=0.,disturbance_duration=.005,horizon_seconds=.02)
    env=RecoveryEnv(replace(base,disturbance_rear_torque=-2.))
    neutral=RecoveryEnv(base);s=env.reset(4);z=neutral.reset(4)
    n=env.step(s,jp.zeros(2));nz=neutral.step(z,jp.zeros(2))
    assert n.data.qfrc_applied[env.bundle.rear_dof]==-2.
    assert n.data.qvel[env.bundle.rear_dof]<nz.data.qvel[env.bundle.rear_dof]
    np.testing.assert_array_equal(n.data.ctrl,nz.data.ctrl)
    n=env.step(n,jp.zeros(2));assert n.data.qfrc_applied[env.bundle.rear_dof]==0.


def test_assisting_disturbance_reports_surplus_not_only_deficit():
    from sttw_control.speed_recovery import paired_speed
    t=np.arange(0,2.01,.01);v=np.full_like(t,2.1);v[(t>=.5)&(t<1.)]+=.1
    result=paired_speed(t,v,t,np.full_like(t,2.1),start=.5,end=1.,target=2.1)
    assert result['summary']['speed_excursion']
    np.testing.assert_allclose(result['summary']['maximum_extra_rise_m_s'],.1)
    assert result['summary']['deficit_integral_m']==0
    assert result['summary']['surplus_integral_m']>0
