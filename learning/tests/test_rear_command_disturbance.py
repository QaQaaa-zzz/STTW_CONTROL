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
