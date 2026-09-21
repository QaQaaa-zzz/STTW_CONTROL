"""Real CPU transitions: finite terminal, reward reconstruction and speed schema."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
import pytest
from sttw_control.env import RecoveryEnv, config_from_dict
from sttw_control.observation import observation_fields
from sttw_control.priority_return_v2 import reward_terms
from sttw_control.network import make_policy_identity


def config(objective='priority_return_v2', speed='wheel'):
    c=json.loads(Path('learning/configs/soft_budget_ecbc1.json').read_text())
    c['observation']['include_priority_v2']=True
    c['tracking']['objective']=objective
    c['forward_speed_source']=speed
    c['preparation_seconds']=0.
    c['horizon_seconds']=.02
    return config_from_dict(c)


def test_finite_end_is_not_physical_failure_and_rewards_reconstruct(tmp_path):
    from sttw_control.evaluation import evaluate
    c=config();env=RecoveryEnv(c)
    evaluate(env,tmp_path/'trace',seed=66)
    t=dict(np.load(tmp_path/'trace/trace.npz'))
    assert t['truncated'][-1] and not t['terminated'][-1]
    assert not t['physical_failed'][-1]
    assert 'priority_v2_q' in t
    from sttw_control.priority_v2_analysis import replay
    _,_,parts=replay(t,asdict(c))
    np.testing.assert_allclose(t['reward'][1:],[sum(p.values()) for p in parts],rtol=1e-5,atol=1e-6)


def test_oracle_changes_only_new_speed_field_and_identity():
    wheel=RecoveryEnv(config());true=RecoveryEnv(config(speed='true'))
    a=wheel.reset(66);b=true.reset(66)
    for act in [np.array([.1,-.2]),np.zeros(2)]:
        a=wheel.step(a,act);b=true.step(b,act)
        np.testing.assert_array_equal(a.base,b.base)
        np.testing.assert_array_equal(a.measurement,b.measurement)
        np.testing.assert_array_equal(a.history.frames[:,:-6],b.history.frames[:,:-6])
        np.testing.assert_allclose(a.reward,b.reward)
    fields=observation_fields(wheel.config.observation)
    assert (len(fields)+1)*10==a.obs.size
    assert make_policy_identity(wheel.bundle.identity,asdict(wheel.config),10)!=make_policy_identity(true.bundle.identity,asdict(true.config),10)


def test_a_auxiliary_state_keeps_v1_reward_identical():
    c=config('soft_budget_v1');old=replace(c,observation=replace(c.observation,include_priority_v2=False))
    env=RecoveryEnv(c);legacy=RecoveryEnv(old);a=env.reset(66);b=legacy.reset(66)
    for action in [np.array([.1,-.2]),np.array([-.1,.1])]:
        a=env.step(a,action);b=legacy.step(b,action)
        np.testing.assert_array_equal(a.reward,b.reward)
        assert a.tracking_components.keys()==b.tracking_components.keys()
        for key in a.tracking_components:np.testing.assert_array_equal(a.tracking_components[key],b.tracking_components[key])
