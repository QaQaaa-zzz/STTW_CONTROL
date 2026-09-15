"""Execute pure config/metadata functions without requiring a physics install."""
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
SRC = ROOT/'src/sttw_control'


def extract(path, names, globals):
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(keep) == len(names)
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),globals)
    return globals


def obs_namespace():
    tree = ast.parse((SRC/'observation.py').read_text())
    ns = {'dataclass':dataclass}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('FIELDS','PATH_FIELDS','TRACKING_FIELDS') for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'observation.py','exec'),ns)
    return extract(SRC/'observation.py',{'ObservationConfig','observation_fields'},ns)


def test_actor_alpha_context_and_normalization_have_identical_order():
    ns = obs_namespace()
    config = ns['ObservationConfig'](history_steps=10, include_path=True, include_priority=True,
                                     include_tracking=True, include_attitude_risk=False)
    fields = ns['observation_fields'](config)
    assert len(fields) == 24 and fields[-1] == 'speed_priority'
    assert fields[18:23] == ns['TRACKING_FIELDS']
    normal = extract(SRC/'training.py',{'normalization'}, {'np':np})['normalization']
    mean, std = normal(SimpleNamespace(observation=config))
    assert mean.shape == std.shape == (250,)
    np.testing.assert_array_equal(std[18:24], np.ones(6))


def identity_function():
    ns=obs_namespace()
    ns.update(hashlib=hashlib,json=json,ACTION_FIELDS=[])
    return extract(SRC/'network.py',{'make_policy_identity'},ns)['make_policy_identity']


def test_optional_fields_do_not_invalidate_legacy_checkpoints():
    identity=identity_function()
    old={'observation':{'history_steps':10,'include_path':True,'include_priority':True,'include_attitude_risk':False},
         'priority':{'risk_gate':False}, 'alive_reward_rate':1.}
    current=json.loads(json.dumps(old))
    current['tracking_reward']=None
    current['observation']['include_tracking']=False
    assert identity({'model':'unchanged'},old,10)==identity({'model':'unchanged'},current,10)


def test_new_reward_and_context_bind_checkpoint_and_keep_alpha():
    identity=identity_function()
    config=json.loads((ROOT/'configs/path_recovery.json').read_text())
    x=identity({},config,10)
    assert x['observation_fields'][-1]=='speed_priority'
    assert len(x['observation_fields'])==24
    assert x['observation_fields'][15]=='path_right_error'
    changed=json.loads(json.dumps(config));changed['tracking_reward']['tracking_rate']=5.
    assert identity({},changed,10)!=x


def test_ppo_config_loads_and_budget_is_4194304_not_100_long_updates():
    cls=extract(SRC/'training.py',{'TrainingConfig'},{'dataclass':dataclass,'math':math})['TrainingConfig']
    config=cls(**json.loads((ROOT/'configs/ppo_path_recovery.json').read_text()))
    assert config.num_envs*config.rollout_steps*config.updates==4194304
    assert config.epochs*config.num_envs*config.rollout_steps//config.minibatch_size==16
    assert config.learning_rate==.0003 and config.gamma==.9995


def test_new_task_requires_alpha_and_does_not_silently_replace_command():
    task=json.loads((ROOT/'configs/path_recovery.json').read_text())
    assert task['observation']['include_priority'] is True
    assert task['priority']['risk_gate'] is False
    assert task['actuator']['composition']=='additive'
    assert task.get('motion_commands') is None and 'bend' in task
    assert task['horizon_seconds']==10 and task['learning_roll_reference'] is None
    assert task['random_events']['start_max']+task['random_events']['duration_max']+3 <= 10


def test_selection_is_safety_and_nominal_gated_not_just_max_reward():
    rank=extract(SRC/'tracking_validation.py',{'rank_tracking_candidate'},{'np':np})['rank_tracking_candidate']
    base={k:[0.] for k in ('failed','nominal_failed','nominal_radial_rmse','nominal_speed_rmse',
                            'deadline_missed','speed_budget_fraction','path_budget_fraction')}
    base.update(final_tracking_hold=[1.],episode_return=[0.])
    good,_=rank(base,base)
    unsafe=dict(base,failed=[1.],episode_return=[100.])
    bad,_=rank(unsafe,base)
    assert good<bad
    regress=dict(base,nominal_speed_rmse=[.2],episode_return=[100.])
    assert good<rank(regress,base)[0]
    assert rank(dict(base,episode_return=[float('nan')]),base)[0] is None
