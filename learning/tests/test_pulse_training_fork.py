import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from sttw_control.tolerance_training import pulse_training_config,restore_pulse_parent
from sttw_control.preference_best import digest


def test_config_changes_schedule_only():
    parent=dict(path_reward={'scale':.01},training_future_phase_C={'num_envs':512,'rollout_steps':128,'episode_s':20,'updates_initial':50,'updates_max':50,'validation_updates':[],'save_every':5})
    original=deepcopy(parent);cfg=pulse_training_config(parent)
    assert parent==original and cfg['path_reward']==parent['path_reward']
    assert cfg['training_future_phase_C']['updates_initial']==25
    cfg['training_future_phase_C']['updates_initial']=50;cfg['training_future_phase_C']['updates_max']=50
    assert cfg==parent
    parent['path_reward']['tolerance']={'weight':10}
    with pytest.raises(ValueError):pulse_training_config(parent)


def test_parent_restore_and_wrong_endpoint_rejection(tmp_path):
    policy=torch.nn.Linear(2,1);target=torch.nn.Linear(2,1);opt=torch.optim.Adam(target.parameters())
    identity=dict(arm='baseline',alpha=0,lower_actor_sha256='frozen')
    p=tmp_path/'parent.pt';torch.save(dict(identity=identity,update=50,policy=policy.state_dict()),p)
    record=dict(checkpoint=str(p),checkpoint_sha256=digest(p),identity=identity)
    algo=SimpleNamespace(policy=target,optimizer=opt);source={'candidate':{'actor_sha256':'frozen'}}
    restore_pulse_parent(algo,record,source,0)
    assert not opt.state
    for k,v in policy.state_dict().items():assert torch.equal(target.state_dict()[k],v)
    with pytest.raises(ValueError,match='wrong pulse parent'):restore_pulse_parent(algo,record,source,1)
    bad=dict(record,checkpoint_sha256='wrong')
    with pytest.raises(ValueError,match='checksum'):restore_pulse_parent(algo,bad,source,0)
