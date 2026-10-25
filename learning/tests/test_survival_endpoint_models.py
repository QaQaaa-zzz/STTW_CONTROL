"""Native models and exported evaluation Actors must identify the same weights."""
import copy
import json
import numpy as np
import pytest
import torch
from sttw_control.path_command_policy import make_torch_path_network, export_torch_actor
from sttw_control.survival_online_learner import OnlineSurvivalLearner
from sttw_control.survival_endpoint_models import verify_native_export, authenticate_endpoint


def model():
    cfg={'network':{'architecture':'alpha_modulated_v1'}}
    torch.manual_seed(13)
    actor=make_torch_path_network(351,2,cfg);critic=make_torch_path_network(352,1,cfg)
    policy={'actor.'+k:v for k,v in actor.state_dict().items()}
    policy.update({'critic.'+k:v for k,v in critic.state_dict().items()})
    policy['log_std']=torch.tensor([-1.,-2.])
    learner=OnlineSurvivalLearner(cfg,policy,beta=1.,seed=17)
    record=dict(state_sha256=learner.fingerprint(),sampling_sha256=learner.fingerprint(sampling_only=True))
    return cfg,learner,record


def test_native_export_check_preserves_rng_and_reads_actual_saved_distribution():
    cfg,obj,record=model(); rng=torch.get_rng_state().clone()
    result=verify_native_export(cfg,obj.checkpoint(),export_torch_actor(obj.actor,cfg),record)
    assert torch.equal(torch.get_rng_state(),rng)
    np.testing.assert_allclose(result['log_std'],[-1.,-2.])


@pytest.mark.parametrize('change',['export','native','record'])
def test_wrong_export_or_native_fingerprint_is_rejected(change):
    cfg,obj,record=model(); native=obj.checkpoint(); exported=copy.deepcopy(export_torch_actor(obj.actor,cfg))
    if change=='export': exported['params']['Dense_0']['bias']=exported['params']['Dense_0']['bias']+.01
    if change=='native': native['log_std'][0]+=.01
    if change=='record': record['sampling_sha256']='wrong'
    with pytest.raises(ValueError):verify_native_export(cfg,native,exported,record)


def test_live_training_cannot_be_authenticated_as_a_terminal_endpoint(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='running',completed_updates=10)))
    with pytest.raises(ValueError,match='completed64'):
        authenticate_endpoint(tmp_path)
