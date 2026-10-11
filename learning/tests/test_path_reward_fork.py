import torch
import pytest
from types import SimpleNamespace
from sttw_control.path_command_fork import load_weights_only, physical_config_identity

def algorithm(seed):
    torch.manual_seed(seed)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.actor=torch.nn.Linear(3,2);self.critic=torch.nn.Linear(4,1);self.log_std=torch.nn.Parameter(torch.zeros(2))
    p=Policy();return SimpleNamespace(policy=p,optimizer=torch.optim.Adam(p.parameters()))

def test_weights_only_preserves_fresh_critic_optimizer_and_rng():
    parent=algorithm(3);a=algorithm(4);b=algorithm(4)
    initial={k:v.clone() for k,v in a.policy.state_dict().items()};rng=torch.get_rng_state().clone()
    saved={'policy':parent.policy.state_dict(),'optimizer':{'inherited':'forbidden'}}
    load_weights_only(a,saved);load_weights_only(b,saved)
    assert torch.equal(rng,torch.get_rng_state()) and a.optimizer.state_dict()['state']=={}
    for k,v in a.policy.state_dict().items():
        assert torch.equal(v,b.policy.state_dict()[k])
        assert torch.equal(v,parent.policy.state_dict()[k] if k.startswith('actor.') or k=='log_std' else initial[k])
    a.optimizer.state[next(a.policy.parameters())]={'bad':'state'}
    with pytest.raises(ValueError,match='fresh'):load_weights_only(a,saved)
    del saved['policy']['log_std']
    with pytest.raises(ValueError,match='keys'):load_weights_only(b,saved)

def test_physical_identity_excludes_only_reward_selection_and_schedule():
    old={'path_reward':{'weight':1},'task_selection_schema':'old','timing':{'dt':.005},'training_future_phase_C':{'validation_updates':[1]}}
    new={**old,'path_reward':{'weight':2},'task_selection_schema':'new','training_future_phase_C':{'validation_updates':[50]}}
    assert physical_config_identity(old)==physical_config_identity(new)
    new['timing']={'dt':.006}
    assert physical_config_identity(old)!=physical_config_identity(new)

def test_real_endpoint_architecture_same_fresh_critic_and_adam_for_both_arms():
    import json
    import numpy as np
    import random
    from pathlib import Path
    from tensordict import TensorDict
    from sttw_control.path_command_training import load_plan,make_algorithm
    root=Path(__file__).parents[2]
    configs=[load_plan(root/f'learning/configs/path_safety_{arm}_run.json')[1] for arm in ('control','candidate')]
    record=json.loads((root/'runs/path_reward_peak_review_20261011/alpha0/best_task_model.json').read_text())
    saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    learners=[];rngs=[]
    for c in configs:
        torch.manual_seed(911101);np.random.seed(911101);random.seed(911101)
        obs=TensorDict({'policy':torch.zeros(8,351),'critic':torch.zeros(8,352)},batch_size=[8])
        learner=make_algorithm(obs,16,c,'cpu');load_weights_only(learner,saved)
        learners.append(learner);rngs.append((torch.get_rng_state(),np.random.get_state(),random.getstate()))
    for k,v in learners[0].policy.state_dict().items():assert torch.equal(v,learners[1].policy.state_dict()[k])
    assert learners[0].optimizer.state_dict()==learners[1].optimizer.state_dict()
    assert learners[0].optimizer.state_dict()['state']=={}
    assert torch.equal(rngs[0][0],rngs[1][0]) and rngs[0][2]==rngs[1][2]
    np.testing.assert_array_equal(rngs[0][1][1],rngs[1][1][1])
    assert any(not torch.equal(v,saved['policy'][k]) for k,v in learners[0].policy.state_dict().items() if k.startswith('critic.'))
