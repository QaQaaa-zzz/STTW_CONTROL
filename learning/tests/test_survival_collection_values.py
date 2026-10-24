"""Actual Flax collection values must use the same frozen Torch networks."""
import numpy as np
import torch
import jax
import jax.numpy as j
from sttw_control.path_command_policy import make_path_actor,make_path_critic,make_torch_path_network
from sttw_control.survival_online_learner import OnlineSurvivalLearner
from sttw_control.survival_collection import evaluate_collection_networks


def test_collection_heads_match_torch_at_pre_action_actor_and_critic_views():
    cfg={'network':{'architecture':'alpha_modulated_v1'}}
    torch.manual_seed(4);actor=make_torch_path_network(351,2,cfg);critic=make_torch_path_network(352,1,cfg)
    policy={'actor.'+k:v for k,v in actor.state_dict().items()};policy.update({'critic.'+k:v for k,v in critic.state_dict().items()});policy['log_std']=torch.tensor([-1.,-2.])
    learner=OnlineSurvivalLearner(cfg,policy,beta=1.,seed=5)
    with torch.no_grad():
        learner.physical_critic[0].weight.fill_(.01);learner.physical_critic[0].bias.fill_(.2)
    obs=np.sin(np.arange(351,dtype=np.float32))*.1;view=np.append(obs,np.float32(.7)).astype(np.float32)
    params=learner.sampling_parameters()
    actual=jax.jit(lambda p,x,v:evaluate_collection_networks(make_path_actor(cfg),make_path_critic(cfg),p,x,v))(params,j.asarray(obs),j.asarray(view))
    with torch.no_grad():expected=[learner.actor(torch.tensor(obs)).numpy(),learner.task_critic(torch.tensor(view)).item(),learner.physical_critic(torch.tensor(view)).item()]
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
    assert abs(float(actual[2])-.5)>.01 # catches silently retained legacy .5 placeholder


def test_batched_value_check_uses_each_environments_own_parameters():
    from sttw_control.survival_collection import evaluate_collection_batch
    cfg={'network':{'architecture':'alpha_modulated_v1'}}
    torch.manual_seed(9);actor=make_torch_path_network(351,2,cfg);critic=make_torch_path_network(352,1,cfg)
    policy={'actor.'+k:v for k,v in actor.state_dict().items()};policy.update({'critic.'+k:v for k,v in critic.state_dict().items()});policy['log_std']=torch.tensor([-1.,-2.])
    learners=[OnlineSurvivalLearner(cfg,policy,beta=1.,seed=10) for _ in range(2)]
    with torch.no_grad():
        learners[1].actor[6].bias.add_(.5);learners[1].task_critic[6].bias.add_(1.)
        learners[1].physical_critic[0].bias.add_(.7)
    params=jax.tree.map(lambda *x:j.stack(x),*[x.sampling_parameters() for x in learners])
    obs=j.zeros((2,351));view=j.zeros((2,352))
    actual=jax.jit(lambda p,x,v:evaluate_collection_batch(make_path_actor(cfg),make_path_critic(cfg),p,x,v))(params,obs,view)
    for i,obj in enumerate(learners):
        with torch.no_grad():expected=[obj.actor(torch.zeros(351)).numpy(),obj.task_critic(torch.zeros(352)).item(),obj.physical_critic(torch.zeros(352)).item()]
        for a,b in zip(actual,expected):np.testing.assert_allclose(a[i],b,rtol=1e-5,atol=1e-6)
