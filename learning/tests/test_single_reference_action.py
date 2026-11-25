"""Exercise the upstream PPO distribution, storage and channel expansion on CPU."""
import sys

import pytest
import torch

sys.path.insert(0,'/home/qy/projects/My_unitree_go2_gym/rsl_rl')
from rsl_rl.algorithms import PPO
from sttw_control.go2_ppo_adapter import H75ActorCritic
from sttw_control.reference_action_channel import CHANNEL_INDEX, expand_reference_action
from sttw_control.torch_path_policy import make_torch_path_network


@pytest.mark.parametrize('channel',[None,'speed','steer'])
def test_rollout_storage_only_optimizes_enabled_coordinates(channel):
    torch.manual_seed(18)
    cfg={'network':{'architecture':'alpha_modulated_v1'}}
    actor=make_torch_path_network(351,2,cfg)
    critic=make_torch_path_network(352,1,cfg)
    net=H75ActorCritic(actor,critic,torch.tensor([.25,.1]).log(),active_channel=channel)
    alg=PPO(net)
    index=CHANNEL_INDEX.get(channel)
    n=2 if index is None else 1
    alg.init_storage(4,5,[351],[352],[n])
    initial=actor[-1].weight.detach().clone()
    for t in range(5):
        obs=torch.randn(4,351)
        actions=alg.act(obs,torch.nn.functional.pad(obs,(0,1)))
        assert actions.shape==alg.transition.action_mean.shape==alg.transition.action_sigma.shape==(4,n)
        dist=torch.distributions.Normal(actor(obs)[...,net.action_slice],net.std)
        torch.testing.assert_close(net.get_actions_log_prob(actions),dist.log_prob(actions).sum(-1))
        torch.testing.assert_close(net.entropy,dist.entropy().sum(-1))
        physical=expand_reference_action(actions,index)
        if index is None:assert physical is actions
        else:
            assert not physical[:,1-index].count_nonzero()
            torch.testing.assert_close(physical[:,index],actions[:,0])
        alg.process_env_step(torch.arange(1,5).float(),torch.full((4,),t==4),{})
    assert alg.storage.actions.shape==(5,4,n)
    alg.compute_returns(torch.zeros(4,352)); alg.update()
    after=actor[-1].weight.detach()
    if index is None: assert all((after!=initial).any(1))
    else:
        assert torch.equal(after[1-index],initial[1-index])
        assert torch.any(after[index]!=initial[index])
