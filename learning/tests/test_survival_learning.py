"""Dual-objective PPO must isolate the task-only control and frozen targets."""
import torch
import pytest
from sttw_control.survival_learning import dual_ppo_loss


def build(beta, target=1.):
    new=torch.zeros(2,requires_grad=True)
    old=torch.zeros(2,requires_grad=True)
    task_adv=torch.tensor([1.,-1.],requires_grad=True)
    returns=torch.tensor([2.,3.],requires_grad=True)
    physical=torch.full((2,),target,requires_grad=True)
    loss=dual_ppo_loss(new_log_prob=new,old_log_prob=old,task_advantage=task_adv,
        task_values=torch.zeros(2,requires_grad=True),task_returns=returns,
        physical_values=torch.full((2,),.5,requires_grad=True),physical_targets=physical,
        old_physical_values=torch.full((2,),.5,requires_grad=True),
        beta=beta,fixed_denominator=4)
    return loss,new,old,task_adv,returns,physical


def test_beta_zero_is_task_only_actor_even_when_physical_labels_change():
    a=build(0.,1.);b=build(0.,0.)
    ga=torch.autograd.grad(a[0]['actor_loss'],a[1])[0]
    gb=torch.autograd.grad(b[0]['actor_loss'],b[1])[0]
    torch.testing.assert_close(ga,gb)
    torch.testing.assert_close(ga,torch.tensor([-.25,.25]))


def test_combined_actor_is_sum_and_collection_targets_are_detached():
    loss,new,*frozen=build(1.)
    loss['total_loss'].backward()
    torch.testing.assert_close(new.grad,torch.tensor([-.375,.125]))
    assert all(x.grad is None for x in frozen)


@pytest.mark.parametrize('beta',[-1.,float('nan'),float('inf')])
def test_bad_objective_weight_rejected(beta):
    with pytest.raises(ValueError):build(beta)
