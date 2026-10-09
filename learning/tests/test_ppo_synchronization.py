import copy
import torch
from test_preference_v5_ppo import algorithm
from test_direct_command_ppo import _obs,_rollout
from sttw_control.direct_command_ppo import _finite_tensors


def test_zero_temporal_coefficient_skips_actor_but_keeps_pairs_and_rng():
    a=algorithm();a.prepare_temporal_pairs()
    rng=torch.get_rng_state().clone();calls=[]
    handle=a.policy.actor.register_forward_hook(lambda *args:calls.append(True))
    try:loss=a.temporal_loss(torch.arange(len(a.temporal_left)))
    finally:handle.remove()
    assert not calls
    assert a.temporal_denominator==8
    assert torch.equal(torch.get_rng_state(),rng)
    assert loss.requires_grad and float(loss.detach())==0


def test_finite_empty_and_nonfinite():
    assert _finite_tensors()
    assert _finite_tensors(torch.ones(3),torch.empty(0))
    assert not _finite_tensors(torch.ones(3),torch.tensor(float('nan')))
    assert not _finite_tensors(torch.tensor(float('inf')),torch.ones(3))
