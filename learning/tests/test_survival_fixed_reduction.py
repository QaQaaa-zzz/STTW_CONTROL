"""Physical policy gradient must not rescale when masked tails change."""
import pytest
import torch
from sttw_control.survival_credit import survival_ppo_loss


def loss(mask, denominator):
    count = len(mask)
    new = torch.zeros(count, requires_grad=True)
    selected = torch.tensor(mask)
    nan = torch.full((count,), float('nan'))
    old = torch.where(selected, torch.zeros(count), nan)
    target = torch.where(selected, torch.ones(count), nan)
    value = torch.where(selected, torch.full((count,), .5), nan)
    result = survival_ppo_loss(new, old, value, target, selected,
                               old_values=value, fixed_denominator=denominator)
    result['actor_loss'].backward()
    return result, new.grad


def test_fixed_episode_horizon_preserves_per_action_gradient_when_tail_is_masked():
    full, full_grad = loss([True]*4, 8)
    short, short_grad = loss([True, True, False, False], 8)
    assert torch.equal(full_grad[:2], short_grad[:2])
    assert torch.equal(short_grad[2:], torch.zeros(2))
    assert full['actor_loss'].item() == -.25
    assert short['actor_loss'].item() == -.125
    assert short['value_loss'].item() == .25  # critic regression stays sample mean


@pytest.mark.parametrize('denominator', [0, -1, 1, float('nan'), float('inf'), True])
def test_invalid_fixed_denominator_is_rejected(denominator):
    with pytest.raises(ValueError, match='denominator'):
        loss([True, True], denominator)
