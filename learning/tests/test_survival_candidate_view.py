"""A candidate evaluation view must retain updated weights, not parent weights."""
import torch
import pytest
from sttw_control.survival_candidate_evaluation import candidate_policy_view


def test_candidate_view_preserves_updated_actor_critic_and_distribution():
    actor=torch.nn.Linear(3,2);critic=torch.nn.Linear(4,1);log_std=torch.tensor([-.4,-1.2])
    candidate=dict(actor=actor.state_dict(),task_critic=critic.state_dict(),log_std=log_std)
    view=candidate_policy_view(candidate)
    restored=torch.nn.Linear(3,2)
    restored.load_state_dict({k[6:]:v for k,v in view['policy'].items() if k.startswith('actor.')})
    x=torch.tensor([[.1,.2,.3]])
    torch.testing.assert_close(restored(x),actor(x))
    torch.testing.assert_close(view['policy']['log_std'],log_std)
    assert view['exact_training_resume'] is False
    assert 'optimizer' not in view


def test_missing_updated_distribution_cannot_fall_back_to_parent():
    with pytest.raises(ValueError):candidate_policy_view(dict(actor={},task_critic={}))
