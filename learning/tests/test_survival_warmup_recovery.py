"""Recovery may reuse a completed Critic-only update, never replay Actor steps."""
import pytest
from sttw_control.survival_warmup_recovery import validate_first_warmup_states


def states():
    # These are lifecycle receipts, not synthetic physical trajectories.
    done=dict(next_collection=1,actor_steps=0,critic_steps=4,optimizer_steps=[0,4,4],settings={'beta':0.,'epochs':4})
    pending=dict(next_collection=0,actor_steps=0,critic_steps=0,optimizer_steps=[0,0,0],settings={'beta':1.,'epochs':4})
    return done,pending


def test_only_completed_task_critic_and_unstarted_physical_critic_are_recoverable():
    done,pending=states();validate_first_warmup_states(done,pending,epochs=4)


@pytest.mark.parametrize('field,value',[('actor_steps',1),('next_collection',2),('optimizer_steps',[1,4,4])])
def test_actor_updated_or_wrong_boundary_cannot_be_called_first_warmup_reuse(field,value):
    done,pending=states();done[field]=value
    with pytest.raises(ValueError):validate_first_warmup_states(done,pending,epochs=4)


def test_partially_consumed_second_update_is_not_silently_replayed():
    done,pending=states();pending['optimizer_steps']=[0,1,0]
    with pytest.raises(ValueError):validate_first_warmup_states(done,pending,epochs=4)
