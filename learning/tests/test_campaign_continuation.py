"""Budget and no-evaluation guards for an explicitly extended campaign."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('campaign',Path('learning/cli/asymmetric_priority_comparison.py'))
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def test_continuation_budget_is_total_not_additional():
    original={'updates':48,'num_envs':1024,'rollout_steps':128,'learning_rate':.0003}
    result=campaign.continuation_config(original,250,48,'saved/checkpoint',True)
    assert result['updates']==202
    assert result['resume_checkpoint']=='saved/checkpoint'
    assert result['training_reward_selection'] is True
    assert result['evaluation_reward_best'] is False
    assert original['updates']==48
    assert result['updates']*result['num_envs']*result['rollout_steps']==26476544


def test_fresh_control_uses_total_budget():
    result=campaign.continuation_config({'updates':48},250,0,None,True)
    assert result['updates']==250 and result['resume_checkpoint'] is None


def test_completed_target_refuses_extra_training():
    with pytest.raises(ValueError,match='target'):
        campaign.continuation_config({},250,250,'checkpoint',True)


def test_no_evaluation_disables_first_middle_and_final():
    from sttw_control.training import TrainingConfig,should_validate
    original={'trainer':'rsl','num_envs':1024,'rollout_steps':128,'minibatch_size':32768}
    config=TrainingConfig(**campaign.continuation_config(original,250,48,'checkpoint',True))
    assert not any(should_validate(config,i) for i in (1,24,48,202))
