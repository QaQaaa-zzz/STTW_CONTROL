import json
import pytest
from sttw_control.training import TrainingConfig, should_validate
from sttw_control import selection


def test_training_reward_mode_skips_even_final_and_every_update_validation():
    c = TrainingConfig(trainer='rsl', training_reward_selection=True,
                       best_model_every_update=True, updates=3)
    assert not any(should_validate(c, i) for i in (1, 2, 3))


def test_training_reward_mode_requires_supported_trainer():
    with pytest.raises(ValueError, match='RSL'):
        TrainingConfig(training_reward_selection=True)


def test_reward_best_points_to_sampling_policy_and_preserves_ties(tmp_path):
    first = tmp_path/'checkpoints'/'update_0042'
    second = tmp_path/'checkpoints'/'update_0043'
    first.mkdir(parents=True)
    second.mkdir()
    chosen = selection.record_training_reward_best(tmp_path, first, 42, 43, -.2)
    assert chosen['update'] == 42
    assert chosen['sampled_during_update'] == 43
    assert (tmp_path/'best_model').resolve() == first
    selection.record_training_reward_best(tmp_path, second, 43, 44, -.3)
    assert (tmp_path/'best_model').resolve() == first
    selection.record_training_reward_best(tmp_path, second, 43, 44, -.2)
    assert (tmp_path/'best_model').resolve() == first
    chosen = selection.record_training_reward_best(tmp_path, second, 43, 44, -.1)
    assert (tmp_path/'best_model').resolve() == second
    assert chosen['task_success_verified'] is False
    assert chosen['development_gates_passed'] is None
    assert json.loads((tmp_path/'best_model.json').read_text()) == chosen
    with pytest.raises(ValueError, match='finite'):
        selection.record_training_reward_best(tmp_path, first, 42, 45, float('nan'))
