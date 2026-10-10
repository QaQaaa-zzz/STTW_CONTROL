"""Reward belongs to the sampling model, not the following optimizer result."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('reward_peaks',Path(__file__).parents[1]/'cli/evaluate_path_reward_peaks.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_peak_brackets_sampling_model_not_logged_update():
    rows=[dict(update=125,sampling_model_update=124,mean_step_reward=-.01),dict(update=150,sampling_model_update=149,mean_step_reward=-.1)]
    peak,neighbors=module.peak_neighbors(rows,[100,120,130,150])
    assert peak['sampling_model_update']==124
    assert neighbors==[120,130]

def test_exact_saved_peak_needs_only_one_candidate():
    row=dict(update=121,sampling_model_update=120,mean_step_reward=-.01)
    assert module.peak_neighbors([row],[110,120,130])[1]==[120]

def test_invalid_attribution_and_nonfinite_rewards_rejected():
    with pytest.raises(ValueError):module.peak_neighbors([dict(update=121,sampling_model_update=121,mean_step_reward=-.01)],[120])
    with pytest.raises(ValueError):module.peak_neighbors([dict(update=121,sampling_model_update=120,mean_step_reward=float('nan'))],[120])
