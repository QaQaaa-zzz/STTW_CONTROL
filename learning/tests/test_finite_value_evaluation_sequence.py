"""Original process identity and fixed-size, non-retrying stage receipts."""
import copy
from pathlib import Path
import pytest
from sttw_control.finite_value_evaluation_sequence import validate_training_identity, validate_stage


def identity(tmp_path):
    return dict(cwd=str(tmp_path),argv=['python',str(tmp_path/'learning/cli/train_finite_value.py'),
                                      '--output','runs/training','--phase','execute'])


def test_only_exact_actual_training_execution_can_trigger_the_sequence(tmp_path):
    info=identity(tmp_path)
    validate_training_identity(info,tmp_path/'runs/training',tmp_path)
    for index,value in [(1,str(tmp_path/'elsewhere/train_finite_value.py')),(3,'runs/other'),(5,'shape')]:
        changed=copy.deepcopy(info);changed['argv'][index]=value
        with pytest.raises(ValueError):validate_training_identity(changed,tmp_path/'runs/training',tmp_path)


def test_explicit_boundary_recovery_cli_can_trigger_only_its_own_output(tmp_path):
    info=identity(tmp_path);info['argv'][1]=str(tmp_path/'learning/cli/resume_finite_value.py')
    validate_training_identity(info,tmp_path/'runs/training',tmp_path)
    with pytest.raises(ValueError):validate_training_identity(info,tmp_path/'runs/other',tmp_path)


def test_complete_physics_receipt_preserves_real_vs_allocated_counts():
    validate_stage('execute',dict(phase='completed',stage='physical_complete_analysis_pending',completed_cases=300,
                                 active_lower_ticks=577000,allocated_lower_slots=600000,optimizer_updates=0))


@pytest.mark.parametrize('change', [dict(phase='error'),dict(completed_cases=299),
    dict(active_lower_ticks=600001),dict(allocated_lower_slots=600001),
    dict(active_lower_ticks=600000,allocated_lower_slots=590000),dict(optimizer_updates=1)])
def test_incomplete_or_overbudget_stage_never_advances(change):
    state=dict(phase='completed',stage='physical_complete_analysis_pending',completed_cases=300,
               active_lower_ticks=577000,allocated_lower_slots=600000,optimizer_updates=0)
    state.update(change)
    with pytest.raises(ValueError):validate_stage('execute',state)


def test_preparation_and_shape_cannot_be_mistaken_for_actual_evaluation():
    validate_stage('prepare',dict(phase='prepared',declared_cases=300,optimizer_updates=0))
    validate_stage('shape',dict(phase='completed',stage='shape_complete',optimizer_updates=0))
    with pytest.raises(ValueError):validate_stage('execute',dict(phase='completed',stage='shape_complete',optimizer_updates=0))
    validate_stage('analyze',dict(phase='completed',stage='numeric_and_figures_complete_review_pending',completed_cases=300,
                                  physics_steps=0,optimizer_updates=0))
    with pytest.raises(ValueError):validate_stage('analyze',dict(phase='completed',completed_cases=299))
