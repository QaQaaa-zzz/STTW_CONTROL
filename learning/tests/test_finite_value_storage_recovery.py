import pytest
from sttw_control.finite_value_storage_recovery import validate_storage_source,validate_cached_accounting

def test_only_latest_complete45_disk_error_can_resume():
 s=dict(phase='error',failure_kind='disk_full',completed_updates=45,declared_updates=64)
 validate_storage_source(s,45)
 for b in [17,44,46]:
  with pytest.raises(ValueError):validate_storage_source(s,b)
 with pytest.raises(ValueError):validate_storage_source(dict(s,phase='running'),45)

def test_cached46_is_charged_once_without_replaying_physics():
 pair=dict(active_lower_ticks=15773063,allocated_lower_slots=16200000)
 state=dict(active_lower_ticks=16133063,allocated_lower_slots=16560000)
 cache=dict(phase='completed',authenticated_boundary=45,sampling_identity_reproduced=True,
  all_180_outcomes_known=True,physics_steps=0,optimizer_updates=0,
  cached_actual_lower_ticks=360000,cached_allocated_lower_slots=360000)
 validate_cached_accounting(pair,state,cache)
 for key in ['physics_steps','optimizer_updates']:
  with pytest.raises(ValueError):validate_cached_accounting(pair,state,dict(cache,**{key:1}))
 with pytest.raises(ValueError):validate_cached_accounting(pair,state,dict(cache,cached_actual_lower_ticks=359999))


def test_storage_sequence_binds_only_correct_live_cli_and_output(tmp_path):
 from sttw_control.finite_value_storage_sequence import validate_training_identity
 info=dict(cwd=str(tmp_path),argv=['python',str(tmp_path/'learning/cli/resume_finite_value_storage.py'),'--output','runs/storage','--phase','execute'])
 validate_training_identity(info,tmp_path/'runs/storage',tmp_path)
 with pytest.raises(ValueError):validate_training_identity(info,tmp_path/'runs/other',tmp_path)
 info['argv'][1]=str(tmp_path/'learning/cli/resume_finite_value.py')
 with pytest.raises(ValueError):validate_training_identity(info,tmp_path/'runs/storage',tmp_path)
