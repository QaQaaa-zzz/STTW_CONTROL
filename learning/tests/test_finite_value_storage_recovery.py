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


def test_latest_host_reboot55_can_resume_but_not_stale54():
 s=dict(phase='error',failure_kind='host_reboot',completed_updates=55,declared_updates=64)
 validate_storage_source(s,55)
 with pytest.raises(ValueError):validate_storage_source(s,54)


def test_recursive_collection_provenance_and_cached_block_accounting(tmp_path):
 import json
 from sttw_control.finite_value_storage_recovery import collection_folder,collection_allocated_slots
 from sttw_control.preference_best import digest
 original=tmp_path/'original';second=tmp_path/'second';third=tmp_path/'third'
 for p in [original,second,third]:p.mkdir()
 (original/'plan.json').write_text(json.dumps({}))
 (second/'plan.json').write_text(json.dumps(dict(resume_boundary=17,resume_source_run=str(original),resume_source_plan_sha256=digest(original/'plan.json'))))
 (third/'plan.json').write_text(json.dumps(dict(resume_boundary=45,resume_source_run=str(second),resume_source_plan_sha256=digest(second/'plan.json'))))
 assert collection_folder(third,1)==original/'execution/update_0001'
 assert collection_folder(third,30)==second/'execution/update_0030'
 assert collection_folder(third,46)==third/'execution/update_0046'
 cached=third/'execution/update_0046';cached.mkdir(parents=True)
 (cached/'cached_physics_receipt.json').write_text(json.dumps(dict(chunk_files=[{}]*10)))
 assert collection_allocated_slots(cached)==360000
 (cached/'chunk_00.npz').touch()
 with pytest.raises(ValueError):collection_allocated_slots(cached)


def test_abandoned_partial_cost_is_reported_separately_and_in_total():
 from sttw_control.finite_value_storage_recovery import reported_costs
 r=reported_costs(dict(active_lower_ticks=23000000,allocated_lower_slots=23040000,
  interrupted_active_lower_ticks=108000,interrupted_allocated_lower_slots=108000))
 assert r['recorded_total_active_lower_ticks']==23108000
 assert r['recorded_total_allocated_lower_slots']==23148000
 assert r['allocated_lower_slots']==23040000


def test_only_coordinator_source_may_change_with_authenticated_old_snapshot(tmp_path):
 from sttw_control.finite_value_storage_recovery import authenticate_previous_runtime
 from sttw_control.preference_best import digest
 source=tmp_path/'source';work=tmp_path/'work';source.mkdir();work.mkdir()
 coordinator=work/'learning/src/sttw_control/finite_value_storage_recovery.py';coordinator.parent.mkdir(parents=True)
 core=coordinator.with_name('survival_live_collection.py')
 for f in [coordinator,core]:
  f.write_text('original');copy=source/'source_at_launch'/f.relative_to(work);copy.parent.mkdir(parents=True,exist_ok=True);copy.write_text('original')
 plan=dict(source_sha256={str(f):digest(f) for f in [coordinator,core]})
 coordinator.write_text('updated recovery wiring')
 audit=authenticate_previous_runtime(source,plan,workdir=work)
 assert len(audit)==1
 core.write_text('changed physics')
 with pytest.raises(ValueError):authenticate_previous_runtime(source,plan,workdir=work)
