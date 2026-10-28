"""Saved boundary selection must not confuse pre-bind carry and new path."""
import importlib.util
from pathlib import Path
import pytest

AUDIT=Path(__file__).resolve().parents[2]/'docs/evidence/tolerance_progress_20261022/deployment_contract_audit/audit.py'
spec=importlib.util.spec_from_file_location('deployment_coordinator_audit',AUDIT)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_boundary_uses_previous_completed_block_and_only_current_replan():
    decisions=[dict(case_index=3,task_tick=800)]
    before=audit.boundary_files('/saved',3,800,decisions)
    assert before['carry'].name=='complete_carry_03.pkl'
    assert before['paths'].name=='executed_paths_03.pkl'
    assert before['rebind']
    after=audit.boundary_files('/saved',3,1000,decisions)
    assert after['paths'].name=='executed_paths_04.pkl'
    assert not after['rebind']


def test_only_five_declared_saved_boundaries():
    assert len(audit.BOUNDARIES)==5
    assert len({c for c,t in audit.BOUNDARIES})==4
    with pytest.raises(ValueError):audit.boundary_files('/saved',0,204,[])


def test_continuous_bounds_are_absolute_and_channel_specific():
    import numpy as np
    result=audit.compare_boundary_field('coord_latents',np.array([2.79e-5]),np.zeros(1))
    assert result['passed']
    import json
    json.dumps(result)
    assert audit.compare_boundary_field('governed',np.array([.000009,.0000019]),np.zeros(2))['passed']
    assert not audit.compare_boundary_field('governed',np.array([0.,.0000021]),np.zeros(2))['passed']
    # Large absolute outputs cannot acquire a large relative allowance.
    assert not audit.compare_boundary_field('governed',np.array([100.0002,0.]),np.array([100.,0.]))['passed']


def test_discrete_decision_and_nonfinite_are_never_relaxed():
    import numpy as np
    for name in ('alpha_exec','coord_task_mask','rule_latched','coord_permission','q_eligible_08','q_restore_085'):
        assert not audit.compare_boundary_field(name,np.array([1.]),np.array([1.-1e-8]))['passed']
    assert not audit.compare_boundary_field('coord_qmin',np.array([np.nan]),np.array([0.]))['passed']
