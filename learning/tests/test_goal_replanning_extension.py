"""H75 extension refuses pending baselines or changed physical/planning code."""
import json
from pathlib import Path
import pytest
from sttw_control.preference_best import atomic_json,digest
from sttw_control.goal_replanning_run import prepare_h75


def baseline(tmp_path,*,status,changed_module=None):
    root=tmp_path/'baseline';root.mkdir();(root/'physical').mkdir()
    sha={}
    if changed_module:
        f=tmp_path/changed_module;f.write_text('old control source\n');sha[str(f)]=digest(f)
        f.write_text('changed control source\n')
    atomic_json(root/'plan.json',dict(input_sha256={},source_sha256=sha))
    atomic_json(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'physical/status.json',status)
    return root


def test_pending_baseline_is_never_replaced_or_implicitly_reevaluated(tmp_path):
    source=baseline(tmp_path,status=dict(phase='running',completed_cases=24))
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='completed48'):
        prepare_h75(source,tmp_path/'historical',out)
    assert not out.exists()
    assert json.loads((source/'physical/status.json').read_text())['phase']=='running'


@pytest.mark.parametrize('name',['goal_replanning.py','path_command_env.py','controller.py'])
def test_control_or_planner_drift_blocks_cache_reuse_before_model_loading(tmp_path,name):
    source=baseline(tmp_path,status=dict(phase='completed',completed_cases=48),changed_module=name)
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='controller/planner source changed'):
        prepare_h75(source,tmp_path/'historical',out)
    assert not out.exists()
