"""Refuse mismatched native evidence before constructing a physical runtime."""
import json
import pytest
from sttw_control.native_zero_comparison import digest, prepare, run, save, verify


def test_incomplete_native_panel_is_not_replaced(tmp_path):
    source=tmp_path/'source';(source/'evaluation').mkdir(parents=True)
    save(source/'evaluation/status.json',dict(phase='running',completed_cases=4))
    output=tmp_path/'new'
    with pytest.raises(ValueError,match='completed native current8'):
        prepare(source,output)
    assert not output.exists()


@pytest.mark.parametrize('category',['input_sha256','native_source_sha256','runner_sha256'])
def test_changed_bound_file_stops_before_runtime(tmp_path,category):
    root=tmp_path/'run';root.mkdir();f=tmp_path/'bound.py';f.write_text('old')
    plan=dict(input_sha256={},native_source_sha256={},runner_sha256={})
    plan[category]={str(f):digest(f)};save(root/'plan.json',plan)
    save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'cached_records.json',[]);f.write_text('changed')
    with pytest.raises(ValueError,match='sealed input/source changed'):
        verify(root)


def test_modern_package_cannot_silently_supply_native_physics(tmp_path):
    root=tmp_path/'run';root.mkdir()
    save(root/'plan.json',dict(native_runtime_root=str(tmp_path/'native'),input_sha256={},native_source_sha256={},runner_sha256={}))
    save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'cached_records.json',[]);save(root/'status.json',dict(phase='prepared'))
    with pytest.raises(ValueError,match='modern sttw_control already imported'):
        run(root)
    assert not (root/'evaluation').exists()
