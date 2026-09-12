import pytest


def test_gate_requires_success_identity_and_idle_resources():
    from sttw_control.deferred_training import gate
    assert gate({'phase':'running','plan_sha256':'x'},'x',[],[])=='waiting_dependency'
    assert gate({'phase':'complete','plan_sha256':'x'},'x',[1],[])=='waiting_processes'
    assert gate({'phase':'complete','plan_sha256':'x'},'x',[],[2])=='waiting_processes'
    assert gate({'phase':'complete','plan_sha256':'x'},'x',[],[])=='ready'
    for status in [{'phase':'error','plan_sha256':'x'},{'phase':'complete','plan_sha256':'other'}]:
        with pytest.raises(ValueError):gate(status,'x',[],[])


def test_frozen_input_changes_are_rejected(tmp_path):
    import hashlib
    from sttw_control.deferred_training import verify_files
    p=tmp_path/'input';p.write_text('first')
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
    verify_files(files)
    p.write_text('changed')
    with pytest.raises(ValueError):verify_files(files)


def test_watcher_launches_once_after_two_ready_checks(tmp_path,monkeypatch):
    import json,time
    import sttw_control.deferred_training as module
    dependency=tmp_path/'dependency.json';dependency.write_text(json.dumps({'phase':'complete','plan_sha256':'x'}))
    out=tmp_path/'result';plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(created_unix=time.time(),max_wait_seconds=60,poll_seconds=0,input_sha256={},output=str(out),dependency_status=str(dependency),dependency_plan_sha256='x',dependency_repository='/unused',allowed_gpu_process_names=[],repository=str(tmp_path),task='task',training='train',panel='panel')))
    checks=[];launches=[]
    monkeypatch.setattr(module,'dependency_processes',lambda root:[])
    monkeypatch.setattr(module,'gpu_processes',lambda allowed: checks.append(1) or [])
    class Child:
        pid=123
        def wait(self):
            out.mkdir();(out/'pipeline_status.json').write_text('{"phase":"complete"}');return 0
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k: launches.append(a) or Child())
    module.watch(plan)
    assert len(checks)==2 and len(launches)==1
    assert json.loads((tmp_path/'status.json').read_text())['phase']=='complete'
    with pytest.raises(ValueError,match='already claimed'):module.watch(plan)
    assert len(launches)==1
