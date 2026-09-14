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
def test_command_queue_requires_success_and_does_not_wait_for_other_gpu_jobs(tmp_path,monkeypatch):
    import json,time
    import sttw_control.deferred_training as m
    dep=tmp_path/'dependency.json';dep.write_text('{"phase":"complete"}')
    launch=tmp_path/'dependency_launch.json';launch.write_text('{"pid":999999999}')
    out=tmp_path/'result';plan=tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(pipeline_kind='command',created_unix=time.time(),max_wait_seconds=60,poll_seconds=0,input_sha256={},output=str(out),dependency_status=str(dep),dependency_launch=str(launch),repository=str(tmp_path),task='task',training='train',panel='panel')))
    monkeypatch.setattr(m,'gpu_processes',lambda _:pytest.fail('must not query unrelated GPU jobs'))
    commands=[]
    class Child:
        pid=123
        def wait(self):
            out.mkdir();(out/'status.json').write_text('{"phase":"complete"}');return 0
    monkeypatch.setattr(m.subprocess,'Popen',lambda command,**kw:commands.append(command) or Child())
    m.watch(plan)
    assert len(commands)==1 and commands[0][1]=='learning/cli/command_experiment.py'
    assert json.loads((tmp_path/'launch.json').read_text())['pid']==123
    with pytest.raises(ValueError,match='already claimed'):m.watch(plan)


def test_command_queue_stops_on_failed_dependency(tmp_path,monkeypatch):
    import json,time
    import sttw_control.deferred_training as m
    dep=tmp_path/'dep.json';dep.write_text('{"phase":"error","error":"training failed"}')
    launch=tmp_path/'dep_launch.json';launch.write_text('{"pid":999999999}')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(pipeline_kind='command',created_unix=time.time(),max_wait_seconds=60,poll_seconds=0,input_sha256={},output=str(tmp_path/'result'),dependency_status=str(dep),dependency_launch=str(launch))))
    monkeypatch.setattr(m.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not launch'))
    with pytest.raises(ValueError,match='unsuccessfully'):m.watch(plan)
    assert json.loads((tmp_path/'status.json').read_text())['phase']=='error'


def test_command_queue_detects_missing_final_status(tmp_path):
    import json,time
    from sttw_control.deferred_training import watch
    launch=tmp_path/'dep_launch.json';launch.write_text('{"pid":999999999}')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(pipeline_kind='command',created_unix=time.time(),max_wait_seconds=60,poll_seconds=0,input_sha256={},output=str(tmp_path/'result'),dependency_status=str(tmp_path/'missing'),dependency_launch=str(launch))))
    with pytest.raises(RuntimeError,match='without readable status'):watch(plan)


def test_command_queue_starts_stage_and_completion_monitor(tmp_path,monkeypatch):
    import json,time
    import sttw_control.deferred_training as m
    dep=tmp_path/'dep.json';dep.write_text('{"phase":"complete"}')
    launch=tmp_path/'dep_launch.json';launch.write_text('{"pid":999999999}')
    out=tmp_path/'result';plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(pipeline_kind='command',notify=True,created_unix=time.time(),max_wait_seconds=60,poll_seconds=0,input_sha256={},output=str(out),dependency_status=str(dep),dependency_launch=str(launch),repository=str(tmp_path),task='task',training='train',panel='panel')))
    commands=[]
    class Child:
        pid=123
        def wait(self):
            out.mkdir();(out/'status.json').write_text('{"phase":"complete"}');return 0
    monkeypatch.setattr(m.subprocess,'Popen',lambda cmd,**kw:commands.append(cmd) or Child())
    m.watch(plan)
    assert len(commands)==2
    assert commands[1][1]=='learning/cli/watch_run.py'
    assert str(out/'status.json') in commands[1]
    assert (tmp_path/'monitor_launch.json').exists()
