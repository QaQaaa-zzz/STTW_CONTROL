import json
from sttw_control.run_watchdog import inspect_run, process_identity


def test_completion_wins_over_exited_process(tmp_path):
    p=tmp_path/'status.json';p.write_text('{"phase":"complete"}')
    assert inspect_run(p,False)==('complete',None)


def test_program_error_and_exit_are_distinct_from_physical_failure(tmp_path):
    p=tmp_path/'status.json';p.write_text('{"phase":"training","failed":[true]}')
    assert inspect_run(p,True)==('running',None)
    assert inspect_run(p,False)[0]=='error'
    p.write_text('{"phase":"error","error":"Out of memory"}')
    assert 'Out of memory' in inspect_run(p,True)[1]


def test_partial_status_write_is_retried(tmp_path):
    p=tmp_path/'status.json';p.write_text('{')
    assert inspect_run(p,True)[0]=='retry'
    assert inspect_run(p,False)[0]=='error'


def test_process_identity_is_real_and_detects_missing_pid():
    import os
    assert process_identity(os.getpid()) is not None
    assert process_identity(999999999) is None


def test_completed_pipeline_notifies_once_on_resume(tmp_path,monkeypatch):
    from sttw_control import run_watchdog as w
    status=tmp_path/'run.json';status.write_text('{"phase":"complete"}')
    launch=tmp_path/'launch.json';launch.write_text('{"pid":999999999}')
    out=tmp_path/'monitor';calls=[]
    monkeypatch.setattr(w,'popup',lambda title,message,**kw:calls.append((title,kw)) or 'test')
    argv=['watch','--status',str(status),'--launch',str(launch),'--output',str(out)]
    monkeypatch.setattr('sys.argv',argv);w.main()
    assert len(calls)==1 and calls[0][1]['error'] is False
    assert json.loads((out/'status.json').read_text())['completion_notified']
    monkeypatch.setattr('sys.argv',argv+['--resume']);w.main()
    assert len(calls)==1


def test_training_completion_notifies_before_pipeline_and_only_once(tmp_path,monkeypatch):
    from sttw_control import run_watchdog as w
    stage=tmp_path/'training/status.json';stage.parent.mkdir();stage.write_text('{"complete":true}')
    calls=[];notified={}
    monkeypatch.setattr(w,'popup',lambda title,message,**kw:calls.append((title,message,kw)) or 'test')
    w.notify_completed_stages([stage],notified)
    w.notify_completed_stages([stage],notified)
    assert len(calls)==1 and calls[0][2]['error'] is False
    assert '训练阶段' in calls[0][1] and str(stage.resolve()) in notified
    restored=json.loads(json.dumps(notified))
    w.notify_completed_stages([stage],restored)
    assert len(calls)==1


def test_incomplete_or_unreadable_stage_does_not_notify(tmp_path,monkeypatch):
    from sttw_control import run_watchdog as w
    p=tmp_path/'status.json';calls=[]
    monkeypatch.setattr(w,'popup',lambda *a,**kw:calls.append(a))
    for text in ('{','{"complete":false}','{"phase":"error"}'):
        p.write_text(text);w.notify_completed_stages([p],{})
    assert not calls
