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
