import importlib.util,json
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'cli/teleop_governor_status.py'

def module():
    spec=importlib.util.spec_from_file_location('teleop_status',PATH)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_finished_failure_is_not_success(tmp_path):
    m=module();(tmp_path/'done.json').write_text(json.dumps({'passed':False}))
    (tmp_path/'checks_watch.json').write_text(json.dumps({'checks':[{'name':'contract','result':'done.json','pass_field':'passed','pid':None}]}))
    x=m.inspect(tmp_path)
    assert x['all_finished'] and x['status']=='finished_with_failed_check'
    assert not x['all_checks_passed']

def test_dead_job_missing_result_not_done(tmp_path):
    m=module();(tmp_path/'checks_watch.json').write_text(json.dumps({'checks':[{'name':'contract','result':'missing.json','pid':None}]}))
    x=m.inspect(tmp_path)
    assert not x['all_finished'] and x['status']=='missing_or_interrupted'

def test_malformed_output_not_success(tmp_path):
    m=module();(tmp_path/'done.json').write_text('{')
    (tmp_path/'checks_watch.json').write_text(json.dumps({'checks':[{'name':'contract','result':'done.json','pid':None}]}))
    assert module().inspect(tmp_path)['status']=='missing_or_interrupted'

def test_notification_text_distinguishes_completion_from_pass():
    m=module()
    title,body=m.notification_text({'status':'finished_with_failed_check','all_finished':True,'checks':[]})
    assert title=='偏好控制器仿真结束'
    assert '未通过' in body and '尚未运行' in body

def test_popup_uses_current_failure_names_instead_of_old_gate():
    title,body=module().notification_text({'status':'finished_with_failed_check','all_finished':True,'checks':[{'name':'Gate A baseline','passed':False}]})
    assert 'Gate A baseline' in body and '144 个重复候选' not in body
