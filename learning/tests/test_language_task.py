import pytest
from sttw_control.language_task import parse_task_request, public_catalog


def catalog():
    return dict(scenes=[dict(scene_id='left_rolling',description='左弯起伏',skills=['heading125'],completion_protocol='contact_filtered_20ms',prior_observation='one DEV completion',config='/private/local/config.json')])


def response(name, args):
    return dict(stop_reason='tool_use',content=[dict(type='tool_use',id='call1',name=name,input=args)])


def test_model_cannot_override_physics_or_assert_completion():
    for args in [dict(scene_id='left_rolling',complete=True),dict(scene_id='left_rolling',seconds=1),dict(scene_id='../../elsewhere')]:
        with pytest.raises(ValueError):parse_task_request(response('execute_scene',args),catalog())
    with pytest.raises(ValueError):parse_task_request(dict(stop_reason='end_turn',content=[dict(type='text',text='已完成')]),catalog())


def test_single_registered_request_and_unsupported_task():
    assert parse_task_request(response('execute_scene',dict(scene_id='left_rolling')),catalog())['input']['scene_id']=='left_rolling'
    assert parse_task_request(response('request_task_revision',dict(reason='目录没有楼梯')),catalog())['name']=='request_task_revision'
    r=response('execute_scene',dict(scene_id='left_rolling'));r['content']*=2
    with pytest.raises(ValueError):parse_task_request(r,catalog())


def test_public_scene_context_excludes_local_execution_paths():
    assert 'config' not in public_catalog(catalog())[0]


def test_chat_provider_roundtrip_preserves_tool_result_and_private_credentials(tmp_path,monkeypatch):
    import io,json
    import sttw_control.language_task as module
    settings=tmp_path/'private.json';settings.write_text(json.dumps(dict(protocol='openai_chat_completions',model='fixture',thinking={'type':'enabled'},reasoning_effort='low',base_url='https://example.invalid/v1',api_key='secret-fixture')))
    output=tmp_path/'out';output.mkdir();requests=[]
    native=dict(role='assistant',content=None,reasoning_content='provider context',tool_calls=[dict(id='c1',type='function',function=dict(name='execute_scene',arguments='{"scene_id":"left_rolling"}'))])
    def serve(request,timeout):
        requests.append(json.loads(request.data))
        message=native if len(requests)==1 else dict(role='assistant',content='未完成')
        return io.BytesIO(json.dumps(dict(model='fixture',choices=[dict(message=message,finish_reason='tool_calls' if len(requests)==1 else 'stop')])).encode())
    monkeypatch.setattr(module,'urlopen',serve)
    messages=[dict(role='user',content='左弯')];tools=module.task_tools(catalog())
    first=module.call_existing_model(messages,tools,settings,output,1);assert module.parse_task_request(first,catalog())['id']=='c1'
    messages.extend([dict(role='assistant',content=first['content'],provider_message=first['provider_message']),dict(role='user',content=[dict(type='tool_result',tool_use_id='c1',content='{"complete":false}')])])
    module.call_existing_model(messages,tools,settings,output,2,allow_tools=False)
    assert requests[1]['messages'][-2]==native
    assert requests[1]['messages'][-1]['tool_call_id']=='c1'
    assert requests[1]['tool_choice']=='none'
    assert all('secret-fixture' not in p.read_text() for p in output.iterdir())
