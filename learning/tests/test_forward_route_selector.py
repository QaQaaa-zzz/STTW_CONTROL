from sttw_control.jump_connection import ForwardRouteSelector


def selector(records):
    routes=[dict(route_id=r,goal_id='goal',task_id=r,skill_id='frozen',candidates=['u'],reference_path=r+'.npz') for r in ('original_jump','bypass')]
    config=dict(map_id='map',feature_scales=[1.],bandwidth=1.,minimum_support=.5,
        minimum_completion=.7,probability_tolerance=.05,time_tolerance_s=.02,
        program=dict(goals=[dict(goal_id='goal',allowed_routes=['original_jump','bypass'],required_skills=['jump'])]),
        routes=routes,candidates=[dict(candidate_id='u',speed_m_s=2.,alpha=.5)])
    return ForwardRouteSelector(records,config,origin_id='test_root',policy_sha256={'ground':'h','jump':'j'},model_contract_sha256='m',physics_dt=.0025,zero_lower=True)


def record(route,origin='train_root',map_id='map'):
    return dict(origin_id=origin,target_semantics='Q_route',supervised_input_complete=True,
        policy_sha256={'ground':'h','jump':'j'},model_contract_sha256='m',physics_dt=.0025,zero_lower=True,map_id=map_id,
        state=[0.],task_id=route,skill_id='frozen',candidate_id='u',complete=1,success_cost=10. if route=='bypass' else 12.)


def test_route_query_excludes_same_origin_and_map_and_retains_unknown():
    s=selector([record('bypass','test_root'),record('original_jump',map_id='different')])
    result=s.choose([0.],['original_jump','bypass'])
    assert result['status']=='revision_required' and result['reason']=='unknown'
    assert result['evidence_rows']==0 and s.task.active is None


def test_unavailable_bypass_cannot_be_selected_and_jump_goal_remains_pending():
    s=selector([record('original_jump'),record('bypass')])
    result=s.choose([0.],['original_jump'])
    assert result['reference']['route_id']=='original_jump'
    result=s.choose([0.],['original_jump','bypass'])
    assert result['reference']['route_id']=='bypass'
    assert not s.task.observe_goal('goal',arrived=True,upright=True,within_allowed_region=True,completed_skills=[])
    assert s.task.completed==[]


def test_cpu_route_query_uses_copied_state_and_shared_rule(monkeypatch):
    from sttw_control import jump_connection
    monkeypatch.setattr(jump_connection, 'decision_features', lambda *args: ([.12], {'roll': .12}))
    requested, choice, named = jump_connection.plan_forward_route(
        {'time': 1.}, None, {}, {}, {'mode': 'rule', 'absolute_limits': {'roll': .1}}, None, ['original_jump', 'bypass'])
    assert requested == 'bypass' and choice['violated'] == ['roll']
    assert named == {'roll': .12}


def test_pending_cpu_query_does_not_block_executor(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from sttw_control import jump_connection
    release = Event()
    entered = Event()
    def features(*args):
        entered.set()
        release.wait(2.)
        return [0.], {'roll': 0.}
    monkeypatch.setattr(jump_connection, 'decision_features', features)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(jump_connection.plan_forward_route, {}, None, {}, {},
                                 {'mode': 'rule', 'absolute_limits': {'roll': .1}}, None, ['original_jump'])
        assert entered.wait(1.)
        assert not future.done()
        # Executor can continue applying its existing reference before query completion.
        release.set()
        assert future.result(timeout=1.)[0] == 'original_jump'
