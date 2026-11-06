from sttw_control.continuous_task import ContinuousTask
from sttw_control.empirical_capability import ConnectionCapability


def manager():
    candidates=[dict(candidate_id='slow',speed_m_s=1.5,alpha=0),dict(candidate_id='fast',speed_m_s=2.3,alpha=1)]
    records=[dict(state=[0.],task_id='left',skill_id='frozen',candidate_id=c['candidate_id'],complete=1,success_cost=8 if c['candidate_id']=='slow' else 6) for c in candidates]
    routes=[dict(route_id='left_route',goal_id='checkpoint',task_id='left',skill_id='frozen',candidates=['slow','fast'],reference_path='left.npz')]
    return ContinuousTask(dict(goals=[dict(goal_id='checkpoint',allowed_routes=['left_route'])]),routes,candidates,ConnectionCapability(records,[1.],1.,.5))


def test_recommendation_is_not_completion_and_requires_physical_evidence():
    m=manager();result=m.choose([0.],['left_route'])
    assert result['reference']['candidate_id']=='fast'
    assert m.completed==[]
    assert not m.observe_goal('checkpoint',arrived=True,upright=False,within_allowed_region=True)
    assert m.observe_goal('checkpoint',arrived=True,upright=True,within_allowed_region=True)
    assert m.choose([0.],['left_route'])['status']=='complete'


def test_unsupported_state_requests_revision_and_keeps_active_reference():
    m=manager();reference=m.choose([0.],['left_route'])['reference']
    event=m.choose([100.],['left_route'])
    assert event['status']=='revision_required'
    assert event['active_reference']==reference
    assert event['remaining_goals']==['checkpoint']
    assert m.completed==[]


def test_blocked_route_is_not_recommended_from_old_success():
    m=manager();event=m.choose([0.],[])
    assert event['status']=='revision_required'
    assert event['queries']==[]
    assert m.active is None


def test_remaining_task_revision_preserves_progress_and_defers_reference_switch():
    m=manager()
    for goal in ('second','third'):
        route=dict(m.routes['left_route'],route_id=goal+'_route',goal_id=goal)
        m.routes[route['route_id']]=route
        m.program['goals'].append(dict(goal_id=goal,allowed_routes=[route['route_id']]))
    m.choose([0.],['left_route'])
    m.observe_goal('checkpoint',arrived=True,upright=True,within_allowed_region=True)
    old=m.choose([0.],['second_route'])['reference']
    revision=m.revise_remaining(list(reversed(m.program['goals'][1:])),reason='remaining route order changed after disturbance')
    assert revision['completed']==['checkpoint']
    assert revision['remaining_goals']==['third','second']
    assert m.active==old
    new=m.choose([0.],['third_route'])['reference']
    assert new['goal_id']=='third'
    assert m.completed==['checkpoint']


def test_revision_cannot_silently_drop_unfinished_task():
    import pytest
    m=manager()
    with pytest.raises(ValueError):m.revise_remaining([],reason='unsupported route')
    assert m.program['goals'][0]['goal_id']=='checkpoint'
