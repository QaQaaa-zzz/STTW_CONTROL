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


def test_revision_reasons_distinguish_support_and_geometry():
    m=manager()
    assert m.choose([100.],['left_route'])['reason']=='unknown'
    assert m.choose([0.],[])['reason']=='no_available_route'
    for row in m.capability.records:
        row['complete']=0;row['success_cost']=None
    assert m.choose([0.],['left_route'])['reason']=='no_feasible'


def test_near_tie_retains_active_reference_without_claiming_goal_completion():
    m=manager();first=m.choose([0.],['left_route'])['reference']
    for row in m.capability.records:
        row['success_cost']=5. if row['candidate_id']=='slow' else 5.01
    second=m.choose([0.],['left_route'])
    assert second['reference']==first and second['held_previous']
    assert m.completed==[]


def test_missed_branch_and_dwell_share_one_geometry_filter():
    from sttw_control.continuous_task import available_forward_routes
    routes=[dict(route_id='main'),dict(route_id='bypass',first_entry_progress_m=1.,last_entry_progress_m=3.,minimum_dwell_s=.5)]
    assert available_forward_routes(routes,['main','bypass'],furthest_progress_m=2.,time_s=1.,current_route='main',last_switch_time_s=.8)['available_routes']==['main']
    assert available_forward_routes(routes,['main','bypass'],furthest_progress_m=2.,time_s=1.4,current_route='main',last_switch_time_s=.8)['available_routes']==['main','bypass']
    missed=available_forward_routes(routes,['main','bypass'],furthest_progress_m=3.1,time_s=2.,current_route='main',last_switch_time_s=.8)
    assert missed['available_routes']==['main']
    assert missed['blocked_routes']==[dict(route_id='bypass',reason='missed_diversion')]
    # Once already executing a branch, passing its entry deadline does not revoke it.
    assert 'bypass' in available_forward_routes(routes,['main','bypass'],furthest_progress_m=4.,time_s=3.,current_route='bypass')['available_routes']


def test_mandatory_platform_skill_cannot_be_replaced_by_goal_arrival():
    m=manager();m.program['goals'][0]['required_skills']=['platform_jump']
    m.choose([0.],['left_route'])
    assert not m.observe_goal('checkpoint',arrived=True,upright=True,within_allowed_region=True)
    assert m.completed==[] and m.events[-1]['missing_required_skills']==['platform_jump']
    assert m.observe_goal('checkpoint',arrived=True,upright=True,within_allowed_region=True,completed_skills=['platform_jump'])


def test_revision_cannot_drop_required_skill_while_preserving_goal_id():
    import pytest
    m=manager();m.program['goals'][0]['required_skills']=['platform_jump']
    m.choose([0.],['left_route']);active=m.active.copy()
    with pytest.raises(ValueError,match='required skills'):
        m.revise_remaining([dict(goal_id='checkpoint',allowed_routes=['left_route'])],reason='route unavailable')
    assert m.program['goals'][0]['required_skills']==['platform_jump']
    assert m.active==active and not m.completed
    m.revise_remaining([dict(goal_id='checkpoint',allowed_routes=['left_route'],required_skills=['platform_jump'])],reason='new approach')
    assert m.program['goals'][0]['required_skills']==['platform_jump']


def test_recorded_execution_survives_an_unapplied_proposal_and_unknown_query():
    m=manager()
    m.record_execution('left_route','slow',time_s=0.,reason='already executing')
    proposal=m.choose([0.],['left_route'])
    assert proposal['reference']['candidate_id']=='fast'
    assert m.active_reference['candidate_id']=='slow'
    unknown=m.choose([100.],['left_route'])
    assert unknown['active_reference']['candidate_id']=='slow'
    m.record_execution('left_route','fast',time_s=1.,reason='executor applied proposal')
    assert m.active_reference['candidate_id']=='fast' and not m.completed


def test_next_goal_needs_actual_execution_binding_after_progress_advances():
    m=manager()
    m.routes['second_route']=dict(m.routes['left_route'],route_id='second_route',goal_id='second')
    m.program['goals'].append(dict(goal_id='second',allowed_routes=['second_route']))
    m.record_execution('left_route','fast',time_s=0.,reason='initial execution')
    assert m.observe_goal('checkpoint',arrived=True,upright=True,within_allowed_region=True)
    m.choose([0.],['second_route'])
    assert not m.observe_goal('second',arrived=True,upright=True,within_allowed_region=True)
    m.record_execution('second_route','fast',time_s=2.,reason='next reference applied')
    assert m.observe_goal('second',arrived=True,upright=True,within_allowed_region=True)
    assert m.completed==['checkpoint','second']



def test_completed_goal_does_not_erase_still_executing_reference_or_repeat_credit():
    m=manager()
    m.record_execution('left_route','fast',time_s=0.,reason='executing')
    assert m.observe_goal('checkpoint',arrived=True,upright=True,within_allowed_region=True)
    assert m.active_reference['route_id']=='left_route'
    m.record_execution('left_route','fast',time_s=1.,reason='still executing while awaiting next plan')
    assert not m.observe_goal('checkpoint',arrived=True,upright=True,within_allowed_region=True)
    assert m.completed==['checkpoint']
