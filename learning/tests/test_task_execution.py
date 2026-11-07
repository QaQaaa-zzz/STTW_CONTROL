from sttw_control.task_execution import confirm_task_execution


def test_destination_only_bypass_cannot_satisfy_mandatory_platform_task():
    snapshot=dict(program=dict(goals=[dict(goal_id='destination',allowed_routes=['bypass'])]),routes=[dict(route_id='bypass',goal_id='destination',task_id='bypass',skill_id='ground',candidates=['cruise'],reference_path='bypass.npz')],candidates=[dict(candidate_id='cruise',speed_m_s=2.,alpha=.5)],outcome_goal_id='destination',events=[dict(status='reference_executed',reference=dict(route_id='bypass',candidate_id='cruise'),time_s=1.,reason='actual bypass applied')])
    report=dict(forward_task=dict(platform_skill_complete=False,full_window=True,common_prefix_passed=True,first_goal_arrival_s=10.,mapped_corridor_complete=True),checks=dict(full_window=True,ground_return=False,continued_time=False,continued_forward=False))
    optional=confirm_task_execution(snapshot,report)
    mandatory=confirm_task_execution(snapshot,report,require_platform=True)
    assert optional['status']=='complete'
    assert mandatory['status']=='revision_required'
    assert mandatory['completed']==[]
    assert mandatory['revision_request']['missing_required_skills']==['platform_jump']
    assert snapshot['program']['goals'][0].get('required_skills') is None


def test_observed_completion_resolves_old_revision_without_erasing_actual_reference():
    snapshot=dict(program=dict(goals=[dict(goal_id='destination',allowed_routes=['main'])]),routes=[dict(route_id='main',goal_id='destination',task_id='main',skill_id='frozen',candidates=['cruise'],reference_path='main.npz')],candidates=[dict(candidate_id='cruise',speed_m_s=2.,alpha=.5)],outcome_goal_id='destination',events=[dict(status='reference_executed',reference=dict(route_id='main',candidate_id='cruise'),time_s=0.,reason='existing reference'),dict(status='revision_required',reason='no_feasible')])
    report=dict(forward_task=dict(platform_skill_complete=True,full_window=True,common_prefix_passed=True,first_goal_arrival_s=12.,mapped_corridor_complete=True),checks=dict(full_window=True,ground_return=True,continued_time=True,continued_forward=True))
    result=confirm_task_execution(snapshot,report,require_platform=True)
    assert result['completed']==['destination'] and result['revision_request'] is None
    assert result['request_disposition']=='resolved_by_observed_completion'
    assert result['active_reference']['route_id']=='main'
