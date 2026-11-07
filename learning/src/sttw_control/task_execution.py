"""Confirm recorded task execution using existing physical adjudication."""
from copy import deepcopy
from .continuous_task import ContinuousTask


def confirm_task_execution(snapshot, report, *, require_platform=False):
    """Post-run confirmation, never a replacement for online control or labels."""
    program=deepcopy(snapshot['program']);goal_id=snapshot['outcome_goal_id']
    goal=next(g for g in program['goals'] if g['goal_id']==goal_id)
    if require_platform:
        goal['required_skills']=sorted(set(goal.get('required_skills',()))|{'platform_jump'})
    manager=ContinuousTask(program,snapshot['routes'],snapshot['candidates'],None)
    for event in snapshot['events']:
        if event['status']=='reference_executed':
            ref=event['reference']
            manager.record_execution(ref['route_id'],ref['candidate_id'],time_s=event['time_s'],reason=event['reason'])
        else:
            manager.events.append(event)
            if event['status']=='reference_selected':manager.active=event['reference'].copy()
    forward=report['forward_task'];checks=report['checks'];skills=[]
    if forward['platform_skill_complete']:skills.append('platform_jump')
    if checks['ground_return']:skills.append('ground_return')
    if checks['continued_time'] and checks['continued_forward']:skills.append('continued_drive')
    arrived=forward['full_window'] and forward['common_prefix_passed'] and forward['first_goal_arrival_s'] is not None
    accepted=manager.observe_goal(goal_id,arrived=arrived,upright=checks['full_window'],within_allowed_region=forward['mapped_corridor_complete'],completed_skills=skills)
    remaining=[g for g in program['goals'] if g['goal_id'] not in manager.completed]
    revision=dict(reason='physical_execution_incomplete' if not accepted else 'remaining_goals_pending',remaining_goals=remaining,completed=list(manager.completed),active_reference=manager.active_reference,missing_required_skills=manager.events[-1]['missing_required_skills']) if remaining else None
    disposition='still_required' if remaining else ('resolved_by_observed_completion' if any(e['status']=='revision_required' for e in snapshot['events']) else 'not_required')
    return dict(request_disposition=disposition,role='Post-adjudication task progress from actual execution and frozen physical labels; no new planning or simulation',semantic_variant='mandatory_platform' if require_platform else 'declared_program',status='complete' if not remaining else 'revision_required',accepted_goal=accepted,completed=list(manager.completed),remaining_goals=remaining,active_reference=manager.active_reference,last_proposal=manager.active,events=manager.events,revision_request=revision,physical_evidence=dict(full_window=forward['full_window'],common_prefix_passed=forward['common_prefix_passed'],goal_arrival_s=forward['first_goal_arrival_s'],within_allowed_region=forward['mapped_corridor_complete'],completed_skills=skills))
