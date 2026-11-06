"""Task progress and numerical connection choice for known-map task programs.

Language models propose goal order and allowed route IDs. Actual capability data
select speed/alpha. Recommendations never constitute physical task completion.
"""

from .empirical_capability import select_connection_estimate, connection_selection_status

def available_forward_routes(routes, geometric_routes, *, furthest_progress_m, time_s,
                             current_route=None, last_switch_time_s=None):
    """Shared map-entry filter for rule and capability planners.

    The executor supplies geometric reachability and monotone progress on the
    common approach. A missed branch cannot reopen after reversing or replanning.
    Dwell uses the actual executed switch time, not recommendation timestamps.
    """
    available=[];blocked=[]
    for route in routes:
        rid=route['route_id'];reason=None
        if rid not in geometric_routes:reason='geometrically_unavailable'
        elif rid!=current_route:
            if furthest_progress_m>route.get('last_entry_progress_m',float('inf')):reason='missed_diversion'
            elif furthest_progress_m<route.get('first_entry_progress_m',float('-inf')):reason='before_entry'
            elif last_switch_time_s is not None and time_s-last_switch_time_s<route.get('minimum_dwell_s',0.):reason='minimum_dwell'
        if reason is None:available.append(rid)
        else:blocked.append(dict(route_id=rid,reason=reason))
    return dict(available_routes=available,blocked_routes=blocked)


class ContinuousTask:
    def __init__(self, program, routes, candidates, capability):
        self.program=program
        self.routes={r['route_id']:r for r in routes}
        self.candidates={c['candidate_id']:c for c in candidates}
        self.capability=capability
        goals=program['goals']
        if len({g['goal_id'] for g in goals})!=len(goals) or any(
            not g['allowed_routes'] or any(r not in self.routes or self.routes[r]['goal_id']!=g['goal_id'] for r in g['allowed_routes']) for g in goals):
            raise ValueError('Unique goals and known routes terminating at each requested goal required')
        self.completed=[]
        self.active=None
        self.events=[]

    def choose(self, state, available_routes, minimum_completion=.7, probability_tolerance=.05, time_tolerance=.02):
        """Select only supported feasible routes, then efficiency among similar p.

        The caller supplies current geometric/skill availability. An unknown
        capability requests task revision; it does not invent a safe fallback.
        An existing active reference remains the executor's until superseded.
        """
        remaining=[g for g in self.program['goals'] if g['goal_id'] not in self.completed]
        if not remaining:return dict(status='complete',completed=list(self.completed))
        goal=remaining[0];options=[];queries=[]
        for route_id in goal['allowed_routes']:
            if route_id not in available_routes:continue
            route=self.routes[route_id]
            query=self.capability.query(state,route['task_id'],route['skill_id'],route['candidates'])
            queries.append(dict(route_id=route_id,**query))
            for estimate in query['estimates']:
                if estimate['known'] and estimate['complete_probability']>=minimum_completion and estimate['success_cost'] is not None:
                    options.append(dict(route_id=route_id,**estimate))
        previous=self.active if self.active and self.active['goal_id']==goal['goal_id'] else None
        selected=select_connection_estimate(options,minimum_completion,probability_tolerance,
            previous_candidate=previous['candidate_id'] if previous else None,
            previous_route=previous['route_id'] if previous else None,time_tolerance=time_tolerance)
        if selected is None:
            reason=connection_selection_status([e for q in queries for e in q['estimates']],None) if queries else 'no_available_route'
            event=dict(status='revision_required',reason=reason,goal_id=goal['goal_id'],completed=list(self.completed),remaining_goals=[g['goal_id'] for g in remaining],queries=queries,active_reference=self.active)
            self.events.append(event)
            return event
        route=self.routes[selected['route_id']];candidate=self.candidates[selected['candidate_id']]
        self.active=dict(goal_id=goal['goal_id'],route_id=route['route_id'],reference_path=route['reference_path'],skill_id=route['skill_id'],candidate_id=candidate['candidate_id'],speed_m_s=candidate['speed_m_s'],alpha=candidate['alpha'])
        event=dict(status='reference_selected',reference=self.active.copy(),estimate=selected,queries=queries,held_previous=bool(previous and previous==self.active),time_tolerance_s=time_tolerance)
        self.events.append(event)
        return event

    def revise_remaining(self, goals, reason):
        """Apply a task-level reorder/reroute without inventing completed goals.

        A planner may change remaining goal order and mapped route choices.
        Physical reference replacement still requires choose() with current state.
        """
        remaining={g['goal_id'] for g in self.program['goals'] if g['goal_id'] not in self.completed}
        ids=[g['goal_id'] for g in goals]
        if len(ids)!=len(set(ids)) or set(ids)!=remaining or any(
            not g['allowed_routes'] or any(r not in self.routes or self.routes[r]['goal_id']!=g['goal_id'] for r in g['allowed_routes']) for g in goals):
            raise ValueError('Revision must preserve outstanding goals and use mapped routes to each goal')
        done=[g for g in self.program['goals'] if g['goal_id'] in self.completed]
        self.program={**self.program,'goals':done+goals}
        event=dict(status='task_revised',reason=reason,completed=list(self.completed),
                   remaining_goals=ids,active_reference=self.active)
        self.events.append(event)
        return event

    def observe_goal(self, goal_id, *, arrived, upright, within_allowed_region, completed_skills=()):
        """Executor supplies physical evidence; a plan or LLM cannot acknowledge it."""
        goal=next(g for g in self.program['goals'] if g['goal_id']==goal_id)
        missing_skills=sorted(set(goal.get('required_skills',()))-set(completed_skills))
        accepted=bool(not missing_skills and self.active and goal_id==self.active['goal_id'] and arrived and upright and within_allowed_region)
        if accepted:
            self.completed.append(goal_id);self.active=None
        self.events.append(dict(status='goal_observed',goal_id=goal_id,accepted=accepted,arrived=bool(arrived),upright=bool(upright),within_allowed_region=bool(within_allowed_region),missing_required_skills=missing_skills,completed_skills=list(completed_skills)))
        return accepted
