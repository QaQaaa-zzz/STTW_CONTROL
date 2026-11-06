"""Task progress and numerical connection choice for known-map task programs.

Language models propose goal order and allowed route IDs. Actual capability data
select speed/alpha. Recommendations never constitute physical task completion.
"""

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

    def choose(self, state, available_routes, minimum_completion=.7, probability_tolerance=.05):
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
        if not options:
            event=dict(status='revision_required',goal_id=goal['goal_id'],completed=list(self.completed),remaining_goals=[g['goal_id'] for g in remaining],queries=queries,active_reference=self.active)
            self.events.append(event)
            return event
        best_probability=max(o['complete_probability'] for o in options)
        selected=min((o for o in options if o['complete_probability']>=best_probability-probability_tolerance),key=lambda o:o['success_cost'])
        route=self.routes[selected['route_id']];candidate=self.candidates[selected['candidate_id']]
        self.active=dict(goal_id=goal['goal_id'],route_id=route['route_id'],reference_path=route['reference_path'],skill_id=route['skill_id'],candidate_id=candidate['candidate_id'],speed_m_s=candidate['speed_m_s'],alpha=candidate['alpha'])
        event=dict(status='reference_selected',reference=self.active.copy(),estimate=selected,queries=queries)
        self.events.append(event)
        return event

    def observe_goal(self, goal_id, *, arrived, upright, within_allowed_region):
        """Executor supplies physical evidence; a plan or LLM cannot acknowledge it."""
        accepted=bool(self.active and goal_id==self.active['goal_id'] and arrived and upright and within_allowed_region)
        if accepted:
            self.completed.append(goal_id);self.active=None
        self.events.append(dict(status='goal_observed',goal_id=goal_id,accepted=accepted,arrived=bool(arrived),upright=bool(upright),within_allowed_region=bool(within_allowed_region)))
        return accepted
