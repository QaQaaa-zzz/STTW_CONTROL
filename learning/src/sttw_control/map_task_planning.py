"""Known-map task plans and current-state capability checks; no scene selection."""
from copy import deepcopy
import math
import time


class MapTaskPlanner:
    def __init__(self, graph, task):
        self.graph=graph;self.task=deepcopy(task)
        self.regions={r['region_id']:r for r in graph['regions']}
        self.edges={e['connection_id']:e for e in graph['connections']}
        self.skills={s['skill_id']:s for s in graph['skills']}
        if len(self.edges)!=len(graph['connections']) or len(self.regions)!=len(graph['regions']) or len(self.skills)!=len(graph['skills']):
            raise ValueError('Map identifiers must be unique')
        if task['start_region'] not in self.regions or task['goal_region'] not in self.regions:
            raise ValueError('Task endpoints must belong to this map')
        if set(task['hard_constraints'])!={'forbidden_connections','ordered_required_regions','required_skill_families','maximum_remaining_time_s','completion_protocol_id'}:
            raise ValueError('Unsupported hard constraint; every accepted constraint must be checked')
        budget=task['hard_constraints']['maximum_remaining_time_s']
        if budget is not None and (not math.isfinite(budget) or budget<=0):
            raise ValueError('Time budget must be positive finite seconds or null')
        for e in self.edges.values():
            if e['from_region'] not in self.regions or e['to_region'] not in self.regions or not e['skills'] or any(s not in self.skills for s in e['skills']):
                raise ValueError('Every connection needs mapped endpoints and registered skills')
            if not math.isfinite(e['length_m']) or e['length_m']<=0 or not math.isfinite(e['half_width_m']) or e['half_width_m']<=0:
                raise ValueError('Connection geometry must have positive length and road width')

    def validate(self, plan):
        """Strict task schema; model cannot inject execution parameters or relax rules."""
        if set(plan)!={'map_id','goal_region','steps','hard_constraints'}:
            raise ValueError('Plan must contain only map, goal, steps and hard constraints')
        if plan['map_id']!=self.graph['map_id'] or plan['goal_region']!=self.task['goal_region'] or plan['hard_constraints']!=self.task['hard_constraints']:
            raise ValueError('Map, goal and hard constraints are immutable')
        region=self.task['start_region'];visited=[region];families=[]
        for step in plan['steps']:
            if set(step)!={'target_region','candidates'} or not step['candidates']:
                raise ValueError('Each step needs a target and candidate connections/skills')
            step_families=[]
            for candidate in step['candidates']:
                if set(candidate)!={'connection_id','skill_id'}:
                    raise ValueError('Task plans cannot specify speed, alpha or motor actions')
                if candidate['connection_id'] not in self.edges or candidate['skill_id'] not in self.skills:
                    raise ValueError('Unknown connection or skill')
                edge=self.edges[candidate['connection_id']];skill=candidate['skill_id']
                if edge['from_region']!=region or edge['to_region']!=step['target_region'] or skill not in edge['skills']:
                    raise ValueError('Disconnected plan or incompatible skill')
                if edge['connection_id'] in self.task['hard_constraints']['forbidden_connections']:
                    raise ValueError('Forbidden connection')
                step_families.append(self.skills[skill]['family'])
            # A required skill must occur in every selectable execution of the plan.
            families.extend(set(step_families) if len(set(step_families))==1 else [])
            region=step['target_region'];visited.append(region)
        if region!=self.task['goal_region'] or not plan['steps']:
            raise ValueError('Plan does not reach the required goal')
        cursor=0
        for target in self.task['hard_constraints']['ordered_required_regions']:
            if target not in visited[cursor:]:raise ValueError('Required region order is missing')
            cursor=visited.index(target,cursor)+1
        if not set(self.task['hard_constraints']['required_skill_families']).issubset(families):
            raise ValueError('Required skill family missing; bypass cannot satisfy a mandatory jump')
        return plan

    def enumerate_plans(self, max_plans=16):
        """Deterministic graph baseline, independent of LLM and physical outcomes."""
        plans=[]
        def visit(region, seen, steps):
            if len(plans)>=max_plans:return
            if region==self.task['goal_region']:
                plan=dict(map_id=self.graph['map_id'],goal_region=region,steps=steps,hard_constraints=deepcopy(self.task['hard_constraints']))
                try:self.validate(plan)
                except ValueError:return
                plans.append(plan);return
            for edge in self.edges.values():
                if edge['from_region']==region and edge['to_region'] not in seen and edge['connection_id'] not in self.task['hard_constraints']['forbidden_connections']:
                    for skill in edge['skills']:
                        visit(edge['to_region'],seen|{edge['to_region']},steps+[dict(target_region=edge['to_region'],candidates=[dict(connection_id=edge['connection_id'],skill_id=skill)])])
        visit(self.task['start_region'],{self.task['start_region']},[])
        return plans

    def verify_current(self, plan, observation, capability):
        """Query the current segment only; future segments need future actual states."""
        started=time.perf_counter();self.validate(plan);i=observation['step_index']
        if not isinstance(i,int) or not 0<=i<len(plan['steps']):raise ValueError('Current step index outside plan')
        step=plan['steps'][i]
        expected=self.task['start_region'] if i==0 else plan['steps'][i-1]['target_region']
        if observation['region_id']!=expected:raise ValueError('Observation is not at the current plan segment')
        if observation['completion_protocol_id']!=self.task['hard_constraints']['completion_protocol_id']:
            raise ValueError('Physical completion protocol cannot change between task and verification')
        reports=[]
        for candidate in step['candidates']:
            edge=self.edges[candidate['connection_id']];conditions=[]
            if edge['connection_id'] not in observation['available_connections']:conditions.append('connection_unavailable_or_entry_missed')
            if observation['prior_constraint_violation']:conditions.append('prior_hard_constraint_violation')
            if abs(observation['cross_track_m'])>edge['half_width_m']:conditions.append('outside_road_boundary')
            if observation['remaining_preparation_m']<edge.get('minimum_preparation_m',0):conditions.append('insufficient_preparation_distance')
            if conditions:
                assessment=dict(status='infeasible',unmet_conditions=conditions,predicted_completion_cost_s=None,cost_source=None,evidence=[])
            else:
                assessment=capability(edge,candidate['skill_id'],observation)
                if assessment['status'] not in ('feasible','infeasible','unknown'):raise ValueError('Invalid capability disposition')
                cost=assessment['predicted_completion_cost_s']
                if cost is not None and (not math.isfinite(cost) or cost<0):raise ValueError('Completion cost must be finite and nonnegative or unknown')
                if assessment['status']=='feasible' and (cost is None or not assessment['evidence']):
                    assessment=dict(assessment,status='unknown',unmet_conditions=assessment['unmet_conditions']+['missing_cost_or_physical_evidence'])
            limit=self.task['hard_constraints']['maximum_remaining_time_s']
            if assessment['status']=='feasible' and limit is not None and assessment['predicted_completion_cost_s']>limit:
                assessment=dict(assessment,status='infeasible',unmet_conditions=assessment['unmet_conditions']+['predicted_time_budget_exceeded'])
            reports.append(dict(candidate=candidate,**assessment))
        known=[r for r in reports if r['status']=='feasible'];chosen=min(known,key=lambda r:r['predicted_completion_cost_s']) if known else None
        return dict(status='feasible' if chosen else 'infeasible' if all(r['status']=='infeasible' for r in reports) else 'unknown',
                    step_index=i,chosen=chosen,candidates=reports,future_steps='pending_actual_state',
                    plan_complete=False,physical_execution_performed=False,query_wall_s=time.perf_counter()-started)

    def propose_with_feedback(self, proposals, observation, capability, *, allow_revision, max_queries=3):
        """Shared revision protocol for deterministic or LLM-produced plan iterators."""
        history=[];iterator=None if callable(proposals) else iter(proposals)
        for _ in range(max_queries if allow_revision else 1):
            plan=proposals(deepcopy(history)) if callable(proposals) else next(iterator,None)
            if plan is None:break
            report=self.verify_current(plan,observation,capability);history.append(dict(plan=deepcopy(plan),feedback=report))
            if report['status']=='feasible':return dict(status='current_segment_supported',plan=plan,verification=report,history=history)
            if not allow_revision or len(history)>=max_queries:break
        return dict(status='revision_required',plan=None,history=history)


class ApproachCapabilityTool:
    """Bind the frozen approach predictor and observed completion-time estimator.

    Bindings must be supplied by trusted execution setup, never the language plan.
    An approach score alone cannot certify a task or invent an arrival-time model.
    """
    def __init__(self, approach, empirical, bindings, *, minimum_score=.7):
        self.approach=approach;self.empirical=empirical;self.bindings=bindings;self.minimum_score=minimum_score

    def __call__(self, edge, skill, observation):
        binding=self.bindings.get((edge['connection_id'],skill))
        unknown=dict(status='unknown',unmet_conditions=['unbound_segment_skill_or_physics'],predicted_completion_cost_s=None,cost_source=None,evidence=[])
        if binding is None or binding['runtime_identity']!=observation['runtime_identity']:return unknown
        estimates=self.empirical.query(observation['feature_vector'],binding['task_id'],binding['skill_id'],binding['candidate_ids'])['estimates']
        choice=self.approach.choose(observation['feature_vector'],'exit',observation['current_reference']) if binding['use_approach_model'] else None
        if choice and choice['status']=='unknown':return dict(unknown,unmet_conditions=['outside_approach_model_support'],approach=choice)
        candidates=estimates
        if choice:
            commands=binding['candidate_commands'];selected=next((cid for cid,command in commands.items() if command==choice['candidate']),None)
            candidates=[e for e in estimates if e['candidate_id']==selected]
            predicted=next((e for e in choice['estimates'] if e['candidate']==choice['candidate']),None)
            if predicted is None or predicted['score']<self.minimum_score:
                return dict(status='infeasible' if predicted is not None else 'unknown',unmet_conditions=['approach_and_skill_score_below_declared_threshold'],predicted_completion_cost_s=None,cost_source=None,evidence=binding['evidence'],approach=choice)
        supported=[e for e in candidates if e['known'] and e['complete_probability']>=self.minimum_score and e['success_cost'] is not None]
        accepted=min(supported,key=lambda e:e['success_cost']) if supported else None
        status='feasible' if accepted else 'infeasible' if candidates and all(e['known'] for e in candidates) else 'unknown'
        return dict(status=status,unmet_conditions=[] if accepted else ['empirical_completion_not_supported'],
                    predicted_completion_cost_s=accepted['success_cost'] if accepted else None,
                    cost_source='observed_success_time_conditioned_on_completion' if accepted else None,
                    evidence=binding['evidence'],estimates=estimates,approach=choice,
                    execution_reference=binding['candidate_commands'][accepted['candidate_id']] if accepted else None,
                    scope='Current-segment empirical support only; no safety guarantee or future-state completion claim')


def compile_existing_branch_plan(planner, plan, base_spec, observation_region='junction'):
    """Compile the two existing map branches; never load a complete scene by ID.

    This adapter supports exactly the registered junction-to-goal connection
    chains. Other valid graph plans need an executor binding and are rejected.
    Numeric speed/alpha stays in the trusted execution config, not in the plan.
    """
    planner.validate(plan)
    if planner.task['start_region']!=observation_region or any(len(s['candidates'])!=1 for s in plan['steps']):
        raise ValueError('This execution adapter requires a resolved junction-to-goal plan')
    sequence=[s['candidates'][0]['connection_id'] for s in plan['steps']]
    branches={('prepare_jump','cross_platform','land_return','merge_straight','to_goal'):'original_jump',('diverge','side_passage','rejoin','to_goal'):'bypass'}
    route=branches.get(tuple(sequence))
    if route is None:raise ValueError('No execution binding for this graph connection chain')
    jumps=[s['candidates'][0]['skill_id'] for s in plan['steps'] if planner.skills[s['candidates'][0]['skill_id']]['family']=='jump']
    if jumps and jumps!=['jump_update2']:raise ValueError('Alternative jump policy is not yet bound to this executor')
    spec=deepcopy(base_spec)
    spec['ground_route']=planner.graph['assets']['original']['path']
    spec['forward_branch']=dict(map_contract=planner.graph['assets']['map_contract'],requested_route=route,decision_x_m=planner.regions['junction']['center_xy_m'][0],common_waypoint_count=1)
    spec['task_graph_plan']=deepcopy(plan)
    return spec
