from copy import deepcopy
import pytest
from sttw_control.map_task_planning import MapTaskPlanner,ApproachCapabilityTool
from sttw_control.empirical_capability import ConnectionCapability


def fixture():
    edges=[dict(connection_id='jump',from_region='a',to_region='b',skills=['jump'],length_m=5.,half_width_m=.6,minimum_preparation_m=2.),dict(connection_id='around',from_region='a',to_region='b',skills=['drive'],length_m=8.,half_width_m=.6)]
    graph=dict(map_id='fixed',regions=[{'region_id':'a'},{'region_id':'b'}],connections=edges,skills=[dict(skill_id='jump',family='jump'),dict(skill_id='drive',family='ground')])
    task=dict(start_region='a',goal_region='b',hard_constraints=dict(forbidden_connections=[],ordered_required_regions=[],required_skill_families=[],maximum_remaining_time_s=10.,completion_protocol_id='strict'))
    obs=dict(step_index=0,region_id='a',available_connections=['jump','around'],prior_constraint_violation=False,cross_track_m=0.,remaining_preparation_m=3.,completion_protocol_id='strict',runtime_identity={'policy':'fixed'},feature_vector=[0.],current_reference=dict(speed_m_s=2.,alpha=.5))
    return MapTaskPlanner(graph,task),obs


def capability(edge,skill,obs):
    return dict(status='infeasible' if skill=='jump' else 'feasible',unmet_conditions=['entry_skill_limit'] if skill=='jump' else [],predicted_completion_cost_s=None if skill=='jump' else 6.,cost_source='fixture',evidence=['unit_fixture_not_physical_evidence'])


def test_same_map_goal_feedback_changes_route_without_changing_constraints():
    planner,obs=fixture();plans=planner.enumerate_plans();assert len(plans)==2
    frozen=planner.propose_with_feedback(plans,obs,capability,allow_revision=False)
    revised=planner.propose_with_feedback(plans,obs,capability,allow_revision=True)
    assert frozen['status']=='revision_required' and len(frozen['history'])==1
    assert revised['plan']['steps'][0]['candidates'][0]['connection_id']=='around'
    assert len(revised['history'])==2 and revised['plan']['hard_constraints']==planner.task['hard_constraints']
    assert not revised['verification']['physical_execution_performed']


@pytest.mark.parametrize('change',['speed','goal','constraint','unknown_skill','disconnected'])
def test_rejects_language_control_overrides_and_invalid_plans(change):
    planner,_=fixture();plan=planner.enumerate_plans()[0]
    if change=='speed':plan['steps'][0]['candidates'][0]['speed_m_s']=2.
    if change=='goal':plan['goal_region']='a'
    if change=='constraint':plan['hard_constraints']['maximum_remaining_time_s']=100.
    if change=='unknown_skill':plan['steps'][0]['candidates'][0]['skill_id']='invented'
    if change=='disconnected':plan['steps'][0]['target_region']='a'
    with pytest.raises(ValueError):planner.validate(plan)


def test_mandatory_jump_cannot_be_completed_by_bypass_or_revised_away():
    planner,_=fixture();bypass=planner.enumerate_plans()[1]
    planner.task['hard_constraints']['required_skill_families']=['jump']
    bypass['hard_constraints']=deepcopy(planner.task['hard_constraints'])
    with pytest.raises(ValueError,match='Required skill'):planner.validate(bypass)
    assert len(planner.enumerate_plans())==1


@pytest.mark.parametrize('field,value,reason',[('remaining_preparation_m',1.,'insufficient_preparation_distance'),('cross_track_m',.7,'outside_road_boundary'),('available_connections',[],'connection_unavailable_or_entry_missed'),('prior_constraint_violation',True,'prior_hard_constraint_violation')])
def test_hard_physical_conditions_prevent_capability_execution(field,value,reason):
    planner,obs=fixture();obs[field]=value
    def never(*args):pytest.fail('Invalid physical query must not call a capability model')
    result=planner.verify_current(planner.enumerate_plans()[0],obs,never)
    assert reason in result['candidates'][0]['unmet_conditions']


def test_missing_cost_is_unknown_and_future_segments_are_not_certified():
    planner,obs=fixture()
    def incomplete(*args):return dict(status='feasible',unmet_conditions=[],predicted_completion_cost_s=None,cost_source=None,evidence=[])
    r=planner.verify_current(planner.enumerate_plans()[0],obs,incomplete)
    assert r['status']=='unknown' and not r['plan_complete'] and r['future_steps']=='pending_actual_state'


def test_empirical_tool_returns_observed_conditional_cost_and_rejects_wrong_identity():
    planner,obs=fixture();rows=[dict(state=[0.],task_id='around',skill_id='drive',candidate_id='c',complete=1,success_cost=6.)]
    empirical=ConnectionCapability(rows,[1.],1.,.5)
    bindings={('around','drive'):dict(runtime_identity={'policy':'fixed'},task_id='around',skill_id='drive',candidate_ids=['c'],candidate_commands={'c':dict(speed_m_s=2.,alpha=.5)},use_approach_model=False,evidence=['fixture'])}
    tool=ApproachCapabilityTool(None,empirical,bindings)
    result=planner.verify_current(planner.enumerate_plans()[1],obs,tool)
    assert result['chosen']['predicted_completion_cost_s']==6.
    obs['runtime_identity']={'policy':'changed'}
    assert planner.verify_current(planner.enumerate_plans()[1],obs,tool)['status']=='unknown'


def test_cost_budget_is_enforced_without_changing_prediction():
    planner,obs=fixture();planner.task['hard_constraints']['maximum_remaining_time_s']=5.
    result=planner.verify_current(planner.enumerate_plans()[1],obs,capability)
    assert result['status']=='infeasible' and result['candidates'][0]['predicted_completion_cost_s']==6.


def test_proposer_receives_physical_feedback_before_revising():
    planner,obs=fixture();plans=planner.enumerate_plans();seen=[]
    def propose(history):
        seen.append(history)
        return plans[0] if not history else plans[1]
    result=planner.propose_with_feedback(propose,obs,capability,allow_revision=True)
    assert len(seen)==2 and seen[1][0]['feedback']['candidates'][0]['unmet_conditions']==['entry_skill_limit']
    assert result['plan']==plans[1]


def test_real_asset_graph_has_two_chains_to_same_goal_and_compiles_without_scene_ids():
    import json
    from pathlib import Path
    from sttw_control.map_task_planning import compile_existing_branch_plan
    folder=Path(__file__).resolve().parents[1]/'configs/chapter3/fixed_map'
    graph=json.loads((folder/'map.json').read_text());task=json.loads((folder/'task.json').read_text());planner=MapTaskPlanner(graph,task)
    plans=planner.enumerate_plans();assert len(plans)==2
    compiled=[compile_existing_branch_plan(planner,p,dict(skills=['update2'],physics_dt=.0025)) for p in plans]
    assert {c['forward_branch']['requested_route'] for c in compiled}=={'original_jump','bypass'}
    assert all(p['goal_region']==task['goal_region'] for p in plans)
    assert all(c['physics_dt']==.0025 and c['skills']==['update2'] for c in compiled)
    assert all('scene_id' not in str(p) for p in plans)
