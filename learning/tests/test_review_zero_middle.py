"""Check paired claims and contact/cost separation in the offline review."""
import json
import importlib.util
from pathlib import Path


def test_review_keeps_four_factorial_contrasts_and_excludes_contact_from_full_cost(tmp_path):
    file=Path(__file__).parents[1]/'cli/review_feedback_confirmation.py'
    spec=importlib.util.spec_from_file_location('review_feedback',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    roles=['zero_middle','free_GAE','zero_middle_feedback','feedback_010'];rows=[]
    for force in [0.,16.]:
        for role in roles:
            contact=force!=0 and role=='zero_middle'
            arrived=role in ['free_GAE','feedback_010']
            rows.append(dict(case_id=f'left/{force}/{role}',role=role,context='left',alpha=0.,event_id=str(force),
                force_N=force,pulse_onset_task_tick=600,pulse_duration_s=.1,world_direction=[0,1,0],requested_speed_m_s=1.,
                goal_arrived=arrived,minimum_upright_goal_distance_m=.15 if arrived else 1.,
                arrival_outcome='arrived' if arrived else 'not_arrived',physical_outcome='contact' if contact else 'survived',
                scored_cost=1. if contact else 10.))
    (tmp_path/'analysis').mkdir()
    for name in ['status.json','analysis/status.json']:(tmp_path/name).write_text(json.dumps(dict(phase='completed')))
    (tmp_path/'plan.json').write_text(json.dumps(dict(evaluation_roles=roles,goal_radius_m=.5)))
    (tmp_path/'analysis/results.json').write_text(json.dumps(rows))
    result=module.review(tmp_path,tmp_path/'review.json')
    for key in ['free_GAE_vs_zero_middle','feedback_010_vs_zero_middle_feedback',
                'zero_middle_feedback_vs_zero_middle','feedback_010_vs_free_GAE']:
        assert key in result['paired']
    pair=result['paired']['free_GAE_vs_zero_middle']
    assert len(pair['physical_gains'])==1 and pair['both_survived']==0
    assert pair['mean_both_survived_cost_delta'] is None
    assert len(pair['radii']['0.5']['gains'])==1 and len(pair['radii']['0.1']['gains'])==0


def test_alpha_trace_diagnostic_reports_short_observation_without_padding_or_exact_replay_gate():
    import numpy as np
    file=Path(__file__).parents[1]/'cli/audit_zero_middle_execution.py'
    spec=importlib.util.spec_from_file_location('zero_execution_audit',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    a={key:np.zeros((3,2)) for key in module.SIGNALS}
    b={key:np.zeros((2,2)) for key in module.SIGNALS};b['actual_xy'][1,0]=.0001
    result=module.compare_alpha_traces([a,b])
    assert result['observed_lengths']==[3,2] and result['common_ticks']==2
    assert result['max_abs_difference']['actual_xy']==.0001
