"""Periodic geometry must be a matched comparison contract."""
import copy
import pytest
from sttw_control.goal_endpoint_analysis import _validate_h75_comparison
from sttw_control.goal_endpoint_run import h75_supplement_cases
from sttw_control.goal_evaluation import evaluation_cases


def test_replanning_comparison_accepts_legacy_and_rejects_unmatched_geometry():
    primary=dict(training='/training',endpoint='/endpoint',horizon_s=10.,radius_m=1.,
                 cases=evaluation_cases(),inputs={'/training/plan.json':'p',
                    '/training/effective_config.json':'c','/endpoint':'e'},
                 sources={'/'+name:'shared' for name in (
                    'teleop_env.py','path_command_env.py','live_path_continuation.py',
                    'anticipatory_reference.py','reference_reentry.py','goal_evaluation.py',
                    'goal_endpoint_run.py','geometric_path.py','contact_only_environment.py')})
    supplement=copy.deepcopy(primary)
    supplement.update(h75_supplement=True,cases=h75_supplement_cases(primary['cases']))
    _validate_h75_comparison(primary,supplement,{}, {})
    supplement['replanning']=None
    _validate_h75_comparison(primary,supplement,{}, {})
    supplement['replanning']={'period_task_ticks':200,'radius_m':1.8}
    with pytest.raises(ValueError,match='replanning'):
        _validate_h75_comparison(primary,supplement,{}, {})
    primary['replanning']=copy.deepcopy(supplement['replanning'])
    _validate_h75_comparison(primary,supplement,{}, {})
