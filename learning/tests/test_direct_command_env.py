from importlib.util import find_spec

def test_independent_task_environment_exists():
    assert find_spec('sttw_control.direct_command_env') is not None

def test_budget_persists(tmp_path):
    import json
    from pathlib import Path
    from sttw_control.direct_command_budget import ComputeBudget,BudgetStop
    spec=json.loads((Path(__file__).parents[1]/'configs/STTW_Direct_Command_V3.json').read_text())
    b=ComputeBudget(tmp_path,spec);b.reserve(b.tick_limit,'reservation only')
    try:ComputeBudget(tmp_path,spec).reserve(1,'must reject')
    except BudgetStop:pass
    else:raise AssertionError('budget reset')
