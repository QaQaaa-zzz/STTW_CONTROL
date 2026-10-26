import math
import pytest
from sttw_control.reference_bias_runtime import nominal_objective


def test_arrival_does_not_hide_later_contact_and_unknown_is_never_infeasible():
    assert nominal_objective(physical_outcome='survived',goal_arrived=True,cost=12.)==12.
    assert math.isinf(nominal_objective(physical_outcome='survived',goal_arrived=False,cost=12.))
    assert math.isinf(nominal_objective(physical_outcome='contact',goal_arrived=True,cost=12.))
    for kwargs in [dict(physical_outcome='unknown',goal_arrived=False,cost=12.),dict(physical_outcome='survived',goal_arrived=None,cost=12.),dict(physical_outcome='survived',goal_arrived=True,cost=float('nan'))]:
        with pytest.raises(ValueError,match='unknown'):nominal_objective(**kwargs)


def test_objective_rows_encode_strict_json_including_numpy_scalar_infeasibility():
    import json,numpy as np
    from sttw_control.reference_bias_runtime import encode_objective
    for value,feasible in [(np.float64(1.2),True),(np.float64('inf'),False)]:
        result=encode_objective(dict(objective=value))
        decoded=json.loads(json.dumps(result,allow_nan=False))
        assert decoded['nominal_feasible'] is feasible
        assert (decoded['objective'] is not None)==feasible
    with pytest.raises(ValueError,match='unknown'):encode_objective(dict(objective=float('nan')))
