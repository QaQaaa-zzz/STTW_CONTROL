import math
import pytest
from sttw_control.reference_bias_runtime import nominal_objective


def test_arrival_does_not_hide_later_contact_and_unknown_is_never_infeasible():
    assert nominal_objective(physical_outcome='survived',goal_arrived=True,cost=12.)==12.
    assert math.isinf(nominal_objective(physical_outcome='survived',goal_arrived=False,cost=12.))
    assert math.isinf(nominal_objective(physical_outcome='contact',goal_arrived=True,cost=12.))
    for kwargs in [dict(physical_outcome='unknown',goal_arrived=False,cost=12.),dict(physical_outcome='survived',goal_arrived=None,cost=12.),dict(physical_outcome='survived',goal_arrived=True,cost=float('nan'))]:
        with pytest.raises(ValueError,match='unknown'):nominal_objective(**kwargs)
