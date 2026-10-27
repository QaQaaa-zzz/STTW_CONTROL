"""Forecast/actual/padding accounting stays in original lower-tick units."""
import pytest
from sttw_control.native_reference_rollout import limits


def test_end_of_task_prediction_shrinks_but_padding_is_separate():
    # Starts0,4,8,12: horizon8 gives8+8+8+4 actual ticks per candidate.
    b=limits(task_lower=16,hold_lower=4,prediction_lower=8,lanes=2,candidates=3)
    assert b['maximum_planning_rounds']==4
    assert b['maximum_prediction_lower_ticks']==28*6
    assert b['maximum_prediction_slots']==32*6
    assert b['maximum_actual_lower_ticks']==64
    assert b['maximum_actual_slots']==64


def test_declared_native_panel_units_and_invalid_upper_boundaries():
    assert limits()==dict(maximum_planning_rounds=50,maximum_prediction_lower_ticks=1996800,
                         maximum_prediction_slots=2080000,maximum_actual_lower_ticks=32000,maximum_actual_slots=32000)
    with pytest.raises(ValueError):limits(hold_lower=3)
    with pytest.raises(ValueError):limits(candidates=0)
