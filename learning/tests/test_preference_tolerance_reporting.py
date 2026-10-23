import math
import pytest
from sttw_control.preference_tolerance_reporting import normalized_risk


def test_directional_risk_keeps_signed_force_and_original_budget():
    f=dict(lambda_u=7.,b_u=-2.,z_u=.2,remaining_s=8.)
    spec=dict(effort_budget=.03,gamma=.8,duration=.1)
    assert normalized_risk(f,8.,**spec)>normalized_risk(f,-8.,**spec)
    w=4*(-math.expm1(-14))/14
    assert normalized_risk(f,0.,**spec)==pytest.approx(.2/math.sqrt(.03*w))
    assert normalized_risk(f,0.,**spec)==normalized_risk(dict(f,remaining_s=2.),0.,**spec)
    assert normalized_risk(None,8.,**spec) is None
