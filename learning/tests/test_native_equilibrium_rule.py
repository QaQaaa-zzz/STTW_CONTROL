"""Reference-budget semantics and native latent permissions, no physics."""
import jax.numpy as j
import numpy as np
import pytest
from sttw_control.native_equilibrium_rule import target
from sttw_control.path_command_policy import map_latent

CFG=dict(residual_interface=dict(speed_reference_bounds_m_s=[1.,3.5],speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.5,steer_scale_rad=.2))


@pytest.mark.parametrize('sign',[-1.,1.])
def test_speed_priority_keeps_requested_speed_and_steer_direction(sign):
    nominal=j.array([2.,sign*.2]);z,m=target(nominal,1.,CFG,lambda v:1/v**2,.3)
    realized=nominal+map_latent(z,CFG)
    np.testing.assert_allclose(realized,m['realized_target'],atol=1e-7)
    assert float(realized[0])==2.
    assert float(realized[1])*sign>0
    assert abs(float(m['static_lean_target']))<=.300001


@pytest.mark.parametrize('sign',[-1.,1.])
def test_path_priority_preserves_nominal_steering_and_uses_legal_speed(sign):
    nominal=j.array([2.,sign*.2]);z,m=target(nominal,0.,CFG,lambda v:1/v**2,.3)
    realized=nominal+map_latent(z,CFG)
    assert 1.<=float(realized[0])<2.
    np.testing.assert_allclose(realized[1],nominal[1],atol=1e-7)
    assert bool(m['grid_has_feasible'])
    assert abs(float(m['static_lean_target']))<=.300001


def test_unattainable_budget_is_reported_not_called_safe():
    cfg=dict(residual_interface=dict(CFG['residual_interface'],speed_reference_bounds_m_s=[1.5,3.5]))
    z,m=target(j.array([2.,.2]),0.,cfg,lambda v:1/v**2,.3)
    assert not bool(m['grid_has_feasible'])
    assert float(m['static_lean_target'])>.3
    assert np.isfinite(z).all()


def test_no_intervention_in_small_nominal_workload():
    for alpha in (0.,1.):
        z,m=target(j.array([2.,.02]),alpha,CFG,lambda v:1/v**2,.3)
        np.testing.assert_array_equal(z,j.zeros(2))
