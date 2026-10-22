import numpy as np
from sttw_control.directional_tolerance import local_constraint_margin, impulse_margins

def test_constraint_unit_invariance():
    for scale in [1e-18, 1e-13, 1., 1e13]:
        r=local_constraint_margin([-scale],[scale],10.)
        assert r['value']==1.
        assert r['status']=='LOCAL_PROXY'

def test_ill_conditioning_is_unknown_not_physical_unreachability():
    r=impulse_margins(np.diag([1.,1e-12]),[0.,0.],[0.,1.],1.)
    assert r.status=='UNKNOWN_NUMERICAL'
    assert np.isnan(r.plus)

def test_boundary_and_reachable_zero_channel():
    r=impulse_margins([[1.]],[.5],[1.],.25)
    assert r.plus==0. and r.minus==1.
    assert impulse_margins([[1.]],[0.],[0.],1.).status=='NO_EFFECT_IN_SELECTED_MODEL'

def test_no_tolerance_acceptance_of_nominal_failure():
    assert local_constraint_margin([1e-15],[1.],10.)['status']=='NOMINAL_FAILURE'

def test_exact_singular_unresolved_is_not_a_certificate():
    assert impulse_margins(np.diag([1.,0.]),[0.,0.],[0.,1.],1.).status=='UNKNOWN_NUMERICAL'
