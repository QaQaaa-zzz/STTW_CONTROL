import numpy as np
from sttw_control.directional_tolerance import impulse_margins,return_gramian,pulse_map_initial

def test_random_boundaries_use_same_budget_and_are_unit_invariant():
    rng=np.random.default_rng(20261022)
    for _ in range(200):
        q=rng.normal(size=(3,3));W=q@q.T+np.eye(3)*.2;e=rng.normal(size=3);D=rng.normal(size=3)
        budget=e@np.linalg.solve(W,e)+rng.uniform(.01,2.)
        m=impulse_margins(W,e,D,budget)
        for sign,p in [(1,m.plus),(-1,m.minus)]:
            y=e+sign*D*p
            np.testing.assert_allclose(y@np.linalg.solve(W,y),budget,rtol=1e-10)
        S=np.diag([.2,1.,3.]);scaled=impulse_margins(S@W@S.T,S@e,S@D,budget)
        np.testing.assert_allclose([m.plus,m.minus],[scaled.plus,scaled.minus],rtol=1e-9)

def test_scalar_finite_pulse_and_gramian_match_integrals():
    W=return_gramian([[1.]],[[1.]],[[2.]],1.)
    np.testing.assert_allclose(W,[[.21616617919084682]],rtol=1e-11)
    K=pulse_map_initial([[1.]],[1.],0.,.1)
    np.testing.assert_allclose(K,[.09516258196404048],rtol=1e-11)
