"""Conditional stopping-distance geometry, independently checked against kinematics."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('braking',Path(__file__).parents[1]/'cli/analyze_roll_braking_boundary.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_boundary_matches_exact_stop_and_reflection():
    p=np.array([10.,20.,30.]);w=np.array([40.,80.,100.]);b=500.;c=40.
    stop=w/b
    endpoint=p+w*stop-.5*b*stop**2
    hp,hm,h=m.margin(p,w,c,b)
    np.testing.assert_allclose(hp,c-endpoint)
    np.testing.assert_allclose(h,m.margin(-p,-w,c,b)[2])
    np.testing.assert_allclose(m.margin(np.radians(p),np.radians(w),np.radians(c),np.radians(b))[2],np.radians(h),atol=1e-12)

def test_demand_is_exact_boundary_and_not_fixed_capacity():
    p=np.array([10.,-20.,30.]);w=np.array([40.,-80.,-100.]);c=40.
    b=m.demand(p,w,c)
    np.testing.assert_allclose(m.margin(p,w,c,b)[2],0,atol=1e-12)
    # H500<0 does not exclude recovery if actual achievable braking is1000.
    p=np.array([30.]);w=np.array([120.])
    assert m.margin(p,w,40.,500.)[2][0]<0
    assert (p+w**2/(2*1000.))[0]<40.

if __name__=='__main__':
    test_boundary_matches_exact_stop_and_reflection()
    test_demand_is_exact_boundary_and_not_fixed_capacity()
    print('2 braking boundary tests passed')
