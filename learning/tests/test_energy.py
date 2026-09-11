import numpy as np
from sttw_control.energy import mechanical_work


def trace():
    return {'time':np.array([0.,.5,1.]),'actuator_force':np.array([[2.,-3.]]*3),
            'actuator_velocity':np.ones((3,2)),'qpos':np.array([[0.,0.],[.5,0.],[1.,0.]])}


def test_positive_and_negative_work_do_not_cancel_across_actuators():
    r=mechanical_work(trace())
    assert r['available']
    assert r['positive_work_j']==2. and r['negative_work_magnitude_j']==3.
    assert r['positive_work_per_planar_meter_j_m']==2.
    np.testing.assert_allclose(r['positive_work_per_actuator_j'],[2.,0.])


def test_missing_invalid_and_zero_distance_are_not_zero_energy_claims():
    assert not mechanical_work({})['available']
    t=trace();t['actuator_force'][1,0]=np.nan
    assert not mechanical_work(t)['available']
    t=trace();t['time'][1]=t['time'][0]
    assert not mechanical_work(t)['available']
    t=trace();t['qpos'][:]=0
    r=mechanical_work(t)
    assert r['positive_work_per_planar_meter_j_m'] is None
    assert r['positive_work_j']==2.
