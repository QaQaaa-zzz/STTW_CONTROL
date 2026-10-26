import numpy as np
import pytest

from sttw_control.feedback_timing_audit import force_window,cost_partition,angle_delta


def test_observed_impulse_boundary_excludes_affected_transition_from_prefix():
    impulse=np.zeros((6,3));impulse[2:4,1]=-.08;impulse[5]=np.nan
    assert force_window([1,1,1,1,0,0],impulse)==(2,4)
    partition=cost_partition([1.,2.,100.,200.],2)
    assert partition==dict(before=3.,after=300.,total=303.)


def test_no_force_is_not_a_force_at_zero_and_partition_retains_signed_costs():
    assert force_window(np.ones(4,bool),np.zeros((4,3))) is None
    x=cost_partition([1.,-2.,4.],1)
    assert x['before']+x['after']==x['total']==3.
    with pytest.raises(ValueError):cost_partition([1.],2)
    with pytest.raises(ValueError):force_window([1],[[np.nan,0,0]])


def test_yaw_difference_wraps_at_pi_instead_of_creating_full_turn_change():
    np.testing.assert_allclose(angle_delta(-np.pi+.01,np.pi-.01),.02,atol=1e-12)
