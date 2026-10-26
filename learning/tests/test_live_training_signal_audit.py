import numpy as np
import pytest
from sttw_control.live_training_signal_audit import discounted_returns,value_calibration


def test_complete_episode_returns_ignore_inactive_padding_and_preserve_terminal_cost():
    rewards=np.array([[-1.,-2.],[-3.,-8.],[-4.,99.]])
    active=np.array([[True,True],[True,True],[True,False]])
    got=discounted_returns(rewards,active,gamma=.5)
    np.testing.assert_allclose(got[:,0],[-3.5,-5.,-4.])
    np.testing.assert_allclose(got[:2,1],[-6.,-8.]);assert np.isnan(got[2,1])
    with pytest.raises(ValueError):discounted_returns(rewards,np.array([[True,True],[False,True],[True,False]]),gamma=.5)


def test_value_calibration_distinguishes_constant_bias_from_explained_variance():
    target=np.array([-10.,-4.,-1.]);pred=target+2.
    r=value_calibration(pred,target,np.ones(3,bool))
    assert r['mean_prediction_minus_target']==2. and r['rmse']==2. and r['explained_variance']==1.
    assert value_calibration(pred,target,np.zeros(3,bool))['samples']==0
