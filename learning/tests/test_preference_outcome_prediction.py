import numpy as np
import pytest

from sttw_control.preference_outcome_prediction import outcome_features,STATE_INDICES


def example():
    obs=np.arange(351,dtype=np.float32)/100.;obs[336]=.5
    case=dict(alpha=.5,source_task_start_s=1.6,preparation_s=.6,onset_s=2.2,force_N=24.,duration_s=.4,force_direction_world=[0.,1.,0.])
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2)
    return obs,np.array([.4,-.3]),case,interface


def test_features_use_source_observation_and_explicit_counterfactual_not_future_outcomes():
    obs,z,case,interface=example();x=outcome_features(obs,z,np.pi/2,case,interface)
    assert x.shape==(42,);np.testing.assert_array_equal(x[:34],obs[list(STATE_INDICES)])
    np.testing.assert_allclose(x[-8:],[.5,.25*np.tanh(.4),np.tanh(-.3),.75,0.,1.,1.,.84],atol=1e-6)
    changed=dict(case,goal_arrived=True,contact_failure=True,future_roll=100.,future_speed=-100.)
    np.testing.assert_array_equal(x,outcome_features(obs,z,np.pi/2,changed,interface))


def test_feature_query_rejects_alpha_and_preparation_clock_mismatch():
    obs,z,case,interface=example()
    with pytest.raises(ValueError,match='candidate alpha'):outcome_features(obs,z,0.,dict(case,alpha=1.),interface)
    with pytest.raises(ValueError,match='preparation'):outcome_features(obs,z,0.,dict(case,onset_s=2.3),interface)
