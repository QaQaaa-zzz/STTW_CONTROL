"""Mathematical checks for conditional scores; no controller safety claim."""
import json
import numpy as np
import pytest
from sttw_control.conditional_tolerance import (
    fit_conditional_model,predict_conditional,force_crossing,logistic_objective)


def toy():
    state=np.repeat(np.array([[0.],[1.]]),9,axis=0)
    amp=np.tile(np.arange(0.,81.,10.),2)
    fell=amp>=np.repeat([30.,60.],9)
    return state,amp,fell


def test_logistic_gradient_matches_finite_difference():
    rng=np.random.default_rng(19);design=np.c_[rng.normal(size=(20,4)),np.ones(20)]
    y=rng.integers(0,2,20);theta=rng.normal(size=5)
    _,grad=logistic_objective(theta,design,y,l2=1.)
    eps=1e-5
    numerical=np.array([(logistic_objective(theta+np.eye(5)[i]*eps,design,y,l2=1.)[0]-
        logistic_objective(theta-np.eye(5)[i]*eps,design,y,l2=1.)[0])/(2*eps) for i in range(5)])
    np.testing.assert_allclose(grad,numerical,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('kind',['linear','rbf'])
def test_fitted_scores_monotone_and_force_crossing_inverts_probability(kind):
    x,a,y=toy();model=fit_conditional_model(x,a,y,kind=kind,state_scale=[1.],max_centers=4)
    query=np.zeros((101,1));amps=np.arange(101.)
    p=predict_conditional(model,query,amps)['probability']
    assert np.all(np.diff(p)>=-1e-12)
    crossing=force_crossing(model,np.zeros((1,1)),probability=.5)
    assert crossing['positive_slope'][0] and crossing['force_N'][0]>0
    q=predict_conditional(model,np.zeros((1,1)),crossing['force_N'])['probability']
    np.testing.assert_allclose(q,[.5],atol=1e-8)
    restored=json.loads(json.dumps(model))
    np.testing.assert_array_equal(predict_conditional(restored,query,amps)['probability'],p)


def test_invalid_scales_and_negative_query_amplitudes_rejected():
    x,a,y=toy()
    with pytest.raises(ValueError):fit_conditional_model(x,a,y,kind='linear',state_scale=[0.])
    model=fit_conditional_model(x,a,y,kind='linear',state_scale=[1.])
    with pytest.raises(ValueError):predict_conditional(model,[[0.]],[-1.])


def test_zero_slope_reports_unidentified_crossing_and_negative_slope_is_invalid():
    x,a,y=toy();model=fit_conditional_model(x,a,y,kind='linear',state_scale=[1.])
    model['theta'][-2]=0.
    result=force_crossing(model,[[0.]])
    assert not result['positive_slope'][0] and np.isnan(result['force_N'][0])
    model['theta'][-2]=-1.
    with pytest.raises(ValueError):predict_conditional(model,[[0.]],[40.])
