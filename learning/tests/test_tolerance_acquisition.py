import numpy as np
import pytest
from sttw_control.tolerance_acquisition import fit_memory, acquisition_score, query_features, BASE_FIELDS

def test_acquisition_tracks_recoverable_examples_not_monotone_force():
    m=fit_memory([[0.],[1.],[2.]], [0,1,0], k=1)
    np.testing.assert_array_equal(acquisition_score(m, [[.1],[.9],[1.9]]), [0,1,0])

def test_query_does_not_change_training_normalization():
    m=fit_memory([[0.,3.],[2.,3.]], [0,1], k=2)
    old=repr(m)
    scores=acquisition_score(m, [[1e6,3.],[-1e6,3.]])
    assert np.isfinite(scores).all() and repr(m)==old
    assert scores[0]>scores[1]

def test_signed_pulse_features_and_bad_inputs():
    f={x:0. for x in BASE_FIELDS};f.update(remaining_s=4.,lambda_u=7.,b_u=-2.,z_u=.3)
    a=query_features(f,20,.1,hybrid=True);b=query_features(f,-20,.1,hybrid=True)
    assert a[-3]==-b[-3] and a[-1]!=b[-1]
    with pytest.raises(ValueError):query_features(f,1,-1)
    with pytest.raises(ValueError):fit_memory([[1.]], [2])
