import numpy as np
import pytest
from sttw_control.recovery_teaching import teaching_enabled, extract_teaching


def test_opt_in_requires_training_contract():
    assert not teaching_enabled({})
    with pytest.raises(ValueError):
        teaching_enabled({'teaching_records': {'role': 'DEV', 'observation_dim': 351}})
    assert teaching_enabled({'teaching_records': {'role': 'TRAIN', 'observation_dim': 351}, 'recovery_intervention': {'declared': True}})


def test_pre_action_alignment_partial_terminal_and_parent_separate():
    d=dict(time=np.arange(8)*.005,active_tick=np.array([1,1,1,1,1,0,0,0],bool),
        teaching_observation=np.repeat(np.array([[11.]*351,[22.]*351]),4,axis=0),
        teaching_target=np.repeat([[0.,.7],[.2,.8]],4,axis=0),
        teaching_parent=np.repeat([[.5,.7],[.2,.8]],4,axis=0))
    d['teaching_observation'][5:]=np.nan
    got=extract_teaching(d)
    np.testing.assert_allclose(got['time'],[0,.02])
    np.testing.assert_allclose(got['observations'][:,0],[11,22])
    np.testing.assert_allclose(got['targets'][:,0],[0,.2])
    np.testing.assert_allclose(got['parent_means'][:,0],[.5,.2])
    d['teaching_observation'][4,0]=np.nan
    with pytest.raises(ValueError):extract_teaching(d)
