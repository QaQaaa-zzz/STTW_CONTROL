"""Recover authentic task value inputs and GAE without inventing hold actions."""
import numpy as np
import pytest
from sttw_control.survival_task_credit import reconstruct_task_critic, complete_task_gae


def test_critic_appends_remaining_time_only_on_saved_pre_action_rows():
    obs=np.full((3,351),np.nan,dtype=np.float32);obs[:2]=np.arange(351,dtype=np.float32)
    out=reconstruct_task_critic(obs,np.array([0,400,-1]),np.array([True,True,False]),
                                dt_s=.005,horizon_s=10.)
    np.testing.assert_array_equal(out[:2,:351],obs[:2])
    np.testing.assert_allclose(out[:2,-1],[1.,.8])
    assert np.isnan(out[2]).all()


def test_task_terminal_stops_bootstrap_and_preserves_padding_nan():
    out=complete_task_gae(np.array([1.,2.,np.nan]),np.array([.5,.75,np.nan]),
                          np.array([True,True,False]),np.array([False,True,False]),gamma=.9,gae_lambda=.8)
    # last delta=2-.75=1.25; first delta=1+.9*.75-.5=1.175
    np.testing.assert_allclose(out['advantages'][:2],[2.075,1.25])
    np.testing.assert_allclose(out['returns'][:2],[2.575,2.])
    assert np.isnan(out['advantages'][2:]).all()


@pytest.mark.parametrize('valid,terminal', [([True,False,True],[False,False,True]),
                                           ([True,True,False],[False,False,False]),
                                           ([True,True,False],[True,True,False])])
def test_incomplete_or_cross_reset_task_batch_is_rejected(valid,terminal):
    with pytest.raises(ValueError):
        complete_task_gae(np.ones(3),np.zeros(3),np.array(valid),np.array(terminal),gamma=.9,gae_lambda=.8)
