import numpy as np
from sttw_control import training_diagnostics as d


def test_alpha_bins_weight_samples_and_keep_signed_errors():
    # Boundary assignment, empty-bin nulls and sample weighting must not regress.
    a=np.array([0.,1/3,2/3,1.])
    stats=d.alpha_sample_sums(a,np.array([1.,2.,3.,5.]),np.array([0,0,1,0]),{'speed':np.array([-2.,1.,3.,-1.])},xp=np)
    result=d.alpha_sample_summary(stats)
    assert [r['samples'] for r in result]==[1,1,2]
    assert result[2]['mean_step_reward']==4
    assert result[2]['physical_failures']==1
    assert result[2]['speed_mean_error']==1
    assert result[2]['speed_rmse']==np.sqrt(5)
    empty=d.alpha_sample_summary(d.alpha_sample_sums(np.array([.5]),np.ones(1),np.zeros(1),{'speed':np.zeros(1)},xp=np))
    assert empty[0]['mean_step_reward'] is None
    assert empty[0]['speed_rmse'] is None


def test_nonfinite_terminal_errors_are_counted_without_poisoning_json():
    stats=d.alpha_sample_sums(np.array([.5,.5]),np.array([-100.,1.]),np.array([1,0]),{'speed':np.array([np.nan,2.])},xp=np)
    row=d.alpha_sample_summary(stats)[1]
    assert row['samples']==2 and row['physical_failures']==1
    assert row['speed_valid_samples']==1
    assert row['speed_invalid_samples']==1
    assert row['speed_rmse']==2


def test_core_tensorboard_exposes_alpha_errors_and_denominators():
    from sttw_control.tensorboard_logging import scalar_values
    row={'alpha_training_samples':[{'alpha_lower':0.,'alpha_upper':1/3,'upper_inclusive':False,'samples':12,'mean_step_reward':.02,'speed_m_s_rmse':.1,'physical_failures':0}]}
    got=scalar_values(row,profile='core')
    assert got['alpha_training/bin_0/samples']==12
    assert got['alpha_training/bin_0/speed_m_s_rmse']==.1
