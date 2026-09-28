import pytest


def test_metric_extraction_does_not_treat_missing_reward_as_zero():
    from sttw_control.training_diagnostics import series
    import math
    rows=[{'update':1,'control_transitions':100,'loss_metrics':[1,2,3,4]},
          {'update':2,'control_transitions':200,'mean_step_reward':.2,'loss_metrics':[2,3,4,5],
           'validation':{'event_present':[False,True,True],'failed':[False,False,True],'post_event_hold_complete':[False,True,False]}}]
    s=series(rows)
    assert math.isnan(s['reward'][0]) and s['reward'][1]==.2
    assert s['kl']==[4,5] and s['recovery'][1]==.5 and s['failed'][1]==.5


def test_logged_reward_components_have_inspectable_plot_data(tmp_path):
    import json
    from sttw_control.training_diagnostics import plot_reward_components
    rows=[dict(update=1,control_transitions=64,mean_step_reward=-.1,reward_components_sum_mean_step=-.1,reward_components_reconstruction_max_abs=1e-8,reward_components_mean_step={'alive':.005,'speed':-.105})]
    plot_reward_components(rows,tmp_path)
    assert (tmp_path/'reward_components.png').is_file()
    assert (tmp_path/'reward_components.pdf').is_file()
    assert json.loads((tmp_path/'reward_components.json').read_text())[0]['parts']['speed']==-.105

def test_directional_speed_statistics_pool_peaks_by_max_not_sum():
    import numpy as np
    from sttw_control.training_diagnostics import alpha_sample_sums,alpha_sample_summary,merge_sample_statistics
    def sample(errors):
        return alpha_sample_sums(np.array([0.,0.,1.]),np.zeros(3),np.zeros(3),{'speed_m_s':np.array(errors)},xp=np)
    a=sample([-.4,.1,float('nan')]);b=sample([-.2,.2,-.1])
    pooled=merge_sample_statistics(a,b,maximum=np.maximum)
    rows=alpha_sample_summary(pooled,dt=.005)
    assert rows[0]['underspeed_peak_m_s']==pytest.approx(.4)
    assert rows[0]['overspeed_peak_m_s']==pytest.approx(.2)
    assert rows[0]['underspeed_environment_seconds']==pytest.approx(.01)
    assert rows[0]['overspeed_integral_m']==pytest.approx(.0015)
    assert rows[0]['overspeed_above_band_environment_seconds']==pytest.approx(.01)
    assert rows[2]['speed_m_s_invalid_samples']==1
    assert rows[1]['overspeed_peak_m_s'] is None

def test_trace_directional_durations_break_at_invalid_samples():
    import numpy as np
    from sttw_control.training_diagnostics import directional_speed_trace
    d=directional_speed_trace(np.array([.1,.2,np.nan,.1,-.4,-.2,0.]),.005,.05)
    assert d['overspeed_peak_m_s']==pytest.approx(.2)
    assert d['overspeed_seconds']==pytest.approx(.015)
    assert d['overspeed_longest_seconds']==pytest.approx(.01)
    assert d['underspeed_integral_m']==pytest.approx(.003)
    assert d['invalid_samples']==1
