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
