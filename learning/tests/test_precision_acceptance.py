import numpy as np
from sttw_control.tracking_diagnostics import precision_speed_acceptance

def test_only_full_postcommand_window_is_accepted_and_failure_cannot_hide_errors():
    cfg={'controller':{'dt':1.},'horizon_seconds':5.,'timed_reference':{'fixed':[[0,2,0],[3,2,1]]},'tracking':{}}
    def trace(errors):
        return {'time':np.arange(6.),'true_forward_speed':np.r_[2,np.asarray(errors)+2.], 'reference_command':np.tile([2.,0.],(6,1)), 'terminated':np.zeros(6,dtype=bool)}
    base=trace([0,0,0,.1,.1]);good=trace([1,1,1,.07,.07])
    x=precision_speed_acceptance(good,base,cfg,ratio_target=.8)
    assert x['passed'] and x['samples']==2 and x['policy_rmse_m_s']<.071
    bad=trace([0,0,0,.09,.09]);assert not precision_speed_acceptance(bad,base,cfg)['passed']
    good['terminated'][-1]=True;assert not precision_speed_acceptance(good,base,cfg)['passed']
