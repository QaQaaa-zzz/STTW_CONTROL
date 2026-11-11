import numpy as np
from sttw_control.handoff_continuation import label_window,mounted_features

def fixture(duration):
    t=np.arange(1,round(duration/.02)+1)*.02
    r=np.zeros((len(t),13));r[:,0]=t*1.5
    return t,r,np.zeros((len(t),5)),np.zeros(len(t),bool),np.ones(5)

def test_short_goal_is_censored_not_negative():
    x=label_window(*fixture(.1),handoff_s=0,end_code=13)
    assert x['continued_2s'] is None and x['envelope_entry_status']=='censored'

def test_terminal_failure_at_horizon_is_negative():
    x=label_window(*fixture(2),handoff_s=0,end_code=3)
    assert x['continued_2s'] is False

def test_hold_needs_elapsed_time_and_failure_can_follow_entry():
    x=label_window(*fixture(.2),handoff_s=0,end_code=3)
    assert x['envelope_entry_status']=='not_entered_held'
    x=label_window(*fixture(.5),handoff_s=0,end_code=3)
    assert x['envelope_entry_status']=='entered_held' and x['continued_2s'] is False

def test_mount_rotation_removed():
    r=np.zeros((1,13));r[:,[3,6]]=np.sqrt(.5);r[:,7]=1.5
    f=mounted_features(r,[np.sqrt(.5),-np.sqrt(.5),0,0])
    assert np.max(abs(f))<1e-12
