import numpy as np
from sttw_control.lower_command_validation import summarize_parent_recovery


def trace(start=0.4):
    t=np.arange(400)*.005
    return dict(time=t,actual_forward_speed=np.full(400,2.3),
                actual_delta=np.where(t<start,.05,.02),
                limited_command=np.tile([2.3,0.],(400,1)),
                peak_roll=np.full(400,.2),physical_failure=np.zeros(400,bool))


def test_recovery_requires_timely_continuous_hold_and_no_reexit():
    assert summarize_parent_recovery(trace(),initial_roll=.2)['qualified']
    d=trace();d['actual_delta'][350]=.05
    assert not summarize_parent_recovery(d,initial_roll=.2)['qualified']
    assert not summarize_parent_recovery(trace(1.1),initial_roll=.2)['qualified']


def test_recovery_rejects_new_working_limit_and_incomplete_trace():
    d=trace();d['peak_roll'][2]=.303
    assert not summarize_parent_recovery(d,initial_roll=.2)['qualified']
    d={k:v[:-1] for k,v in trace().items()}
    assert not summarize_parent_recovery(d,initial_roll=.2)['qualified']


def test_one_second_deadline_uses_post_step_measurement_time():
    assert summarize_parent_recovery(trace(.995),initial_roll=.2)['qualified']
    assert not summarize_parent_recovery(trace(1.),initial_roll=.2)['qualified']
