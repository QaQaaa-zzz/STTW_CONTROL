import numpy as np
import pytest
from sttw_control.tolerance_outcome_audit import split_outcome

def data(roll):
 n=len(roll)
 return dict(time=np.arange(n)*.005,peak_roll=np.asarray(roll),physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool))

def test_prior_violation_does_not_become_future_violation():
 x=split_outcome(data([.4,.1,.1,.1]),onset=.01,limit=.302,final_hold=True,goal=True)
 assert x['prefix_working_failure'] and not x['suffix_working_failure']
 assert x['suffix_task_success'] and not x['reconstructed_full_success']

def test_first_pulse_tick_belongs_to_suffix():
 x=split_outcome(data([.1,.1,.4,.1]),onset=.01,limit=.302,final_hold=True,goal=True)
 assert not x['prefix_working_failure'] and x['suffix_working_failure']
 assert not x['suffix_task_success']

def test_no_suffix_is_unknown_not_success():
 x=split_outcome(data([.1,.1]),onset=.01,limit=.302,final_hold=False,goal=False)
 assert x['suffix_task_success'] is None

def test_final_task_failure_is_retained_with_safe_roll():
 x=split_outcome(data([.1]*4),onset=.01,limit=.302,final_hold=False,goal=True)
 assert not x['suffix_working_failure'] and not x['suffix_task_success']
