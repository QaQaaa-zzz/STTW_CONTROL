"""Offline capability labels never turn censoring into success."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

CLI = Path(__file__).parents[1] / 'cli'
sys.path.insert(0, str(CLI))
spec = importlib.util.spec_from_file_location('native_capability_analysis', CLI / 'analyze_native_capability.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(n=20):
    path = dict(s=np.arange(6.), xy=np.column_stack([np.arange(6.), np.zeros(6)]),
                curvature=np.array([0., 1., 0., -1., 0., 0.]), goal=np.array(5.), turn_end=np.array(4.))
    raw = dict(time=np.arange(n)*.1, actual_xy=np.tile([5.,0.],(n,1)),
               path_progress=np.full(n,5.), goal_section_signed_distance=np.zeros(n), path_cross_track=np.zeros(n), path_heading_error=np.zeros(n),
               actual_forward_speed=np.ones(n), v_user=np.ones(n),
               physical_failure=np.zeros(n,bool),domain_exit=np.zeros(n,bool),contact_failure=np.zeros(n,bool))
    return raw,path


def test_complete_final_window_and_both_bends():
    raw,path=fixture()
    out=module.labels(raw,path,.1,2.)
    assert out['final_1s_thresholds_met']
    assert out['reached_second_bend_before_native_stop']
    assert out['contact_observation']=='complete_no_contact'


def test_domain_stop_is_censored_and_fatal_goal_arrival_excluded():
    raw,path=fixture()
    raw['actual_xy'][:-1]=[0.,0.]
    raw['path_progress'][:-1]=0.
    raw['domain_exit'][-1]=True
    out=module.labels(raw,path,.1,2.)
    assert out['contact_observation']=='censored_no_contact'
    assert not out['goal1m_before_native_stop']
    assert not out['original_goal_crossed_before_native_stop']
    assert not out['final_1s_thresholds_met']


def test_full_window_required_instead_of_final_sample():
    raw,path=fixture()
    raw['path_heading_error'][-5]=.051
    assert not module.labels(raw,path,.1,2.)['final_1s_thresholds_met']


def test_goal_progress_without_final_section_crossing_is_not_goal_crossing():
    raw,path=fixture()
    raw["goal_section_signed_distance"][:]=-0.1
    assert not module.labels(raw,path,.1,2.)["original_goal_crossed_before_native_stop"]
