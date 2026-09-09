from copy import deepcopy
import numpy as np
from sttw_control.selection import rank_candidate


def row():
    return {'radial_rmse':[.15,.16],'speed_rmse':[.06,.07],
            'nominal_radial_rmse':[.14,.15],'failed':[False,False],
            'nominal_failed':[False,False],'event_present':[True,True],
            'post_event_hold_complete':[True,True],'settling_seconds':[2.,3.],
            'post_event_peak':[.2,.3]}


def test_rank_rejects_speed_and_nominal_regression_even_if_path_improves():
    base=row();candidate=deepcopy(base);candidate['radial_rmse']=[.01,.01]
    candidate['speed_rmse']=[.09,.07]
    assert rank_candidate(candidate,base)[0] is None
    candidate=deepcopy(base);candidate['nominal_radial_rmse']=[.2,.15]
    assert rank_candidate(candidate,base)[0] is None


def test_recovery_rank_prefers_completed_hold_over_lower_whole_episode_rmse():
    base=row();missing=deepcopy(base)
    missing['post_event_hold_complete']=[True,False]
    missing['settling_seconds']=[2.,10.]
    missing['radial_rmse']=[.01,.01]
    assert rank_candidate(base,base)[0]<rank_candidate(missing,base)[0]
    invalid=deepcopy(base);invalid['post_event_peak']=[np.nan,0.]
    assert rank_candidate(invalid,base)[0] is None


def test_invalid_reference_cannot_disable_regression_guards():
    base=row();candidate=row()
    base['speed_rmse']=[np.nan,.07]
    candidate['speed_rmse']=[.19,.07]
    assert rank_candidate(candidate,base)[1]=='nonfinite baseline reference'
    base=row();base['nominal_radial_rmse']=[None,.15]
    assert rank_candidate(candidate,base)[0] is None
    candidate=row();candidate['settling_seconds']=[None,3.]
    assert rank_candidate(candidate,row())[1]=='nonfinite validation'
