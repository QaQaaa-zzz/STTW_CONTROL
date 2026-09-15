from copy import deepcopy
import numpy as np
from sttw_control.selection import rank_tracking_candidate


def bank():
    return dict(radial_rmse=[.04],speed_rmse=[.3],heading_rmse=[.03],nominal_radial_rmse=[.04],
                nominal_speed_rmse=[.03],nominal_heading_rmse=[.03],episode_return=[10.],
                failed=[False],nominal_failed=[False],terminal_tracking_hold=[True],
                nominal_terminal_tracking_hold=[True],return_deadline_missed=[False],
                nominal_return_deadline_missed=[False],speed_tolerance_exceed_fraction=[.01],path_tolerance_exceed_fraction=[.01])


def test_candidate_needs_final_hold_deadline_nominal_and_safety_not_just_return():
    a=bank();assert rank_tracking_candidate(a,a)[0] is not None
    for key in ('failed','nominal_failed','return_deadline_missed','nominal_return_deadline_missed'):
        b=deepcopy(a);b[key]=[True];b['episode_return']=[1e6]
        assert rank_tracking_candidate(b,a)[0] is None
    for key in ('terminal_tracking_hold','nominal_terminal_tracking_hold'):
        b=deepcopy(a);b[key]=[False];assert rank_tracking_candidate(b,a)[0] is None
    b=deepcopy(a);b['nominal_speed_rmse']=[.3];assert rank_tracking_candidate(b,a)[0] is None
    b=deepcopy(a);b['path_tolerance_exceed_fraction']=[.2];assert rank_tracking_candidate(b,a)[0] is None
    for key in ('radial_rmse','speed_tolerance_exceed_fraction','path_tolerance_exceed_fraction'):
        b=deepcopy(a);b[key]=[np.nan];assert rank_tracking_candidate(b,a)[0] is None
