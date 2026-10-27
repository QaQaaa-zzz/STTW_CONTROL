import numpy as np
from sttw_control.preference_history_task_review import shadow_cross_track, paired_complete_metrics


def test_shadow_starts_at_recorded_progress_and_does_not_jump_to_early_nearest_branch():
    # A folded route: the current point is globally nearest to the obsolete first leg.
    path = dict(s=np.arange(7.), xy=np.array([[0,0],[1,0],[2,0],[2,1],[1,1],[0,1],[-1,1.]]),
                heading=np.array([0,0,0,np.pi,np.pi,np.pi,np.pi]))
    cfg={'path': dict(projection_backward_m=.05, projection_forward_floor_m=.3,
                      projection_forward_motion_multiplier=2.,projection_forward_padding_m=.1)}
    error, progress=shadow_cross_track([[.1,.1]],path,cfg,initial_xy=[.1,.1],initial_progress=4.9)
    assert progress[0] >= 4.9
    assert abs(error[0]-.9) < 1e-8


def test_failure_short_prefix_does_not_win_cost_ranking_or_disappear():
    rows=[]
    for role,label,cost in [('future1','PASS',3.),('zero_latent','CONTACT',.1)]:
        rows.append(dict(state_id='s',force_N=32.,role=role,physical_label=label,speed_mae=cost))
    result=paired_complete_metrics(rows,'future1','zero_latent',['speed_mae'])
    assert result['common_complete']==0 and len(result['excluded'])==1
    assert result['mean_differences']['speed_mae'] is None
