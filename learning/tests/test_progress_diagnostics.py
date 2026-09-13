import numpy as np
from sttw_control.progress_diagnostics import circle_progress


def test_progress_ignores_extra_radius_and_unwraps_both_directions():
    for direction in (-1,1):
        t=np.linspace(0,20,401);a=direction*t/2
        trace=dict(time=t,pose=np.column_stack((5*np.cos(a),5*np.sin(a),a)),motion_command=np.tile([0.,2.],(len(t),1)))
        x=circle_progress(trace,dict(center_x=0,center_y=0,radius=4,direction=direction))
        np.testing.assert_allclose(x['progress_m'],2*t,atol=1e-12)
        np.testing.assert_allclose(x['schedule_lag_s'],0,atol=1e-12)
        np.testing.assert_allclose(x['radial_error_m'],1,atol=1e-12)
