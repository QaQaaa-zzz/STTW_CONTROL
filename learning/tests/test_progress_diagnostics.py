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


def test_smooth_bend_geometry_and_zero_straight_steer():
    import jax.numpy as jp
    from sttw_control.path import BendConfig,bend_table,bend_command,bend_features
    c=BendConfig();table=bend_table(c)
    assert table[0,4]==0 and abs(table[-1,4])<1e-10
    np.testing.assert_allclose(table[-1,3],c.peak_curvature*c.bend_length/2)
    assert abs(float(bend_command(jp.array([1.,0.,0.]),c,jp.asarray(table),.408,.436)))<1e-6
    features,_=bend_features(jp.array([1.,.1,0.]),jp.asarray(table))
    np.testing.assert_allclose(features[0],-.1,atol=1e-6)


def test_random_rear_load_is_separate_from_force_and_steering():
    import jax
    from sttw_control.events import RandomEvents,sample_event
    c=RandomEvents(rear_probability=1.,nominal_probability=0.)
    event=np.asarray(sample_event(jax.random.PRNGKey(5),c,.005))
    assert event[2]==0 and event[3]==0 and c.rear_min<=event[5]<=c.rear_max


def test_dynamic_roll_and_rear_load_cpu_contract():
    import jax
    import jax.numpy as jp
    from sttw_control.env import RecoveryEnv,TaskConfig
    from sttw_control.path import BendConfig
    from sttw_control.observation import ObservationConfig
    c=TaskConfig(bend=BendConfig(),observation=ObservationConfig(include_path=True),disturbance_start=0.,disturbance_duration=.01,disturbance_rear_torque=.24)
    env=RecoveryEnv(c,backend='cpu');s=env.reset(jax.random.PRNGKey(1))
    assert abs(float(s.reference))<1e-6
    n=env.step(s,jp.zeros(2));assert n.data.qfrc_applied[env.bundle.rear_dof]>.23
    assert n.data.qvel[env.bundle.rear_dof]<0  # positive external torque opposes forward axle rotation
    for _ in range(3):n=env.step(n,jp.zeros(2))
    assert n.data.qfrc_applied[env.bundle.rear_dof]==0
