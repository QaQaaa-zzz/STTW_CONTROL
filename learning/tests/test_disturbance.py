from dataclasses import replace
import jax.numpy as jp
import numpy as np
from sttw_control.env import RecoveryEnv,TaskConfig


def test_steer_event_is_half_open_and_still_limited():
    env=RecoveryEnv(TaskConfig(disturbance_start=0.,disturbance_duration=.005,disturbance_steer_rate=100.))
    state=env.reset(0)
    assert env.has_disturbance
    assert bool(env.disturbance_active(0)) and not bool(env.disturbance_active(1))
    nxt=env.step(state,jp.zeros(2))
    assert float(nxt.actuator.previous[0])==env.config.actuator.steer_rate_limit
    reset=env.reset(0)
    np.testing.assert_array_equal(reset.actuator.previous,state.actuator.previous)


def test_vehicle_com_wrench_has_zero_moment_about_vehicle_com():
    env=RecoveryEnv(TaskConfig(disturbance_start=0.,disturbance_force=2.,disturbance_force_frame='heading_lateral',disturbance_force_point='vehicle_com'))
    state=env.reset(0)
    force=env.disturbance_wrench(state.data,0)
    offset=state.data.subtree_com[env.bundle.chassis]-state.data.xipos[env.bundle.chassis]
    np.testing.assert_allclose(force[3:]-np.cross(offset,force[:3]),0,atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(force[:3]),2.,atol=1e-6)
    np.testing.assert_allclose(env.disturbance_wrench(state.data,env.event_end),0)


def test_path_recovery_requires_excursion_and_final_hold():
    from sttw_control.disturbance import recovery_metrics
    from sttw_control.path import CircleConfig
    from dataclasses import asdict
    cfg=asdict(TaskConfig(circle=CircleConfig(),disturbance_start=.2,disturbance_duration=.1))
    t=np.arange(301)*.005;n=len(t)
    q=np.zeros((n,11));q[:,0]=3.;q[:,1]=3.;q[:,3]=np.sqrt(.5);q[:,6]=np.sqrt(.5)
    v=np.zeros((n,10));v[:,1]=2.
    nominal={'time':t,'qpos':q,'qvel':v,'measurement':np.zeros((n,7)),
             'motion_command':np.tile([0.,2.],(n,1)),'reference_roll':np.zeros(n),'terminated':np.zeros(n,dtype=bool)}
    no_event=recovery_metrics(nominal,nominal,cfg)
    assert not no_event['left_extra_radial_band'] and not no_event['recovered_after_excursion']
    disturbed={**nominal,'qpos':q.copy()}
    disturbed['qpos'][40:90,0]+=.1
    result=recovery_metrics(disturbed,nominal,cfg)
    assert result['recovered_after_excursion']
    late={**disturbed,'qpos':disturbed['qpos'].copy()}
    late['qpos'][220:230,0]+=.1
    assert recovery_metrics(late,nominal,cfg)['settled_joint_hold_completion_after_event_end_seconds'] is None
    disturbed['terminated']=np.ones(n,dtype=bool)
    assert not recovery_metrics(disturbed,nominal,cfg)['recovered_after_excursion']

    short={key:value[:20] for key,value in nominal.items()}
    short['terminated']=np.ones(20,dtype=bool)
    early=recovery_metrics(short,nominal,cfg)
    assert early['failed'] and not early['event_reached']
    assert early['post_event_radial_peak_m'] is None
