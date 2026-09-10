import json
import numpy as np
import pytest
from sttw_control.env import RecoveryEnv,TaskConfig
from sttw_control.evaluation import evaluate


def test_trace_captures_initial_and_terminal_and_refuses_overwrite(tmp_path):
    env=RecoveryEnv(TaskConfig(horizon_seconds=.02))
    result=evaluate(env,tmp_path/'run',seed=3)
    assert result['transitions']==4
    assert not result['task_recovery_success']
    assert not result['recovery_eligible']
    trace=np.load(tmp_path/'run/trace.npz')
    assert len(trace['qpos'])==5
    assert trace['truncated'][-1]
    assert len(trace['command'])==5
    assert json.loads((tmp_path/'run/summary.json').read_text())['controller']=='baseline'
    with pytest.raises(FileExistsError): evaluate(env,tmp_path/'run',seed=3)


def test_history_length_does_not_change_physical_roll_summary(tmp_path):
    from sttw_control.observation import ObservationConfig
    cfg=TaskConfig(horizon_seconds=.025,initial_roll_range=.05,observation=ObservationConfig(history_steps=4))
    result=evaluate(RecoveryEnv(cfg),tmp_path/'run',seed=3)
    trace=np.load(tmp_path/'run/trace.npz')
    expected=float(np.abs(trace['measurement'][:,0]).max())
    assert result['roll_abs_max_rad']==expected


def test_command_limit_statistics_exclude_initial_state():
    from sttw_control.evaluation import command_limit_metrics
    from sttw_control.actuator import ActuatorConfig
    c=ActuatorConfig()
    result=command_limit_metrics(np.array([[3.,60.],[3.,20.],[0.,60.],[0.,20.]]),c)
    assert result['steer_rate_limit_fraction']==pytest.approx(1/3)
    assert result['rear_rate_limit_fraction']==pytest.approx(1/3)
    assert result['either_rate_limit_fraction']==pytest.approx(2/3)


def test_headroom_distinguishes_joint_position_and_rate_constraints():
    from sttw_control.evaluation import headroom_metrics
    from sttw_control.actuator import ActuatorConfig
    c=ActuatorConfig()
    result=headroom_metrics(np.array([[0.,20.],[2.9,59.9],[0.,20.]]),np.array([0.,0.,.8]),c)
    assert result['residual_box_restricted_fraction']==pytest.approx(2/3)
    assert result['baseline_outside_command_bounds_fraction']==0.
    assert result['minimum_positive_steer_margin_rad_s']==0.


def test_actuator_trace_aligns_request_before_limits_and_terminal_response(tmp_path):
    env=RecoveryEnv(TaskConfig(horizon_seconds=.01,disturbance_start=0.,disturbance_duration=.005,disturbance_steer_rate=100.))
    evaluate(env,tmp_path/'run')
    t=np.load(tmp_path/'run/trace.npz')
    assert not t['actuator_diagnostic_valid'][0]
    assert t['actuator_diagnostic_valid'][1:].all()
    assert t['prelimit_command'][1,0]>90.
    assert t['command'][1,0]==3.
    assert abs(t['prelimit_command'][2,0])<90.
    assert t['actuator_force'].shape==(3,env.model.nu)
    np.testing.assert_allclose(t['request_time'][1:],t['time'][:-1])
    assert np.isfinite(t['actuator_force']).all()


def test_figure_eight_trace_metrics_media_and_pairing(tmp_path):
    from dataclasses import asdict,replace
    from sttw_control.env import load_config
    from sttw_control.media import plot_states
    from sttw_control.disturbance import recovery_metrics
    c=replace(load_config('learning/configs/figure_eight_tracking.json'),horizon_seconds=.02,disturbance_start=0.,disturbance_duration=.005)
    result=evaluate(RecoveryEnv(c),tmp_path/'run')
    assert 'path_tracking' in result and 'circle_tracking' not in result
    tr=dict(np.load(tmp_path/'run/trace.npz'))
    r=recovery_metrics(tr,tr,asdict(c))
    assert r['error_coordinate'].startswith('figure_eight')
    assert r['post_event_extra_radial_peak_m']==0.
    out=tmp_path/'plots';out.mkdir()
    plot_states(tr,asdict(c),out,'Baseline')
    assert (out/'trajectory.png').exists()
