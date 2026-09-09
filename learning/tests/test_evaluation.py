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
