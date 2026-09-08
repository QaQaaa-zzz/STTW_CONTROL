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
