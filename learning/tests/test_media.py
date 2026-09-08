import numpy as np
import pytest
from sttw_control.media import frame_indices, validate_trace


def test_video_samples_include_terminal_without_continuing_simulation():
    times=np.arange(1601)*.005
    ids=frame_indices(times,30)
    assert len(ids)==241 and ids[0]==0 and ids[-1]==1600
    assert np.all(np.diff(ids)>0)
    failed=times[:138]
    ids=frame_indices(failed,30)
    assert ids[-1]==137


def test_invalid_trace_is_not_renderable():
    with pytest.raises(ValueError): frame_indices(np.array([0.,0.]),30)
    with pytest.raises(ValueError): frame_indices(np.array([0.,1.]),0)
    with pytest.raises(ValueError): validate_trace({'time':np.array([0.,1.]),'qpos':np.zeros((1,11)),'qvel':np.zeros((2,10))})


def test_circle_environment_captures_localization_driven_target(tmp_path):
    from sttw_control.env import RecoveryEnv,TaskConfig
    from sttw_control.path import CircleConfig
    from sttw_control.evaluation import evaluate
    env=RecoveryEnv(TaskConfig(horizon_seconds=.02,circle=CircleConfig()))
    evaluate(env,tmp_path/'run')
    trace=np.load(tmp_path/'run/trace.npz')
    assert trace['motion_command'].shape==(5,2)
    assert trace['pose'].shape==(5,3)
    assert np.isfinite(trace['reference_roll']).all()
    assert trace['motion_command'][0,0]>0
