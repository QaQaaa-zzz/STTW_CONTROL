"""Time reference belongs to the command clock, not the vehicle projection."""
from dataclasses import replace,asdict
import numpy as np
import pytest
from sttw_control.env import RecoveryEnv,config_from_dict


def task():
    import json
    from pathlib import Path
    c=json.loads(Path('learning/configs/path_reference_recovery.json').read_text())
    c.pop('reference_paths');c.pop('bend')
    c['timed_reference']={}
    c['tracking']['timed']=True
    c['observation']['include_timed']=True
    return config_from_dict(c)


def test_timed_reset_resamples_schedule_and_exposes_clock_errors():
    env=RecoveryEnv(task());a=env.reset(71);b=env.reset(72)
    assert a.obs.shape==(310,)
    assert not np.array_equal(a.command_schedule,b.command_schedule)
    np.testing.assert_allclose(a.reference_pose,a.pose)
    np.testing.assert_array_equal(a.reference_command,[np.float32(2.1),0.])
    assert len(a.tracking_components)>14


def test_time_reference_advances_independently_of_actions_and_resets():
    env=RecoveryEnv(task());a=env.reset(71);b=env.reset(71)
    for _ in range(5):
        a=env.step(a,np.zeros(2));b=env.step(b,np.array([.2,-.1]))
        np.testing.assert_allclose(a.reference_pose,b.reference_pose,atol=1e-7)
        np.testing.assert_allclose(a.reference_command,b.reference_command,atol=1e-7)
    assert not np.allclose(a.pose,b.pose,atol=1e-7)
    reset=env.reset(71)
    np.testing.assert_array_equal(reset.tracking_state.previous_action,[0.,0.])
    np.testing.assert_allclose(reset.reference_pose,reset.pose)
    assert reset.tick==0


def test_legacy_checkpoint_identity_stays_compatible():
    import json
    from pathlib import Path
    from sttw_control.network import make_policy_identity
    from sttw_control.env import config_from_dict
    path=Path('runs/path_projection_rsl_4096_20260915/training')
    if not path.exists():pytest.skip('local historical checkpoint not available')
    decl=json.loads((path/'declaration.json').read_text());cfg=config_from_dict(decl['task'])
    env=RecoveryEnv(cfg)
    assert make_policy_identity(env.bundle.identity,asdict(cfg),cfg.observation.history_steps)==decl['policy_identity']
