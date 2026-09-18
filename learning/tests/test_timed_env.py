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


def test_closed_loop_preparation_preserves_eso_actuator_and_history_at_task_zero():
    c=task()
    c=replace(c,speed_reference=2.3,eso_start=.005,preparation_seconds=.02,
              timed_reference=replace(c.timed_reference,training_mix=True,fixed_scenario='core_left'))
    env=RecoveryEnv(c);state=env.reset(71)
    assert state.tick==0
    assert state.eso_enabled and not state.preparation_failed
    assert float(state.data.time)==pytest.approx(.02)
    assert np.count_nonzero(np.asarray(state.history.mask))>1
    assert np.linalg.norm(np.asarray(state.controller.eso))>0
    np.testing.assert_allclose(state.raw_reference_request,[2.3,0.])
    np.testing.assert_allclose(state.reference_pose,state.pose)


def test_prepared_state_is_reused_without_recomputing_or_mutating_physics():
    c=task()
    c=replace(c,speed_reference=2.3,eso_start=.005,preparation_seconds=.02,
              timed_reference=replace(c.timed_reference,training_mix=True))
    env=RecoveryEnv(c);prepared=env.prepare_state(71)
    a=env.reset_from_prepared(prepared,81);b=env.reset_from_prepared(prepared,82)
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
    np.testing.assert_array_equal(a.controller.eso,b.controller.eso)
    assert a.tick==b.tick==0 and a.data.time==b.data.time==pytest.approx(.02)
    assert not np.array_equal(a.command_schedule,b.command_schedule)


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
    rebuilt=make_policy_identity(env.bundle.identity,asdict(cfg),cfg.observation.history_steps)
    assert rebuilt['config_sha256']==decl['policy_identity']['config_sha256']
    assert rebuilt==make_policy_identity(env.bundle.identity,decl['task'],cfg.observation.history_steps)
    # Model identity remains binding; current user-edited physics is not the frozen old XML.


def test_geometry_state_and_control_use_projected_path_not_clock_yaw():
    from sttw_control.timed_reference import errors
    c=task()
    assert 'geometric' in c.tracking.__dataclass_fields__, 'explicit geometric mode required'
    c=replace(c,tracking=replace(c.tracking,geometric=True))
    env=RecoveryEnv(c);s=env.reset(7)
    assert s.geometric_table.shape==(env.horizon+1,6)
    # Curvature zero: a nonzero clock yaw must not influence geometric feedforward.
    command=env.control_reference(s.pose,s.reference_pose,np.array([2.1,.5]),np.zeros(3))
    np.testing.assert_allclose(command,[0,2.1],atol=1e-6)
    nxt=env.step(s,np.zeros(2))
    assert float(nxt.path_progress)>=0
    assert float(nxt.path_progress)<=float(nxt.geometric_table[1,0])+1e-6
    np.testing.assert_allclose(nxt.geometric_table[:1],s.geometric_table[:1])
    np.testing.assert_allclose(nxt.geometric_table[1,1:4],nxt.reference_pose)
    assert nxt.obs.shape==(310,)


def test_geometric_environment_only_constructs_committed_reference_prefix():
    import jax.numpy as jp
    from sttw_control.timed_reference import committed_geometry_table
    c=task();c=replace(c,tracking=replace(c.tracking,geometric=True))
    env=RecoveryEnv(c);s=env.reset(19)
    assert np.count_nonzero(np.asarray(s.geometric_table[1:]))==0
    expected=committed_geometry_table(s.pose,s.reference_command,s.command_schedule,c.controller.dt,c.timed_reference,5)
    for _ in range(5):s=env.step(s,np.zeros(2))
    np.testing.assert_allclose(s.geometric_table[:6],expected,atol=2e-7)
    assert np.count_nonzero(np.asarray(s.geometric_table[6:]))==0
