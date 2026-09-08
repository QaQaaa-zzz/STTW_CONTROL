from dataclasses import replace
import numpy as np
import pytest
from sttw_control.env import RecoveryEnv, TaskConfig
from sttw_control.model import load_model


def test_portable_model_preserves_physics():
    bundle=load_model()
    assert bundle.model.opt.timestep==.0002
    assert bundle.model.nu==4
    assert bundle.model.nplugin==0
    np.testing.assert_allclose(bundle.model.body_mass.sum(),5.43421)
    np.testing.assert_allclose(bundle.model.actuator_ctrlrange[2],[-3,3])
    assert bundle.model.opt.disableactuator==2


def test_exported_model_loads_without_plugin_or_relative_asset_cwd(tmp_path):
    import mujoco
    from sttw_control.model import export_model
    output=tmp_path/'space in directory'/'bike.xml'
    export_model(output)
    model=mujoco.MjModel.from_xml_path(str(output))
    assert model.nplugin==0 and model.nu==4
    np.testing.assert_allclose(model.body_mass.sum(),5.43421)
    assert model.opt.disableactuator==2


def test_host_timing_reset_and_terminal_absorption():
    env=RecoveryEnv(TaskConfig(horizon_seconds=.03,disturbance_start=.01,disturbance_duration=.005,disturbance_force=1.))
    state=env.reset(1)
    assert state.obs.shape==(16,)
    for _ in range(6):
        state=env.step(state,np.zeros(2))
    assert bool(state.done) and bool(state.truncated)
    assert state.data.time==pytest.approx(.03,abs=1e-10)
    qpos=np.asarray(state.data.qpos).copy()
    state=env.step(state,np.ones(2))
    np.testing.assert_array_equal(state.data.qpos,qpos)
    reset=env.reset(1)
    assert int(reset.tick)==0 and not bool(reset.done)
    assert not bool(reset.balance_recovered)
    np.testing.assert_array_equal(reset.actuator.previous,[0,20])


def test_invalid_action_is_a_failure_not_a_success():
    env=RecoveryEnv(TaskConfig())
    state=env.step(env.reset(0),np.array([np.nan,0]))
    assert bool(state.terminated) and int(state.end_code)==3
    assert not bool(state.task_recovered)


def test_recovery_requires_event_and_hold():
    from sttw_control.recovery import RecoveryConfig, initial_recovery, update_recovery
    c=RecoveryConfig(dt=.005,hold_seconds=.01,roll_tolerance=.1,roll_rate_tolerance=.2,speed_tolerance=.2,steer_tolerance=.1)
    state=initial_recovery()
    state=update_recovery(state,0.,0.,0.,0.,False,False,c)
    assert not bool(state.balance_recovered)
    state=update_recovery(state,0.,0.,.5,0.,True,False,c)
    assert not bool(state.balance_recovered)
    state=update_recovery(state,0.,0.,.5,0.,True,False,c)
    assert bool(state.balance_recovered) and not bool(state.task_recovered)
    state=update_recovery(state,0.,0.,0.,0.,True,False,c)
    state=update_recovery(state,0.,0.,0.,0.,True,False,c)
    assert bool(state.task_recovered)


def test_reset_with_roll_bias_is_deterministic_and_distinct():
    env=RecoveryEnv(TaskConfig(initial_roll_range=.05))
    a=env.reset(3)
    b=env.reset(3)
    c=env.reset(4)
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    assert not np.array_equal(a.data.qpos,c.data.qpos)


@pytest.mark.parametrize('changes',[
    {'horizon_seconds':float('nan')},{'roll_failure':0.},
    {'disturbance_force':float('inf')},{'steer_frequency':-1.}])
def test_reject_invalid_task_config(changes):
    with pytest.raises(ValueError): TaskConfig(**changes)


def test_recovery_rejects_zero_hold():
    from sttw_control.recovery import RecoveryConfig
    with pytest.raises(ValueError): RecoveryConfig(hold_seconds=0.)


def test_model_cannot_silently_clip_larger_configured_actuator_limits():
    from sttw_control.actuator import ActuatorConfig
    with pytest.raises(ValueError): RecoveryEnv(TaskConfig(actuator=ActuatorConfig(steer_rate_limit=4.)))


def test_disturbance_window_cannot_fall_between_ticks():
    with pytest.raises(ValueError): TaskConfig(disturbance_start=.001,disturbance_duration=.001,disturbance_force=1.)


def test_nonfinite_controller_state_is_a_terminal_failure():
    import jax.numpy as jp
    env=RecoveryEnv(TaskConfig())
    state=env.reset(0)
    state=state.replace(controller=state.controller.replace(eso=jp.full(4,jp.inf)))
    state=env.step(state,np.zeros(2))
    assert bool(state.terminated) and int(state.end_code)==3
    assert np.isfinite(state.obs).all()
