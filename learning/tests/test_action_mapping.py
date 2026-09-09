from dataclasses import asdict,replace
import jax
import jax.numpy as jp
import numpy as np
import pytest
from sttw_control.action_mapping import MappingConfig,response_matrix,map_action
from sttw_control.actuator import ActuatorConfig
from sttw_control.controller import ControllerConfig


def test_inverse_cancels_speed_induced_lateral_change():
    c=MappingConfig();a=ActuatorConfig();ctrl=ControllerConfig()
    J=response_matrix(2.,.1,c,a,ctrl)
    action=jp.array([0.,.02])
    mapped=map_action(action,2.,.1,c,a,ctrl)
    delta=mapped*jp.array([a.steer_residual_scale,a.rear_residual_scale])
    np.testing.assert_allclose(J@delta,[0.,.02*c.speed_proxy_scale],atol=1e-7)
    assert float(mapped[0])<0 and float(mapped[1])>0
    uncoupled=map_action(action,2.,.1,replace(c,coupled=False),a,ctrl)
    assert float(uncoupled[0])==0
    assert float((J@(uncoupled*jp.array([1.,5.])))[0])>0


def test_zero_bounds_low_speed_and_jax_gradient():
    c=MappingConfig();a=ActuatorConfig();ctrl=ControllerConfig()
    f=lambda x:map_action(x,2.,.1,c,a,ctrl)
    np.testing.assert_array_equal(f(jp.zeros(2)),[0,0])
    assert np.max(np.abs(np.asarray(f(jp.array([100.,-100.])))))<=1
    np.testing.assert_array_equal(map_action(jp.ones(2),.1,.1,c,a,ctrl),[0,0])
    np.testing.assert_array_equal(f(jp.array([float('nan'),0.])),[0,0])
    assert np.isfinite(jax.jacfwd(f)(jp.array([.01,.01]))).all()


def test_servo_response_and_delay_affect_mapping():
    c=MappingConfig();a=ActuatorConfig();ctrl=ControllerConfig()
    J=response_matrix(2.,.1,c,a,ctrl)
    delayed=response_matrix(2.,.1,c,replace(a,delay_steps=5),ctrl)
    assert np.all(np.diag(delayed)<np.diag(J))
    with pytest.raises(ValueError):MappingConfig(horizon=-1.)


def test_zero_mapping_matches_direct_cpu_and_checkpoints_bind_semantics(tmp_path):
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.network import make_policy_identity,load_policy,save_policy,ResidualActor
    from sttw_control.training import normalization
    cfg=load_config('learning/configs/disturbance_learning.json')
    direct=RecoveryEnv(cfg);coupled=RecoveryEnv(replace(cfg,action_mapping=MappingConfig()))
    a,b=direct.reset(21),coupled.reset(21)
    for _ in range(5):
        a,b=direct.step(a,jp.zeros(2)),coupled.step(b,jp.zeros(2))
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        np.testing.assert_array_equal(a.actuator.previous,b.actuator.previous)
    old=asdict(cfg);old.pop('action_mapping')
    identity=make_policy_identity(direct.bundle.identity,asdict(cfg),1)
    assert identity==make_policy_identity(direct.bundle.identity,old,1)
    mapped_identity=make_policy_identity(coupled.bundle.identity,asdict(coupled.config),1)
    assert mapped_identity!=identity
    actor=ResidualActor();params=actor.init(jax.random.PRNGKey(1),b.obs)
    mean,std=normalization(coupled.config)
    save_policy(tmp_path/'mapped',params,mean,std,mapped_identity)
    load_policy(tmp_path/'mapped',expected=mapped_identity)
    with pytest.raises(ValueError,match='identity'):load_policy(tmp_path/'mapped',expected=identity)


def test_mapping_preserves_shared_command_bounds_and_rejects_unmodeled_slew():
    from sttw_control.env import TaskConfig
    from sttw_control.actuator import initial_actuator,apply_residual
    a=ActuatorConfig();c=MappingConfig();ctrl=ControllerConfig()
    mapped=map_action(jp.ones(2),2.,a.steer_limit,c,a,ctrl)
    _,command=apply_residual(initial_actuator(a),jp.array([3.,59.]),mapped,a.steer_limit,a)
    assert float(command[0])<=0 and abs(float(command[1]))<=a.rear_rate_limit
    with pytest.raises(ValueError,match='slew'):TaskConfig(action_mapping=c,actuator=replace(a,rear_acceleration=1.))


def test_compensation_uses_achievable_rear_residual_after_saturation():
    c=MappingConfig(speed_proxy_scale=10.);a=ActuatorConfig();ctrl=ControllerConfig()
    mapped=map_action(jp.array([0.,1.]),2.,.1,c,a,ctrl)
    assert float(mapped[1])==1.
    effect=response_matrix(2.,.1,c,a,ctrl)@(mapped*jp.array([1.,5.]))
    np.testing.assert_allclose(effect[0],0,atol=1e-6)


def test_steering_only_config_removes_drive_residual():
    from pathlib import Path
    from sttw_control.env import load_config
    from sttw_control.actuator import initial_actuator, apply_residual
    c=load_config(Path(__file__).parents[1]/'configs/disturbance_steering_only_learning.json')
    assert c.action_mapping is None
    _,command=apply_residual(initial_actuator(c.actuator),jp.array([0.,20.]),jp.ones(2),0.,c.actuator)
    np.testing.assert_allclose(command,[c.actuator.steer_residual_scale,20.])
