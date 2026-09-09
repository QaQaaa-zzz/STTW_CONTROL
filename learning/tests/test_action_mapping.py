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


def test_box_allocation_reallocates_when_steering_is_unavailable():
    from sttw_control.action_mapping import box_allocate
    J=jp.array([[1.,1.],[0.,1.]])
    request=jp.array([1.,0.]);lo=jp.array([0.,-1.]);hi=jp.array([0.,1.])
    u=box_allocate(J,request,lo,hi,jp.ones(2),1e-6)
    np.testing.assert_allclose(u,[0.,.5],atol=1e-5)
    assert np.sum(np.asarray(J@u-request)**2)<1.


def test_authority_mapping_obeys_remaining_command_margin_and_zero():
    c=MappingConfig(authority_aware=True);a=ActuatorConfig();ctrl=ControllerConfig()
    base=jp.array([2.9,59.9])
    f=lambda action:map_action(action,2.,.1,c,a,ctrl,base=base)
    np.testing.assert_array_equal(f(jp.zeros(2)),[0.,0.])
    command=base+f(jp.ones(2))*jp.array([1.,5.])
    assert float(command[0])<=3.000001 and float(command[1])<=60.000001
    assert np.isfinite(jax.jacfwd(f)(jp.array([.01,.01]))).all()


def test_authority_zero_cpu_regression_and_legacy_mapping_identity():
    from sttw_control.env import RecoveryEnv,load_config,TaskConfig
    from sttw_control.network import make_policy_identity
    c=load_config('learning/configs/disturbance_authority_learning.json')
    a=RecoveryEnv(c);b=RecoveryEnv(replace(c,action_mapping=None))
    x,y=a.reset(4),b.reset(4)
    for _ in range(5):
        x,y=a.step(x,jp.zeros(2)),b.step(y,jp.zeros(2))
        np.testing.assert_array_equal(x.data.qpos,y.data.qpos)
    legacy=asdict(replace(c,action_mapping=MappingConfig()))
    old={**legacy,'action_mapping':dict(legacy['action_mapping'])}
    for key in ('authority_aware','lateral_weight','speed_weight','regularization'):old['action_mapping'].pop(key)
    assert make_policy_identity({},old,1)==make_policy_identity({},legacy,1)
    with pytest.raises(ValueError,match='delayed'):
        TaskConfig(action_mapping=c.action_mapping,actuator=ActuatorConfig(delay_steps=1))


def test_box_solver_matches_dense_search_and_infeasible_base_fallback():
    from sttw_control.action_mapping import box_allocate
    rng=np.random.default_rng(15)
    for _ in range(8):
        J=rng.normal(size=(2,2));r=rng.normal(size=2)
        lo=np.array([-.2,-.8]);hi=np.array([.5,.3]);w=np.array([2.,1.])
        u=np.asarray(box_allocate(jp.asarray(J),jp.asarray(r),jp.asarray(lo),jp.asarray(hi),jp.asarray(w),1e-6))
        grid=np.stack(np.meshgrid(np.linspace(lo[0],hi[0],101),np.linspace(lo[1],hi[1],101)),axis=-1).reshape(-1,2)
        cost=lambda v:np.sum(w*(v@J.T-r)**2,axis=-1)+1e-6*np.sum(v*v,axis=-1)
        assert cost(u)<=cost(grid).min()+1e-5
    c=MappingConfig(authority_aware=True)
    np.testing.assert_array_equal(map_action(jp.ones(2),2.,.1,c,ActuatorConfig(),ControllerConfig(),base=jp.array([4.,20.])),[0.,0.])
