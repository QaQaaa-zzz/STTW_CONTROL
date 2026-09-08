import numpy as np
import pytest
import jax
import jax.numpy as jnp
from sttw_control.actuator import ActuatorConfig, initial_actuator, apply_residual
from sttw_control.network import ResidualActor, save_policy, load_policy
from sttw_control.observation import ObservationConfig, make_frame, initial_history, advance_history

def test_limits_delay_endpoint_and_invalid_action():
    cfg=ActuatorConfig(delay_steps=1,steer_acceleration=1000.,rear_acceleration=1000.)
    state=initial_actuator(cfg, rear_command=10.)
    state,out=apply_residual(state,jnp.array([2.,10.]),jnp.array([1.,1.]),0.,cfg)
    np.testing.assert_allclose(out,[0.,10.])
    state,out=apply_residual(state,jnp.array([2.,10.]),jnp.zeros(2),0.,cfg)
    np.testing.assert_allclose(out,[3.,15.])
    _,out=apply_residual(state,jnp.array([2.,10.]),jnp.zeros(2),.8,cfg)
    assert out[0]<=0
    _,out=apply_residual(initial_actuator(ActuatorConfig()),jnp.array([1.,5.]),jnp.array([jnp.nan,0.]),0.,ActuatorConfig())
    assert np.isfinite(out).all()

def test_zero_residual_matches_baseline():
    cfg=ActuatorConfig()
    _,out=apply_residual(initial_actuator(cfg),jnp.array([1.25,12.]),jnp.zeros(2),0.,cfg)
    np.testing.assert_allclose(out,[1.25,12.])

def test_reference_inputs_and_history():
    cfg=ObservationConfig(history_steps=2)
    # state: roll, roll_rate, steer, steer_rate, yaw_rate_body, rear_rate, front_rate
    frame=make_frame(jnp.array([.2,.3,.1,.4,.5,20.,19.]),jnp.array([.1,2.]),
                     .15,1.,jnp.array([.7,18.]),.6)
    np.testing.assert_allclose(frame[:4],[.05,.2,.3,2.],atol=1e-7)
    history,obs=advance_history(initial_history(cfg),frame,cfg)
    assert obs.shape==(32,)
    np.testing.assert_allclose(obs[-2:],[0,1])
    _,obs=advance_history(history,frame*2,cfg)
    np.testing.assert_allclose(obs[-2:],[1,1])

def test_network_gradient_and_checkpoint_contract(tmp_path):
    actor=ResidualActor()
    x=jnp.ones(16)
    params=actor.init(jax.random.PRNGKey(5),x)
    output=actor.apply(params,x)
    assert output.shape==(2,) and np.all(np.abs(output)<=1)
    grad=jax.grad(lambda z: actor.apply(params,z).sum())(x)
    assert np.isfinite(grad).all() and np.linalg.norm(grad)>0
    save_policy(tmp_path/'policy',params,np.zeros(16),np.ones(16),{'model_sha256':'abc','history_steps':1})
    policy=load_policy(tmp_path/'policy',expected={'model_sha256':'abc','history_steps':1})
    np.testing.assert_allclose(policy(x),output,atol=5e-7,rtol=1e-6)
    with pytest.raises(ValueError):
        load_policy(tmp_path/'policy',expected={'model_sha256':'other','history_steps':1})


def test_identity_includes_meshes_not_only_xml():
    from sttw_control.network import make_policy_identity
    a=make_policy_identity({'source_xml_sha256':'same','assets':{'wheel':'one'}},{},1)
    b=make_policy_identity({'source_xml_sha256':'same','assets':{'wheel':'two'}},{},1)
    assert a!=b
