"""Zero-base control removes the baseline command, not actuator safety limits."""
from dataclasses import replace
import numpy as np
import pytest
from sttw_control.actuator import ActuatorConfig,effective_base,residual_target


def test_zero_base_drops_both_channels_and_retains_rl_actions():
    c=ActuatorConfig(base_output_scale=0.,steer_residual_scale=1.5,rear_residual_scale=10.)
    for raw in ([2.,30.],[-3.,-60.]):
        base=effective_base(np.array(raw),c)
        np.testing.assert_array_equal(base,[0.,0.])
        np.testing.assert_allclose(residual_target(base,np.array([.5,-.5]),c),[.75,-5.])
        np.testing.assert_array_equal(residual_target(base,np.zeros(2),c),[0.,0.])
    np.testing.assert_array_equal(effective_base(np.array([2.,30.]),c,scale=1.),[2.,30.])


@pytest.mark.parametrize('scale',[-.1,float('nan'),float('inf')])
def test_invalid_base_scales_still_rejected(scale):
    with pytest.raises(ValueError):ActuatorConfig(base_output_scale=scale)


def test_environment_zero_base_uses_same_preparation_and_no_hidden_feedforward():
    import jax
    from sttw_control.env import RecoveryEnv,load_config
    cfg=load_config('learning/configs/asymmetric_priority_rho34.json')
    cfg=replace(cfg,preparation_seconds=0.,actuator=replace(cfg.actuator,base_output_scale=0.))
    env=RecoveryEnv(cfg,backend='cpu')
    state=env.reset(jax.random.PRNGKey(1))
    base,action=env.prepare_action(state,np.array([.2,-.3]))
    np.testing.assert_array_equal(base,[0.,0.])
    np.testing.assert_allclose(residual_target(base,action,cfg.actuator),[.3,-3.],atol=1e-6)
    assert cfg.preparation_base_output_scale==1.


def test_full_rl_config_has_same_task_and_preparation_but_full_command_authority():
    import json
    from pathlib import Path
    from sttw_control.env import load_config
    source=json.loads(Path('learning/configs/asymmetric_priority_rho34.json').read_text())
    direct=json.loads(Path('learning/configs/asymmetric_direct_rl.json').read_text())
    expected=dict(source);expected['actuator']=dict(source['actuator'],base_output_scale=0.,steer_residual_scale=3.,rear_residual_scale=60.)
    assert direct==expected
    c=load_config('learning/configs/asymmetric_direct_rl.json').actuator
    np.testing.assert_allclose(residual_target(effective_base(np.array([2.,28.]),c),np.array([1.,-1.]),c),[c.steer_rate_limit,-c.rear_rate_limit])


def test_prepared_physics_matches_and_task_has_zero_base():
    import jax
    from sttw_control.env import RecoveryEnv,load_config
    reference=RecoveryEnv(load_config('learning/configs/asymmetric_priority_rho34.json'),backend='cpu')
    direct=RecoveryEnv(load_config('learning/configs/asymmetric_direct_rl.json'),backend='cpu')
    a=reference.reset(jax.random.PRNGKey(8));b=direct.reset(jax.random.PRNGKey(8))
    np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=1e-9)
    np.testing.assert_allclose(a.data.qvel,b.data.qvel,atol=1e-9)
    assert float(b.active_base_output_scale)==0.
    assert not bool(b.preparation_failed)
    for action in [np.zeros(2),np.array([.2,.4])]:
        base,mapped=direct.prepare_action(b,action)
        np.testing.assert_array_equal(base,[0.,0.])
        np.testing.assert_allclose(residual_target(base,mapped,direct.config.actuator),np.array([3.,60.])*action,atol=2e-6)
