"""Opt-in real GPU contract test, independent of CPU fixture verification."""
import os
import numpy as np
import pytest

@pytest.mark.skipif(os.environ.get('STTW_TEST_GPU')!='1',reason='explicit GPU engineering check')
def test_gpu_reset_step_vmap():
    import jax
    import jax.numpy as jp
    from sttw_control.env import RecoveryEnv,TaskConfig
    assert jax.default_backend()=='gpu'
    env=RecoveryEnv(TaskConfig(horizon_seconds=.02),backend='mjx')
    reset=jax.jit(jax.vmap(env.reset))
    step=jax.jit(jax.vmap(env.step))
    state=reset(jax.random.split(jax.random.PRNGKey(4),2))
    state=step(state,jp.zeros((2,2)))
    np.testing.assert_allclose(state.data.time,.005,atol=1e-7)
    assert state.obs.shape==(2,16)
    assert np.isfinite(state.data.qpos).all()
    assert np.isfinite(state.obs).all()
