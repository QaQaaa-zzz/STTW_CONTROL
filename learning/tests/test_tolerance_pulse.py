import numpy as np
import jax.numpy as j
from sttw_control.teleop_env import TeleopEnv

def test_zero_pulse_preserves_cpu_transition_and_state():
    env=TeleopEnv(backend='cpu');s=env.initial();q=np.array(s.data.qpos)
    a,la=env._step(s,s.raw,True)
    b,lb=env._step(s,s.raw,True,force_pulse=j.array([0.,.01,0.,0.,0.]))
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
    np.testing.assert_array_equal(s.data.qpos,q)
    np.testing.assert_array_equal(lb['applied_force_impulse'],[0.,0.,0.])

def test_partial_substep_pulse_impulse_and_physical_effect():
    env=TeleopEnv(backend='cpu');s=env.initial()
    a,_=env._step(s,s.raw,True)
    b,log=env._step(s,s.raw,True,force_pulse=j.array([.0001,.0049,0.,2.,0.]))
    np.testing.assert_allclose(log['applied_force_impulse'],[0.,.0096,0.],rtol=2e-5,atol=1e-9)
    assert np.linalg.norm(b.data.qvel-a.data.qvel)>1e-6
