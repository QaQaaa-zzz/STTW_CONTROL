from types import SimpleNamespace
import jax.numpy as j
import numpy as np
from sttw_control.passive_sensor_probe import attach_sensor_log


def test_logging_preserves_step_arguments_and_carry_without_actuating_detector():
    obs=j.zeros(351).at[318].set(.1).at[298].set(.12).at[334:336].set(1.)
    class Env:
        def observation(self,state,path,speed,horizon):return obs,j.zeros(1),j.bool_(False)
    carry=SimpleNamespace(state=object());params=object();path=object();pulse=object();end=object()
    def advance(c,p,pa,s,pu,m):
        assert c is carry and p is params and pa is path and pu is pulse
        return end,dict(issued_latent=j.ones((4,2)),gate_detected=j.zeros(4,dtype=bool))
    result,logs=attach_sensor_log(advance,Env(),10.)(carry,params,path,1.1,pulse,1)
    assert result is end
    np.testing.assert_array_equal(logs['issued_latent'],np.ones((4,2)))
    assert not np.asarray(logs['gate_detected']).any()
    np.testing.assert_allclose(logs['probe_eso_signal'],.02,atol=1e-7)
    assert np.asarray(logs['probe_eso_history_valid']).all()
