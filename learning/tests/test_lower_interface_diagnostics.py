import numpy as np
import jax.numpy as jp
from sttw_control.controller import ControllerConfig, initial_controller, controller_step
from sttw_control.lower_interface_diagnostics import ecbc_contributions

def test_terms_reconstruct_controller_with_eso_and_clipping():
    cc=ControllerConfig();state=initial_controller(cc).replace(disturbance=jp.asarray(9.))
    for enabled in (False,True):
        m=jp.array([2.3,-.07,.12,.15,-.11,.03]);new,out=controller_step(state,m,enabled,cc)
        terms=ecbc_contributions(m,new.gains,out,enabled)
        np.testing.assert_allclose(jp.clip(terms.sum(),-cc.max_steer_rate,cc.max_steer_rate),out.steer_rate,rtol=2e-6,atol=2e-6)
        if not enabled:assert float(terms[2])==0
