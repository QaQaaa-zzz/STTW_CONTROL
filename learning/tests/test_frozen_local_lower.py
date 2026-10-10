import importlib
import jax.numpy as jp
import numpy as np
from types import SimpleNamespace

def api():
    try:return importlib.import_module('sttw_control.frozen_local_lower')
    except ModuleNotFoundError:assert False,'diagnostic frozen local lower not implemented'

def make():
    m=api();from sttw_control.controller import ControllerConfig,initial_controller
    from sttw_control.network import ResidualActor
    import jax
    net=ResidualActor((256,128),activation='elu');params=net.init(jax.random.PRNGKey(1),jp.zeros(210))
    return m.FrozenLocalLower(params),ControllerConfig(),initial_controller(ControllerConfig())

def test_prepare_uses_unscreened_current_governed_and_one_causal_frame():
    lower,cc,controller=make();s=lower.initial(jp.array([0.,0.,0.]));gov=jp.array([3.,.3]);measurement=jp.array([0.,0.,0.,0.,0.,23.,23.])
    physical=SimpleNamespace(controller=controller,actuator=SimpleNamespace(previous=jp.array([0.,23.])),governor=SimpleNamespace(current_reference=jp.array([2.3,0.])))
    s,action,obs,flags=lower.prepare_local(s,measurement,jp.array([0.,0.,0.]),2.3,gov,physical,cc)
    assert obs.shape==(210,);np.testing.assert_array_equal(s.frames[-1,8:10],gov);assert float(s.mask.sum())==1
    assert bool(flags['finite']);np.testing.assert_array_less(abs(action),1.000001)

def test_after_step_retains_history_and_next_prepare_uses_executed_residual():
    lower,cc,controller=make();s=lower.initial(jp.zeros(3));physical=SimpleNamespace(controller=controller,actuator=SimpleNamespace(previous=jp.array([0.,23.])),governor=SimpleNamespace(current_reference=jp.array([2.3,0.])))
    args=(jp.array([0.,0.,0.,0.,0.,23.,23.]),jp.zeros(3),2.3,jp.array([2.3,0.]),physical,cc)
    s,_,_,_=lower.prepare_local(s,*args);s=lower.finish_local(s,jp.array([1.,5.]),.01)
    assert float(s.mask.sum())==1
    s,_,_,_=lower.prepare_local(s,*args);assert float(s.mask.sum())==2
    np.testing.assert_array_equal(s.frames[-1,16:18],jp.array([1.,5.]))
    assert float(s.frames[-1,5])==float(jp.asarray(.01))
    assert not hasattr(s,'tick') and not hasattr(s,'path') and not hasattr(s,'alpha')
