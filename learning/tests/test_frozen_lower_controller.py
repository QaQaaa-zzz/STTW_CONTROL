import json
from pathlib import Path
import numpy as np
import jax.numpy as jp
from types import SimpleNamespace
from sttw_control.frozen_lower_controller import FrozenLowerController,path_features

SPEC=json.loads((Path(__file__).parents[1]/'configs/direct_command_frozen_lower_500.json').read_text())

def test_pinned_actor_observation_and_reset():
    lower=FrozenLowerController(SPEC['lower_controller'])
    s=lower.initial(jp.zeros(3));m=jp.array([0.,0.,0.,0.,0.,23.,23.])
    out=SimpleNamespace(reference_roll=jp.asarray(0.),steer_rate=jp.asarray(0.),disturbance=jp.asarray(0.))
    s,a,obs,flags=lower.prepare(s,m,jp.zeros(3),jp.array([2.3,0.]),out,jp.array([0.,23.]))
    assert obs.shape==(280,) and a.shape==(2,) and bool(flags['finite'])
    assert np.max(np.abs(np.asarray(a)))<=1
    np.testing.assert_allclose(obs[9*27+18],1.)
    s=lower.after_step(s,jp.array([2.3,0.]),jp.array([.0115,0.,0.]),m,2.3,a,False,0)
    assert int(s.valid_count)==2
    np.testing.assert_allclose(s.return_state.previous_action,a)
    np.testing.assert_allclose(s.reference_pose,[.0115,0.,0.],atol=1e-6)
    reset=lower.initial(jp.array([4.,5.,.2]));assert int(reset.valid_count)==1
    assert not np.asarray(reset.history.mask).any()

def test_path_right_normal_and_endpoint_no_future_turn():
    lower=FrozenLowerController(SPEC['lower_controller']);s=lower.initial(jp.zeros(3))
    s=s.replace(path_points=s.path_points.at[1].set(jp.array([1.,0.,0.,.2])),valid_count=jp.int32(2))
    f,ext=path_features(s,jp.array([.5,1.,.1]));np.testing.assert_allclose(f,[-1.,.1,.1],atol=1e-6)
    assert not bool(ext)
    f,ext=path_features(s,jp.array([2.,1.,.1]));np.testing.assert_allclose(f,[-1.,.1,.2],atol=1e-6)
    assert bool(ext)

def test_hash_mismatch_rejected():
    import pytest
    bad=dict(SPEC['lower_controller'],payload_sha256='wrong')
    with pytest.raises(ValueError,match='payload'):FrozenLowerController(bad)
