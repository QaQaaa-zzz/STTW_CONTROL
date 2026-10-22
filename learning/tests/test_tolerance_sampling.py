import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import struct
from sttw_control.tolerance_sampling import reset_finished_batch

@struct.dataclass
class Episode:
    env_id: object
    episode: object
    history: object

def reset(eid, episode):
    return Episode(eid, episode, j.stack((eid.astype(float), episode.astype(float))))

def observe(ep):
    return ep.history * 2, ep.history + 1, j.asarray(False)

@pytest.mark.parametrize('count', [0, 1, 15, 16, 17, 32, 33])
def test_packed_reset_preserves_nonterminal_and_fault_lanes(count):
    n=33
    ep=Episode(j.arange(n), j.arange(n)+100, j.ones((n,2)))
    flags=dict(done=j.arange(n)<count, engineering_fault=j.arange(n)==2)
    eligible=flags['done'] & ~flags['engineering_fault']
    expected=jax.vmap(lambda e,ok:jax.lax.cond(ok,lambda _:reset(e.env_id,e.episode+1),lambda _:e,None))(ep,eligible)
    actual,obs=jax.jit(lambda e,f:reset_finished_batch(e,f,reset,final_observation=jax.vmap(observe)(e),observe=observe))(ep,flags)
    for a,b in zip(jax.tree.leaves((actual,obs)),jax.tree.leaves((expected,jax.vmap(observe)(expected)))):
        np.testing.assert_allclose(a,b)
