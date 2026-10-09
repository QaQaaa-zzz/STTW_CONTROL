import jax
import jax.numpy as jp
import numpy as np
import pytest
from sttw_control.direct_command_training import conditional_batch_reset


@pytest.mark.parametrize('count',[0,1,2,4])
def test_batch_guard_preserves_every_leaf_and_reset_rng(count):
    state=dict(env_id=jp.arange(4),episode=jp.arange(4)+10,history=jp.arange(24).reshape(4,3,2).astype(float),eso=jp.ones((4,2)))
    def reset(s):
        key=jax.random.fold_in(jax.random.PRNGKey(87),s['env_id'])
        key=jax.random.fold_in(key,s['episode']+1)
        return dict(env_id=s['env_id'],episode=s['episode']+1,history=jp.zeros_like(s['history']),eso=jax.random.normal(key,(2,)))
    done=jp.arange(4)<count
    reference=jax.jit(jax.vmap(lambda s,d:jax.lax.cond(d,reset,lambda x:x,s)))
    candidate=jax.jit(lambda s,d:conditional_batch_reset(s,d,reset))
    for a,b in zip(jax.tree.leaves(reference(state,done)),jax.tree.leaves(candidate(state,done))):np.testing.assert_array_equal(a,b)
    text=candidate.lower(state,done).compiler_ir(dialect='hlo').as_hlo_text()
    assert 'conditional(' in text
