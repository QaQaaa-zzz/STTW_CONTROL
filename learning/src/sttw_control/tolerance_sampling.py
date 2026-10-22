"""Packed resets ported from adopted path-command e381cff; task unchanged."""
import jax
import jax.numpy as j
from .path_command_statistics import merge_statistics, reduce_statistics, reset_statistics

def reset_finished_batch(ended,flags,reset,*,final_observation=None,observe=None):
    """Reset packed lanes and optionally replace only their observations."""
    eligible=flags['done']&~flags['engineering_fault'];n=eligible.shape[0]
    width=min(16,n);slots=j.arange(width)
    def packed_reset(initial):
        # Installed CPU XLA misfolds a closed one-hot reduction used as a
        # while bound. The barrier preserves that scalar, with no extra math.
        count=jax.lax.optimization_barrier(j.sum(eligible))
        indices=j.nonzero(eligible,size=n,fill_value=0)[0]
        def step(carry):
            offset,episodes,observations=carry
            ids=indices[j.minimum(offset+slots,n-1)]
            fresh=jax.vmap(reset)(ended.env_id[ids],ended.episode[ids]+1)
            destination=j.where(offset+slots<count,ids,n)
            episodes=jax.tree.map(lambda old,new:old.at[destination].set(new,mode='drop'),episodes,fresh)
            if observe is not None:
                fresh_obs=jax.vmap(observe)(fresh)
                observations=jax.tree.map(lambda old,new:old.at[destination].set(new,mode='drop'),observations,fresh_obs)
            return offset+width,episodes,observations
        return jax.lax.while_loop(lambda carry:carry[0]<count,step,(j.int32(0),*initial))[1:]
    episodes,observations=jax.lax.cond(j.any(eligible),packed_reset,lambda initial:initial,(ended,final_observation))
    return episodes if observe is None else (episodes,observations)


def make_packed_advance(batch):
    def advance(episodes, actions):
        ended, final, reward, flags, _, stats = jax.vmap(
            lambda ep, a: batch.advance_one(ep, a, auto_reset=False))(episodes, actions)
        nxt, obs = reset_finished_batch(ended, flags, batch.reset,
                                      final_observation=final, observe=batch.observation)
        resets = jax.vmap(reset_statistics)(nxt.route,
                    flags['done'] & ~flags['engineering_fault'])
        stats['diagnostics'] = reduce_statistics(merge_statistics(stats['diagnostics'], resets))
        return nxt, obs, reward, flags, stats
    return advance
