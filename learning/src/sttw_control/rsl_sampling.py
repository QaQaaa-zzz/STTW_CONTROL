"""Real closed-loop phase spreading and training-only complete episode statistics."""
import jax
import jax.numpy as jp
import torch


def phase_spread(state, key, count, horizon_steps, step, reset, action):
    """Freeze each full state after a stratified number of actual transitions.

    Never assign clocks independently of physics/history. Early ends use the normal
    reset path, so failures can reduce the intended phase coverage; report it.
    """
    key, draw = jax.random.split(key)
    targets = jax.random.permutation(draw, jp.arange(count)*horizon_steps//count)
    def choose(mask, new, old):
        return jax.tree.map(lambda a,b:jp.where(mask.reshape((count,)+(1,)*(a.ndim-1)),a,b),new,old)
    def tick(carry, i):
        current, key, resets = carry
        key, rk = jax.random.split(key)
        active = i < targets
        nxt = step(current, action(current))
        ended = active & nxt.done
        nxt = jax.lax.cond(jp.any(ended),lambda s:choose(ended,reset(jax.random.split(rk,count)),s),lambda s:s,nxt)
        return (choose(active,nxt,current),key,resets+jp.sum(ended)),None
    (state,key,resets),_ = jax.lax.scan(tick,(state,key,jp.array(0)),jp.arange(horizon_steps-1))
    return state,key,{'active_transitions':jp.sum(targets),'reset_count':resets,
                      'computed_transitions':jp.array(count*(horizon_steps-1)),
                      'target_ticks':targets}


class EpisodeStatistics:
    """Accumulate true reward (no value bootstrap), including physical failure.

    Completed training episodes can span multiple behavior policies. They are not
    fixed-checkpoint evaluations and must never be used to attribute a best model.
    """
    def __init__(self, complete_start):
        self.valid=complete_start.clone().bool()
        self.returns=torch.zeros_like(complete_start,dtype=torch.float32)
        self.lengths=torch.zeros_like(self.returns)
        self.totals=torch.zeros(4,device=self.returns.device)

    def add(self,reward,done,failed):
        self.returns+=reward;self.lengths+=1
        ended=done.bool() & self.valid
        self.totals+=torch.stack((ended.sum(),(self.returns*ended).sum(),
                                 (self.lengths*ended).sum(),(failed.bool() & ended).sum()))
        self.returns.masked_fill_(done.bool(),0);self.lengths.masked_fill_(done.bool(),0)
        self.valid |= done.bool()

    def flush(self):
        count,reward,length,failures=self.totals.cpu().tolist()
        self.totals.zero_()
        return {'count':int(count),'mean_return':reward/count if count else None,
                'mean_length_steps':length/count if count else None,
                'failure_fraction':failures/count if count else None,
                'scope':'complete training episodes, may span changing policies; not checkpoint selection'}
