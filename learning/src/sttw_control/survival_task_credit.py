"""Task-view reconstruction and complete-task GAE for physical-tail collections.

These functions do not extend the task past its domain boundary. The authentic
PathCommandEnvironment critic view is Actor351 + remaining-horizon fraction.
A caller must authenticate the saved observations, pre-action ticks, configuration
and old checkpoint before using this reconstruction for learning.
"""
import math
import numpy as np


def reconstruct_task_critic(observations, task_ticks, valid, *, dt_s, horizon_s):
    obs, ticks, mask = map(np.asarray, (observations, task_ticks, valid))
    if (obs.ndim != 2 or obs.shape[1] != 351 or not np.issubdtype(obs.dtype,np.floating)
            or mask.dtype != np.bool_ or mask.shape != (len(obs),) or ticks.shape != mask.shape
            or not np.issubdtype(ticks.dtype,np.integer)
            or not math.isfinite(dt_s) or not math.isfinite(horizon_s)
            or dt_s <= 0 or horizon_s <= 0):
        raise ValueError('aligned Actor351, Boolean mask, integer clock and positive finite times required')
    if (not np.isfinite(obs[mask]).all() or np.any(ticks[mask] < 0)
            or np.any(ticks[mask]*dt_s >= horizon_s)):
        raise ValueError('legal pre-action observation/tick outside task window')
    out=np.full((len(obs),352),np.nan,dtype=obs.dtype)
    out[mask,:351]=obs[mask]
    # Match the environment float32 clock arithmetic, not post-action time.
    dtype=obs.dtype
    out[mask,-1]=(np.asarray(horizon_s,dtype=dtype)-ticks[mask].astype(dtype)*np.asarray(dt_s,dtype=dtype))/np.asarray(horizon_s,dtype=dtype)
    return out


def complete_task_gae(rewards, old_values, valid, terminal, *, gamma, gae_lambda):
    """GAE on one contiguous task prefix with a verified final task terminal.

    `rewards` are sums of actually executed lower-tick scored rewards in each
    policy interval, including the genuine task terminal penalty. Hold/padding
    is not a task sample. The final task value bootstrap is zero, even if the
    physical episode continues. This estimator retains standard GAE truncation
    and value-approximation bias; it is not a new credit-assignment method.
    """
    reward,value,mask,done=map(np.asarray,(rewards,old_values,valid,terminal))
    if (reward.ndim != 1 or value.shape != reward.shape or mask.shape != reward.shape
            or done.shape != reward.shape or mask.dtype != np.bool_ or done.dtype != np.bool_
            or not np.issubdtype(reward.dtype,np.floating) or not np.issubdtype(value.dtype,np.floating)
            or not math.isfinite(gamma) or not math.isfinite(gae_lambda)
            or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1):
        raise ValueError('aligned task vectors and finite discount parameters required')
    indices=np.flatnonzero(mask)
    if (not indices.size or not np.array_equal(indices,np.arange(len(indices)))
            or not np.array_equal(np.flatnonzero(done),indices[-1:])
            or not np.isfinite(reward[mask]).all() or not np.isfinite(value[mask]).all()):
        raise ValueError('finite complete task prefix with exactly its last action terminal required')
    advantage=np.full(reward.shape,np.nan,dtype=np.float64)
    next_value=0.;tail=0.
    for t in indices[::-1]:
        delta=float(reward[t])+gamma*next_value-float(value[t])
        tail=delta+gamma*gae_lambda*tail
        advantage[t]=tail
        next_value=float(value[t])
    returns=np.full(reward.shape,np.nan,dtype=np.float64)
    returns[mask]=advantage[mask]+value[mask]
    return dict(advantages=advantage,returns=returns)
