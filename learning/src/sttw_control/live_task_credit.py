"""Physical-time SMDP GAE between legal middle-policy decisions.

This is an optional live-task estimator, not a change to the historical task
prefix estimator. Fixed fallback actions are not invented policy samples.
"""
import math
import numpy as np


def live_task_gae(rewards, old_values, actor_valid, active, terminal, *, gamma, gae_lambda):
    """Fold real fallback rewards into the preceding legal policy transition.

    For legal decisions at i and j, R_i=sum(gamma**k*r[i+k], k<j-i)
    and Gamma_i=gamma**(j-i). The trace coefficient is (gamma*lambda)**
    (j-i), explicitly interpreting lambda in actual upper-control-step units.
    The final transition extends to the physical contact/window terminal and
    has zero value bootstrap. Values are required only at legal decisions.

    This has the usual value-approximation and lambda-return bias. It does not
    establish causal recovery benefits, safety, or an unbiased PPO theorem.
    The caller must verify same-policy provenance and a genuine terminal.
    """
    reward,value,legal,seen,done=map(np.asarray,(rewards,old_values,actor_valid,active,terminal))
    if (reward.ndim!=1 or any(x.shape!=reward.shape for x in [value,legal,seen,done])
            or any(x.dtype!=np.bool_ for x in [legal,seen,done])
            or not np.issubdtype(reward.dtype,np.floating)
            or not np.issubdtype(value.dtype,np.floating)
            or not math.isfinite(gamma) or not math.isfinite(gae_lambda)
            or not 0<=gamma<=1 or not 0<=gae_lambda<=1):
        raise ValueError('aligned reward/value vectors, Boolean masks and finite unit-interval discounts required')
    indices=np.flatnonzero(seen);decisions=np.flatnonzero(legal)
    if (not indices.size or not decisions.size
            or not np.array_equal(indices,np.arange(len(indices)))
            or not np.array_equal(np.flatnonzero(done),indices[-1:])
            or np.any(legal&~seen)
            or not np.isfinite(reward[seen]).all()
            or not np.isfinite(value[legal]).all()):
        raise ValueError('one complete active episode, real last terminal and finite legal decisions/rewards required')
    advantages=np.full(reward.shape,np.nan,dtype=np.float64)
    returns=np.full(reward.shape,np.nan,dtype=np.float64)
    transition_rewards=np.full(reward.shape,np.nan,dtype=np.float64)
    discounts=np.full(reward.shape,np.nan,dtype=np.float64)
    durations=np.zeros(reward.shape,dtype=np.int64)
    tail=0.
    for position in range(len(decisions)-1,-1,-1):
        start=int(decisions[position]);has_next=position+1<len(decisions)
        end=int(decisions[position+1]) if has_next else len(indices)
        duration=end-start
        powers=np.power(float(gamma),np.arange(duration,dtype=np.int64))
        aggregated=float(np.dot(powers,reward[start:end].astype(np.float64)))
        discount=float(gamma)**duration if has_next else 0.
        next_value=float(value[end]) if has_next else 0.
        delta=aggregated+discount*next_value-float(value[start])
        tail=delta+(float(gamma)*float(gae_lambda))**duration*tail
        advantages[start]=tail;returns[start]=tail+float(value[start])
        transition_rewards[start]=aggregated;discounts[start]=discount;durations[start]=duration
    prefix=int(decisions[0])
    prefix_reward=float(np.dot(np.power(float(gamma),np.arange(prefix)),reward[:prefix].astype(np.float64)))
    return dict(advantages=advantages,returns=returns,
        discounted_transition_rewards=transition_rewards,duration_steps=durations,
        continuation_discounts=discounts,unassigned_prefix_steps=prefix,
        unassigned_discounted_prefix_reward=prefix_reward,
        semantics='Only legal policy samples; rewards and trace discount follow actual elapsed upper steps.')
