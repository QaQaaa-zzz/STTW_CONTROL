"""Margin-conditioned states with force and alpha-stratum marginals fixed.

This is a TRAIN sampler, not a policy input or a safety filter. Zero-force
episodes keep the uniform state draw. Matched seeds give identical force and
stratum draws at equal reset counters; early termination may change counters.
"""
import math
import numpy as np
import jax
import jax.numpy as j


def alpha_strata(alpha):
    """Endpoints separate; continuous preferences occupy three interior bins."""
    return j.where(alpha==0,0,j.where(alpha==1,4,1+j.minimum((alpha*3).astype(j.int32),2)))


def exact_alpha_groups(alpha):
    """Host-side table identity; never mix different actual alpha values.

    A useful guided bank needs repeated authentic contexts at each alpha.
    Unique continuous alphas form singleton groups and cannot be reweighted.
    """
    alpha=np.asarray(alpha)
    if alpha.ndim!=1 or not len(alpha) or not np.isfinite(alpha).all() or np.any((alpha<0)|(alpha>1)):
        raise ValueError('finite alpha[N] in [0,1] required')
    values,inverse=np.unique(alpha,return_inverse=True)
    return values,j.asarray(inverse,dtype=j.int32)


def stratified_permutation(groups,seed):
    groups=np.asarray(groups)
    if groups.ndim!=1 or not len(groups) or not np.issubdtype(groups.dtype,np.integer) or np.any(groups<0):
        raise ValueError('nonempty nonnegative integer alpha groups required')
    rng=np.random.default_rng(seed);permutation=np.arange(len(groups))
    for group in np.unique(groups):
        ids=np.flatnonzero(groups==group);permutation[ids]=rng.permutation(ids)
    return permutation


def state_probabilities(risk,groups,*,strength=.7,bandwidth=.2,target=1.,num_groups=5):
    """P(state | force, alpha group), shape [F,num_groups,N].

    Focus near the predicted return boundary; retain uniform support. Empty
    strata are unreachable in draw(), but get a finite uniform fallback table.
    Invalid risk is never treated as measured low reserve.
    """
    if not 0<=strength<=1 or not math.isfinite(bandwidth) or bandwidth<=0 or not math.isfinite(target) or not isinstance(num_groups,int) or num_groups<1:
        raise ValueError('invalid declared sampler parameters')
    risk=j.asarray(risk);groups=j.asarray(groups)
    if risk.ndim!=2 or groups.shape!=(risk.shape[0],) or not risk.shape[0]:
        raise ValueError('risk[N,F] and matching groups[N] required')
    mask=groups[None,:]==j.arange(num_groups)[:,None]
    count=mask.sum(-1,keepdims=True)
    base=j.where(count>0,mask/j.maximum(count,1),j.ones_like(mask,dtype=j.float32)/len(groups))
    weights=j.where(j.isfinite(risk),j.exp(-.5*((risk-target)/bandwidth)**2),0.).T[:,None,:]*mask[None,:,:]
    total=weights.sum(-1,keepdims=True)
    focus=j.where(total>0,weights/j.where(total>0,total,1.),base[None,:,:])
    return (1-strength)*base[None,:,:]+strength*focus


def draw(key,probabilities,groups,*,enabled_probability=.8):
    choose,enable,force,state=jax.random.split(key,4)
    base=jax.random.randint(choose,(),0,len(groups))
    group=groups[base]
    force_id=jax.random.randint(force,(),0,probabilities.shape[0])
    enabled=jax.random.bernoulli(enable,enabled_probability)
    idx=jax.random.categorical(state,j.log(probabilities[force_id,group]))
    return dict(state_index=j.where(enabled,idx,base),base_state_index=base,
        force_index=j.where(enabled,force_id+1,0),enabled=enabled,stratum=group)


def bank_risks(states,controller,forces,*,effort_budget,gamma,duration,horizon):
    """Same fixed scalar coordinate as measurement_reserve, without refitting."""
    from .controller import _system
    def one(st):
        frame=st.history.frames[-1]
        a1,a2,a4,_,_=_system(frame[6],st.physical.controller.gains,controller)
        lam=j.sqrt(a4);z=a2/lam*frame[2]+lam*frame[0]+frame[1]+st.physical.controller.disturbance/lam
        T=j.minimum(1.,horizon-st.tick*controller.dt)
        W=(a1+a2/lam)**2*(-j.expm1(-2*lam*T))/(2*lam)
        K=gamma*(-j.expm1(-lam*duration))/lam
        risk=j.abs(z+K*forces)/j.sqrt(effort_budget*W)
        valid=(T>=duration)&(W>0)&j.all(j.isfinite(risk))
        return j.where(valid,risk,j.nan)
    return jax.vmap(one)(states)
