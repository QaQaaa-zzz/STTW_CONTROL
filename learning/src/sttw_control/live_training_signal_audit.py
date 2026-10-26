"""Offline descriptive return/calibration checks; never updates a policy."""
import math
import numpy as np


def discounted_returns(rewards,active,*,gamma):
    rewards=np.asarray(rewards,float);active=np.asarray(active)
    if rewards.ndim!=2 or active.shape!=rewards.shape or active.dtype!=np.bool_ or not math.isfinite(gamma) or not 0<=gamma<=1:
        raise ValueError('aligned time-by-episode rewards/Boolean mask and unit discount required')
    if np.any(np.diff(active.astype(np.int8),axis=0)>0) or not np.all(active[0]) or not np.isfinite(rewards[active]).all():
        raise ValueError('complete finite contiguous episode prefixes required')
    end=np.zeros(rewards.shape[1]);result=np.full(rewards.shape,np.nan)
    for t in range(len(rewards)-1,-1,-1):
        end=np.where(active[t],np.where(active[t],rewards[t],0.)+gamma*end,0.)
        result[t]=np.where(active[t],end,np.nan)
    return result


def value_calibration(predicted,target,mask):
    predicted,target,mask=map(np.asarray,(predicted,target,mask));mask=mask.astype(bool)
    if predicted.shape!=target.shape or mask.shape!=target.shape:raise ValueError('aligned calibration arrays required')
    if not mask.any():return dict(samples=0,mean_prediction_minus_target=None,rmse=None,explained_variance=None)
    p,t=predicted[mask].astype(float),target[mask].astype(float)
    if not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('finite selected values required')
    error=p-t;variance=float(np.var(t))
    return dict(samples=int(mask.sum()),mean_prediction_minus_target=float(error.mean()),rmse=float(np.sqrt(np.mean(error**2))),
        explained_variance=None if variance<1e-12 else float(1-np.var(error)/variance),target_std=float(np.std(t)))
