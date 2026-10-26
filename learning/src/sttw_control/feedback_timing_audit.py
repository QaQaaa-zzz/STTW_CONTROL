"""Offline force-window diagnostics; not a causal prefix/suffix intervention."""
import numpy as np


def force_window(active, impulse):
    active=np.asarray(active,bool);impulse=np.asarray(impulse,float)
    if active.ndim!=1 or impulse.shape!=(len(active),3):
        raise ValueError('aligned physical activity and world impulse required')
    if not np.isfinite(impulse[active]).all():raise ValueError('nonfinite active impulse')
    hits=np.flatnonzero(active & np.any(impulse!=0,axis=1))
    return (int(hits[0]),int(hits[-1]+1)) if len(hits) else None


def cost_partition(tick_cost, count):
    cost=np.asarray(tick_cost,float)
    if cost.ndim!=1 or not 0<=count<=len(cost) or not np.isfinite(cost).all():
        raise ValueError('finite observed costs and in-range boundary required')
    return dict(before=float(cost[:count].sum()),after=float(cost[count:].sum()),total=float(cost.sum()))


def angle_delta(a,b):
    return np.arctan2(np.sin(np.asarray(a)-np.asarray(b)),np.cos(np.asarray(a)-np.asarray(b)))


def describe(values):
    x=np.asarray(values,float)
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max())) if len(x) else dict(count=0,mean=None,median=None,minimum=None,maximum=None)
