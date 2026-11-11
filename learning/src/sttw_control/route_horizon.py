"""Nominal route-time screening; requested speed is not a physical speed bound."""
import numpy as np


def remaining_route_length(roots, roads, goal):
    segment=np.diff(roads,axis=1);length=np.linalg.norm(segment,axis=-1)
    cumulative=np.concatenate((np.zeros((len(roads),1)),np.cumsum(length,axis=1)),axis=1)
    delta=roots[:,None,None,:2]-roads[None,:,:-1]
    fraction=((delta*segment[None]).sum(-1)/(length**2)[None]).clip(0,1)
    distance=np.linalg.norm(delta-fraction[...,None]*segment[None],axis=-1)
    nearest=distance.argmin(-1);route=np.arange(len(roads))[None]
    progress=cumulative[route,nearest]+fraction[np.arange(len(roots))[:,None],route,nearest]*length[route,nearest]
    inside=(roads[:,:,0]>=goal['x_min_m'])&(roads[:,:,0]<=goal['x_max_m'])&(abs(roads[:,:,1])<=goal['abs_y_max_m'])
    end=np.where(inside.any(1),cumulative[np.arange(len(roads)),inside.argmax(1)],np.inf)
    return np.maximum(end[None]-progress,0.)


def nominal_horizon(roots,roads,parameters,goal,time_s,horizon_s,margin_s=0.):
    length=remaining_route_length(roots,roads,goal)
    duration=length/parameters[None,:,5]+margin_s
    allowed=(duration<=horizon_s-np.asarray(time_s).reshape(-1,1))|(parameters[None,:,-1]>.5)
    return allowed,duration
