"""Offline true-contact intervals on a generated embedded-pebble mesh."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def raised_contact_and_load(samples,wheel,lane,origin,terrain):
    count=samples[wheel+'_normal_count'][:,lane].reshape(len(samples['physics_step']),-1)[:,0]
    start=samples[wheel+'_normal_start'][:,lane].reshape(len(count),-1)[:,0]
    width=int(count.max(initial=0));ids=start[:,None]+np.arange(width)
    valid=np.arange(width)[None,:]<count[:,None]
    ids=np.clip(ids,0,samples[wheel+'_normal_force'].shape[1]-1);rows=np.arange(len(count))[:,None]
    force=samples[wheel+'_normal_force'][rows,ids,0]*valid
    point=samples[wheel+'_normal_point'][rows,ids]-origin
    interp=RegularGridInterpolator((terrain['grid_x'],terrain['grid_y']),terrain['height_field']*.001,bounds_error=False,fill_value=0)
    height=interp(point[...,:2].reshape(-1,2)).reshape(force.shape)
    raised=valid&(force>.01)&(height>.0005)&(point[...,2]>.00025)
    direction=samples[wheel+'_normal_direction'][rows,ids]
    return raised.any(1),force.sum(1),(force*direction[...,2]).sum(1),point[...,2][raised]-height[raised]


def intervals(mask,time,dt):
    edge=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
    return [dict(on_s=float(time[i]-dt),off_s=float(time[j-1]),duration_s=float((j-i)*dt)) for i,j in zip(starts,ends)]
