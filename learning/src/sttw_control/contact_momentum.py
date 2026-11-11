"""Whole-vehicle angular momentum about its moving center of mass.

Input body states use world link poses (xyzw quaternions), link-origin linear
velocity, and world angular velocity. Inertia is about COM in link coordinates,
as specified by Isaac Lab ArticulationData.default_inertia.
"""
import numpy as np


def rotate_xyzw(q,v):
    t=2*np.cross(q[...,:3],v)
    return v+q[...,3:]*t+np.cross(q[...,:3],t)


def roll_momentum(body,masses,com_offsets,inertias,chassis_index=0):
    q=body[...,3:7];omega=body[...,10:13]
    offsets=rotate_xyzw(q,com_offsets)
    positions=body[...,:3]+offsets
    velocities=body[...,7:10]+np.cross(omega,offsets)
    center=np.sum(positions*masses[...,None],axis=-2)/np.sum(masses,axis=-1)[...,None]
    inverse=q.copy();inverse[...,:3]*=-1
    local_omega=rotate_xyzw(inverse,omega)
    spin=rotate_xyzw(q,np.einsum('...ij,...j->...i',inertias,local_omega))
    orbital=np.cross(positions-center[...,None,:],masses[...,None]*velocities)
    angular_momentum=np.sum(spin+orbital,axis=-2)
    axis=-rotate_xyzw(q[...,chassis_index,:],np.array([1.,0.,0.]))
    axis_rate=np.cross(omega[...,chassis_index,:],axis)
    return dict(roll_momentum=np.sum(angular_momentum*axis,axis=-1),
                axis_transport_rate=np.sum(angular_momentum*axis_rate,axis=-1),
                angular_momentum_world=angular_momentum,center_world=center,
                roll_axis_world=axis)
