"""Elliptic MJX-Warp contact normals, reduced by world and robot part."""
import jax.numpy as jp


def normal_loads(data, terrain_geoms, wheel_ids, body_ids):
    """Return front/rear/body normal load (N) and nonwheel contact flag.

    Elliptic contact force component zero is the normal constraint force.
    Contacts are a global Warp buffer; constraint addresses are world-local.
    """
    c=data._impl;g=c.contact__geom;world=c.contact__worldid
    address=c.contact__efc_address[:,0];n=c.efc__force.shape[0]
    valid=(jp.arange(g.shape[0])<c.nacon[0])&(address>=0)&(address<c.efc__force.shape[1])
    valid &= (world>=0)&(world<n)&jp.any(terrain_geoms[jp.clip(g,0,terrain_geoms.shape[0]-1)]&(g>=0),-1)
    force=jp.where(valid,c.efc__force[jp.clip(world,0,n-1),jp.clip(address,0,c.efc__force.shape[1]-1)],0.)
    front=jp.any(g==wheel_ids[0],-1);rear=jp.any(g==wheel_ids[1],-1)
    body=jp.any(g[:,:,None]==body_ids[None,None,:],axis=(1,2))
    values=jp.stack((force*front,force*rear,force*body,(valid&body&(c.contact__dist<=0)).astype(force.dtype)),1)
    out=jp.zeros((n,4),dtype=force.dtype).at[jp.clip(world,0,n-1)].add(values)
    return out.at[:,3].set(out[:,3]>0)
