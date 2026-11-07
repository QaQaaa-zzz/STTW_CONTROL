"""Batched Torch torque geometry from same-step native front-wheel patches."""
import torch
from .torch_observation import rotate_inverse


def contact_torque_time_batch(samples,body_order):
    """Flatten time/environment axes, retaining independent native patch offsets."""
    body=samples['body_state'];ticks,lanes=body.shape[:2]
    flat={'body_state':body.reshape(ticks*lanes,*body.shape[2:])}
    for kind in ('normal','friction'):
        prefix='frontwheel_'+kind+'_'
        capacity=samples[prefix+'point'].shape[1]
        for name,value in samples.items():
            if not name.startswith(prefix):continue
            if name.endswith('count'):
                flat[name]=value.reshape(ticks*lanes,-1)
            elif name.endswith('start'):
                offset=torch.arange(ticks,device=value.device).reshape(ticks,1,1)*capacity
                flat[name]=(value+offset).reshape(ticks*lanes,-1)
            else:flat[name]=value.reshape(ticks*capacity,*value.shape[2:])
    return {k:v.reshape(ticks,lanes) for k,v in front_contact_torques(flat,body_order).items()}


def front_contact_torques(samples,body_order,radii=(.1,.1,.025)):
    body=samples['body_state'];device=body.device;n=len(body)
    fork=body[:,body_order.index('frontfork')];wheel=body[:,body_order.index('frontwheel')]
    def rotate(q,v):
        conjugate=torch.cat((-q[...,:3],q[...,3:]),-1)
        return rotate_inverse(conjugate,v)
    basis=torch.eye(3,device=device,dtype=body.dtype)
    axis=rotate(fork[:,3:7],basis[1].expand(n,3));axle=rotate(wheel[:,3:7],basis[2].expand(n,3))
    forward=rotate(body[:,body_order.index('chasis'),3:7],basis[0].expand(n,3))
    zero=torch.zeros(n,device=device,dtype=body.dtype)
    result=dict(normal_moment=zero.clone(),friction_moment=zero.clone(),lateral_moment=zero.clone(),negative_lateral=zero.clone(),reference_delta=zero.clone())
    patches={}
    for kind in ['normal','friction']:
        count=samples['frontwheel_'+kind+'_count'][:,0];start=samples['frontwheel_'+kind+'_start'][:,0]
        slots=torch.arange(int(count.max()),device=device);mask=slots[None,:]<count[:,None]
        ids=(start[:,None]+slots[None,:]).clamp_max(len(samples['frontwheel_'+kind+'_point'])-1)
        point=samples['frontwheel_'+kind+'_point'][ids];force=samples['frontwheel_'+kind+'_force'][ids]
        if kind=='normal':
            normal=samples['frontwheel_normal_direction'][ids];force=force*normal
        lever=point-fork[:,None,:3]
        moment=(torch.cross(lever,force,dim=-1)*axis[:,None]).sum(-1)
        result[kind+'_moment']=torch.where(mask,moment,0.).sum(1)
        patches[kind]=(point,force,mask,ids)
    pn,_,mn,ni=patches['normal'];p,f,m,_=patches['friction']
    if p.shape[1]==0 or pn.shape[1]==0:return result
    distance=(p[:,:,None]-pn[:,None]).square().sum(-1)
    distance=torch.where(mn[:,None,:],distance,float('inf'));nearest=distance.argmin(-1)
    normal=samples['frontwheel_normal_direction'][torch.gather(ni,1,nearest)]
    usable=m&mn.any(1)[:,None]
    normal=torch.where(usable[:,:,None],normal,basis[2])
    direction=torch.cross(axle[:,None].expand_as(normal),normal,dim=-1)
    direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(1e-12)
    lateral=torch.cross(normal,direction,dim=-1)
    direction=direction*torch.where((direction*forward[:,None]).sum(-1)>=0,1.,-1.)[:,:,None]
    lateral_force=(f*lateral).sum(-1)[:,:,None]*lateral;lever=p-fork[:,None,:3]
    dot=(axis[:,None]*normal).sum(-1);regular=dot.abs()>1e-6
    distance=(lever*normal).sum(-1)/torch.where(regular,dot,1.)
    intersection=fork[:,None,:3]+axis[:,None]*distance[:,:,None]
    trail=((intersection-p)*direction).sum(-1)
    moment=(torch.cross(lever,lateral_force,dim=-1)*axis[:,None]).sum(-1)
    result['lateral_moment']=torch.where(usable,moment,0.).sum(1)
    result['negative_lateral']=torch.where(usable&regular&(trail<0),moment,0.).sum(1)
    up=basis[2].expand(n,3);q_up=rotate(wheel[:,3:7],rotate_inverse(wheel[:,3:7],up)*torch.tensor(radii,device=device,dtype=body.dtype).square())
    reference=wheel[:,:3]-q_up/torch.sqrt((q_up*up).sum(1))[:,None]
    horizontal=torch.cross(axle,up,dim=-1);horizontal/=torch.linalg.vector_norm(horizontal,dim=-1,keepdim=True).clamp_min(1e-12)
    horizontal*=torch.where((horizontal*forward).sum(1)>=0,1.,-1.)[:,None]
    regular_reference=axis[:,2].abs()>1e-6
    ref_trail=((fork[:,:3]-reference)*horizontal).sum(1)+(reference[:,2]-fork[:,2])*(axis*horizontal).sum(1)/torch.where(regular_reference,axis[:,2],1.)
    delta=(torch.cross(reference[:,None]-p,lateral_force,dim=-1)*axis[:,None]).sum(-1)
    result['reference_delta']=torch.where(usable&(regular_reference&(ref_trail>0))[:,None],delta,0.).sum(1)
    return result
