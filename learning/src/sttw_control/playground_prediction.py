"""Batched GPU nominal prediction with the same causal ARX and pose integrator.

Route and model tensors are resident on the execution device. The caller can
capture rollout in a CUDA graph; no host conversion occurs during prediction.
"""
import torch


def rotate(q, v):
    return v+2*torch.cross(q[:,1:],torch.cross(q[:,1:],v,dim=-1)+q[:,:1]*v,dim=-1)


def multiply(a,b):
    return torch.cat((a[:,:1]*b[:,:1]-(a[:,1:]*b[:,1:]).sum(-1,keepdim=True),
        a[:,:1]*b[:,1:]+b[:,:1]*a[:,1:]+torch.cross(a[:,1:],b[:,1:],dim=-1)),1)


def euler(q):
    w,x,y,z=q.unbind(-1)
    return torch.stack((torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
        torch.asin((2*(w*y-z*x)).clamp(-1,1)),torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))),1)


class NominalRollout:
    def __init__(self, model, follower, device='cuda', dtype=torch.float32):
        self.mean=torch.as_tensor(model['mean'],device=device,dtype=dtype)
        self.scale=torch.as_tensor(model['scale'],device=device,dtype=dtype)
        self.weight=torch.as_tensor(model['weights'][:-1],device=device,dtype=dtype)
        self.bias=torch.as_tensor(model['weights'][-1],device=device,dtype=dtype)
        self.identity=bool(model.get('identity_prior',False));self.follower=follower

    def features(self,pose,xi,s,xy,heading,curvature):
        nearest=(xy-pose[:,None,:2]).square().sum(-1).argmin(-1)
        index=torch.stack(((nearest-1).clamp_min(0),nearest.clamp_max(s.shape[0]-2)),1)
        gather=lambda values,i:torch.gather(values,1,i[:,:,None].expand(-1,-1,values.shape[-1]))
        a=gather(xy,index);v=gather(xy,index+1)-a
        u=(((pose[:,None,:2]-a)*v).sum(-1)/v.square().sum(-1)).clamp(0,1)
        point=a+u[:,:,None]*v;j=(pose[:,None,:2]-point).square().sum(-1).argmin(-1,keepdim=True)
        index=index.gather(1,j).squeeze(1);fraction=u.gather(1,j).squeeze(1)
        point=point.gather(1,j[:,:,None].expand(-1,-1,2)).squeeze(1)
        distance=s[index]+fraction*(s[index+1]-s[index])
        def interp(values,at):
            i=torch.searchsorted(s,at.contiguous(),right=True).sub(1).clamp(0,len(s)-2)
            f=((at-s[i])/(s[i+1]-s[i])).clamp(0,1)
            return values.gather(1,i[:,None]).squeeze(1)*(1-f)+values.gather(1,(i+1)[:,None]).squeeze(1)*f
        h=interp(heading,distance);delta=pose[:,:2]-point
        ey=-h.sin()*delta[:,0]+h.cos()*delta[:,1];c=self.follower
        look=(c['lookahead_time_s']*xi[:,0].clamp_min(c['lookahead_min_speed_m_s'])).clamp(c['lookahead_min_m'],c['lookahead_max_m'])*c.get('lookahead_scale',1.)
        ahead=torch.stack((interp(xy[:,:,0],distance+look),interp(xy[:,:,1],distance+look)),1)-pose[:,:2]
        angles=euler(pose[:,3:7]);yaw=angles[:,2];cs=yaw.cos();sn=yaw.sin()
        preview=torch.stack((cs*ahead[:,0]+sn*ahead[:,1],-sn*ahead[:,0]+cs*ahead[:,1]),1)
        dh=h-yaw
        return torch.cat((pose[:,2:3],angles[:,:2],sn[:,None],cs[:,None],pose[:,7:8],
            torch.stack([interp(curvature,distance+d) for d in (0.,.5,1.)],1),ey[:,None],
            torch.atan2(dh.sin(),dh.cos())[:,None],preview,torch.full_like(look[:,None],1.8),look[:,None]),1)

    def step(self,pose,xi,previous,s,xy,heading,curvature):
        eta=self.features(pose,xi,s,xy,heading,curvature)
        x=torch.cat((xi,previous,eta),1)
        nxt=((x-self.mean)/self.scale)@self.weight+self.bias
        if self.identity:nxt=nxt+xi
        mid=(xi+nxt)*.5;rv=mid[:,3:6]*.02;angle=rv.norm(dim=-1,keepdim=True)
        dq=torch.cat(((angle*.5).cos(),rv*.5*torch.sinc(angle/(2*torch.pi))),1)
        q=multiply(pose[:,3:7],dq);q=q/q.norm(dim=-1,keepdim=True)
        xyz=pose[:,:3]+.01*(rotate(pose[:,3:7],xi[:,:3])+rotate(q,nxt[:,:3]))
        pose=torch.cat((xyz,q,pose[:,7:10]+mid[:,6:9]*.02,pose[:,10:12]),1)
        return pose,nxt

    def compile_step(self):
        self.step=torch.compile(self.step,fullgraph=True)

    def __call__(self,pose,xi,previous,s,xy,heading,curvature,steps=100):
        pose=pose.contiguous();xi=xi.contiguous();previous=previous.contiguous()
        poses=[pose];velocities=[xi]
        for _ in range(steps):
            pose,nxt=self.step(pose,xi,previous,s,xy,heading,curvature)
            previous,xi=xi,nxt;poses.append(pose);velocities.append(xi)
        return torch.stack(poses,1),torch.stack(velocities,1)


class BodyEnvelope:
    """Collision-geometry planar support, including the locked pendulum links.

    MuJoCo model topology is read once; joint coordinates come from each
    predicted pose. Bounds are exact axis supports, not root-point checks.
    """
    def __init__(self,model,device='cuda',dtype=torch.float32):
        import numpy as np
        names=['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint']
        coordinate={model.joint(name).id:7+i for i,name in enumerate(names)}
        tensor=lambda x:torch.as_tensor(np.array(x),device=device,dtype=dtype)
        self.links=[]
        for body in range(2,model.nbody):
            joint=int(model.body_jntadr[body]);self.links.append((int(model.body_parentid[body]),
                tensor(model.body_pos[body]),tensor(model.body_quat[body]),tensor(model.jnt_pos[joint]),
                tensor(model.jnt_axis[joint]),coordinate[joint],float(model.qpos0[model.jnt_qposadr[joint]])))
        self.geoms=[]
        for g in range(model.ngeom):
            if model.geom_bodyid[g]>0 and (model.geom_contype[g] or model.geom_conaffinity[g]):
                self.geoms.append((int(model.geom_bodyid[g]),tensor(model.geom_pos[g]),tensor(model.geom_quat[g]),int(model.geom_type[g]),tensor(model.geom_size[g]),float(model.geom_rbound[g])))

    def __call__(self,pose):
        shape=pose.shape[:-1];p=pose.reshape(-1,12);n=len(p)
        position=[None,p[:,:3]];orientation=[None,p[:,3:7]]
        for parent,offset,localq,anchor,axis,column,reference in self.links:
            pq=orientation[parent];base=position[parent]+rotate(pq,offset.expand(n,-1))
            q=multiply(pq,localq.expand(n,-1));angle=(p[:,column:column+1]-reference)*.5
            turn=torch.cat((angle.cos(),angle.sin()*axis),1)
            nq=multiply(q,turn);base=base+rotate(q,anchor.expand(n,-1))-rotate(nq,anchor.expand(n,-1))
            position.append(base);orientation.append(nq)
        lower=[];upper=[]
        for body,offset,localq,kind,size,radius in self.geoms:
            q=orientation[body];center=position[body]+rotate(q,offset.expand(n,-1));q=multiply(q,localq.expand(n,-1))
            w,x,y,z=q.unbind(-1)
            directions=torch.stack((torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),1),
                                    torch.stack((2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),1)),1)
            if kind==6:extent=(directions.abs()*size).sum(-1)
            elif kind==4:extent=(directions*size).square().sum(-1).sqrt()
            elif kind==5:extent=directions[:,:,:2].square().sum(-1).sqrt()*size[0]+directions[:,:,2].abs()*size[1]
            else:extent=torch.full_like(center[:,:2],radius)
            lower.append(center[:,:2]-extent);upper.append(center[:,:2]+extent)
        return torch.stack(lower,1).reshape(*shape,len(lower),2),torch.stack(upper,1).reshape(*shape,len(upper),2)
