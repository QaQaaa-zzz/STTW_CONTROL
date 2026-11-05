"""Fixed-route Torch interpolation, bounded projection and pursuit.

Routes are exported before rollout; never reanchor to the moving vehicle.
One shared route or a saved route bank is indexed by lane, with per-lane progress.
"""
import math
import torch


def wrap(angle):
    return torch.atan2(torch.sin(angle),torch.cos(angle))


class TorchPathReference:
    def __init__(self, route, config, device='cpu',route_ids=None):
        self.config=config
        self.s=torch.as_tensor(route['s'],device=device,dtype=torch.float32)
        self.xy=torch.as_tensor(route['xy'],device=device,dtype=torch.float32)
        self.heading=torch.as_tensor(route['heading'],device=device,dtype=torch.float32)
        self.curvature=torch.as_tensor(route['curvature'],device=device,dtype=torch.float32)
        self.batched=self.xy.ndim==3
        self.route_ids=(torch.arange(len(self.xy),device=device) if route_ids is None else torch.as_tensor(route_ids,device=device,dtype=torch.long)) if self.batched else None
        n=len(self.s)
        if n<2 or self.xy.shape[-2:]!=(n,2) or self.heading.shape!=self.xy.shape[:-1] or self.curvature.shape!=self.heading.shape or not all(
                torch.isfinite(x).all() for x in (self.s,self.xy,self.heading,self.curvature)) or not (self.s[1:]>self.s[:-1]).all() or not ((self.xy[...,1:,:]-self.xy[...,:-1,:]).square().sum(-1)>0).all():
            raise ValueError('invalid fixed route')

    def at(self,progress):
        q=progress.clamp(self.s[0],self.s[-1])
        i=(torch.searchsorted(self.s,q.contiguous(),right=True)-1).clamp(0,len(self.s)-2)
        u=(q-self.s[i])/(self.s[i+1]-self.s[i])
        if self.batched:
            rows=self.route_ids
            return (self.xy[rows,i]+u[...,None]*(self.xy[rows,i+1]-self.xy[rows,i]),self.heading[rows,i]+u*(self.heading[rows,i+1]-self.heading[rows,i]),self.curvature[rows,i]+u*(self.curvature[rows,i+1]-self.curvature[rows,i]))
        return (self.xy[i]+u[...,None]*(self.xy[i+1]-self.xy[i]),
                self.heading[i]+u*(self.heading[i+1]-self.heading[i]),
                self.curvature[i]+u*(self.curvature[i+1]-self.curvature[i]))

    def project(self,xy,previous,distance):
        c=self.config['path']
        lo=(previous-c['projection_backward_m']).clamp_min(0)
        hi=torch.minimum(self.s[-1],previous+torch.maximum(torch.full_like(distance,c['projection_forward_floor_m']),c['projection_forward_motion_multiplier']*distance+c['projection_forward_padding_m']))
        selected=self.xy[self.route_ids] if self.batched else self.xy
        a=selected[...,:-1,:];v=selected[...,1:,:]-a;span=self.s[1:]-self.s[:-1]
        u=((xy[:,None]-a)*v).sum(-1)/v.square().sum(-1)
        low=((lo[:,None]-self.s[:-1])/span).clamp(0,1)
        high=((hi[:,None]-self.s[:-1])/span).clamp(0,1)
        u=torch.maximum(low,torch.minimum(high,u))
        ss=self.s[:-1]+u*span;points=a+u[...,None]*v
        valid=(self.s[1:]>=lo[:,None])&(self.s[:-1]<=hi[:,None])
        d2=torch.where(valid,(xy[:,None]-points).square().sum(-1),float('inf'))
        minimum=d2.min(-1).values
        index=torch.where(d2<=minimum[:,None]+1e-10,(ss-previous[:,None]).abs(),float('inf')).argmin(-1)
        lane=torch.arange(len(xy),device=xy.device)
        unclamped=ss[lane,index];point=points[lane,index];heading=self.at(unclamped)[1]
        delta=xy-point;cross=-torch.sin(heading)*delta[:,0]+torch.cos(heading)*delta[:,1]
        return dict(progress=torch.maximum(previous,unclamped),unclamped=unclamped,point=point,cross_track=cross,heading=heading)

    def pursuit(self,progress,pose,speed,v_user):
        c=self.config['follower']
        look=(c['lookahead_time_s']*speed.clamp_min(c['lookahead_min_speed_m_s'])).clamp(c['lookahead_min_m'],c['lookahead_max_m'])
        point=self.at(progress+look)[0];heading=self.at(progress)[1]
        delta=point-pose[:,:2];cs=torch.cos(pose[:,2]);sn=torch.sin(pose[:,2])
        body=torch.stack((cs*delta[:,0]+sn*delta[:,1],-sn*delta[:,0]+cs*delta[:,1]),dim=-1)
        k=2*body[:,1]/body.square().sum(-1).clamp_min(1e-12)
        steer=torch.atan(c['wheelbase_m_expected']*k/math.cos(math.radians(c['caster_deg_expected'])))
        command=torch.stack((v_user,steer.clamp(-c['steer_reference_max_rad'],c['steer_reference_max_rad'])),dim=-1)
        numerical_fault=~torch.isfinite(torch.cat((command,body,heading[:,None],progress[:,None],speed[:,None]),dim=-1)).all(-1)
        invalid=(body[:,0]<=c['preview_body_x_min_m'])|(wrap(heading-pose[:,2]).abs()>=math.pi/2)|numerical_fault
        return dict(command=command,preview_body=body,preview_world=point,lookahead=look,kappa=k,invalid=invalid,numerical_fault=numerical_fault)

    def publish(self,previous,target):
        c=self.config['follower'];dt=self.config['timing']['lower_dt_s']
        limits=dt*torch.tensor([c['nominal_speed_slew_m_s2'],c['nominal_steer_slew_rad_s']],device=previous.device)
        issued=previous+torch.maximum(-limits,torch.minimum(limits,target-previous))
        return issued,(issued-previous)/dt
