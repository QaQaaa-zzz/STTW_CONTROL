"""Fixed-speed ground CEM: shared splines and budget for geometry/dynamics.

The solver returns feasibility explicitly. A least-violating candidate is useful
for diagnostics/search, but is never labelled a publishable plan.
"""
import numpy as np
import torch
from scipy.interpolate import BSpline
from sttw_control.playground_prediction import euler


def feasibility_order(violation,cost):
    order=torch.argsort(cost,stable=True)
    return order[torch.argsort(violation[order],stable=True)]


class SplineRoutes:
    """Six free cubic B-spline controls, with fixed C2 straight end pieces."""
    def __init__(self,start_x,end_x,exit_y=0.,device='cuda',dtype=torch.float32):
        x=np.linspace(-5,26,1551);t=np.clip((x-start_x)/(end_x-start_x),0,1)
        knots=np.r_[np.zeros(4),np.linspace(0,1,10)[1:-1],np.ones(4)]
        basis=BSpline.design_matrix(t,knots,3).toarray()[:,3:-3]
        tensor=lambda v:torch.as_tensor(v,device=device,dtype=dtype)
        self.start_x=start_x;self.x=tensor(x);self.basis=tensor(basis);self.base=tensor(exit_y*(10*t**3-15*t**4+6*t**5));self.s=tensor(np.linspace(0,30,1501));self.ds=.02

    def __call__(self,parameters):
        y=parameters@self.basis.T+self.base
        x=self.x.expand_as(y);arc=torch.cat((torch.zeros_like(y[:,:1]),torch.hypot(x.diff(dim=1),y.diff(dim=1)).cumsum(1)),1)
        query=self.s.expand(len(y),-1).contiguous();i=torch.searchsorted(arc.contiguous(),query,right=True).sub(1).clamp(0,y.shape[1]-2)
        left=arc.gather(1,i);f=((query-left)/(arc.gather(1,i+1)-left)).clamp(0,1)
        interp=lambda value:value.gather(1,i)*(1-f)+value.gather(1,i+1)*f
        xy=torch.stack((interp(x),interp(y)),2)
        prefix=self.s<=self.start_x+5
        xy=torch.stack((torch.where(prefix[None,:],self.s[None,:]-5,xy[:,:,0]),torch.where(prefix[None,:],torch.zeros_like(xy[:,:,1]),xy[:,:,1])),2)
        dx=torch.gradient(xy[:,:,0],spacing=(self.ds,),dim=(1,))[0];dy=torch.gradient(xy[:,:,1],spacing=(self.ds,),dim=(1,))[0]
        heading=torch.atan2(dy,dx);curvature=torch.gradient(heading,spacing=(self.ds,),dim=(1,))[0]
        return xy,heading,curvature


class GroundCEM:
    def __init__(self,routes,predictor,envelope,contract,method):
        self.routes=routes;self.predictor=predictor;self.envelope=envelope;self.contract=contract;self.method=method
        c=contract;self.violation_scales=torch.tensor([c[k] for k in ['margin_m','roll_limit_rad','pitch_limit_rad','min_speed_m_s','curvature_limit_per_m','tail_distance_m','exit_y_tolerance_m','exit_heading_tolerance_rad','exit_speed_tolerance_m_s']],device=routes.s.device,dtype=routes.s.dtype)

    def evaluate(self,parameters,pose,xi,previous,bounds,obstacles):
        c=self.contract;xy,h,k=self.routes(parameters);n=len(parameters);steps=c['steps']
        start=pose.expand(n,-1);velocity=xi.expand(n,-1);prior=previous.expand(n,-1)
        if self.method=='dynamic':
            prediction,speeds=self.predictor(start,velocity,prior,self.routes.s,xy,h,k,steps)
        else:
            nearest=(xy-start[:,None,:2]).square().sum(-1).argmin(1)
            distance=self.routes.s[nearest,None]+torch.arange(steps+1,device=pose.device,dtype=pose.dtype)[None,:]*.036
            i=torch.searchsorted(self.routes.s,distance.contiguous(),right=True).sub(1).clamp(0,len(self.routes.s)-2)
            f=((distance-self.routes.s[i])/.02).clamp(0,1)
            interp=lambda a:a.gather(1,i)*(1-f)+a.gather(1,i+1)*f
            yaw=interp(h);zero=torch.zeros_like(yaw);q=torch.stack(((yaw*.5).cos(),zero,zero,(yaw*.5).sin()),2)
            prediction=torch.cat((torch.stack((interp(xy[:,:,0]),interp(xy[:,:,1]),start[:,2:3].expand_as(yaw)),2),q,
                torch.atan(c['wheelbase_m']*interp(k))[:,:,None],start[:,None,8:].expand(-1,steps+1,-1)),2)
            speeds=velocity[:,None,:].expand(-1,steps+1,-1).clone();speeds[:,:,0]=1.8
        low,high=self.envelope(prediction)
        road=torch.stack((low[:,:,:,0]-bounds[0],bounds[1]-high[:,:,:,0],low[:,:,:,1]-bounds[2],bounds[3]-high[:,:,:,1]),-1).amin((1,2,3))
        separation=torch.stack((obstacles[None,None,None,:,0]-high[:,:,:,None,0],low[:,:,:,None,0]-obstacles[None,None,None,:,1],
            obstacles[None,None,None,:,2]-high[:,:,:,None,1],low[:,:,:,None,1]-obstacles[None,None,None,:,3]),-1).amax(-1).amin((1,2,3))
        clearance=torch.minimum(road,separation);angles=euler(prediction[:,:,3:7].reshape(-1,4)).reshape(n,steps+1,3)
        errors=torch.stack(((c['margin_m']-clearance).clamp_min(0),
            (angles[:,:,0].abs().amax(1)-c['roll_limit_rad']).clamp_min(0),
            (angles[:,:,1].abs().amax(1)-c['pitch_limit_rad']).clamp_min(0),
            (c['min_speed_m_s']-speeds[:,:,0].amin(1)).clamp_min(0),
            (k.abs().amax(1)-c['curvature_limit_per_m']).clamp_min(0),
            (c['exit_x_m']+c['tail_distance_m']-prediction[:,-1,0]).clamp_min(0),
            ((prediction[:,-1,1]-c['exit_y_m']).abs()-c['exit_y_tolerance_m']).clamp_min(0),
            (angles[:,-1,2].abs()-c['exit_heading_tolerance_rad']).clamp_min(0),
            ((speeds[:,-1,0]-1.8).abs()-c['exit_speed_tolerance_m_s']).clamp_min(0)),1)
        violation=torch.where(torch.isfinite(prediction).all((1,2)),(errors/self.violation_scales).sum(1),torch.full_like(clearance,float('inf')))
        # Fixed world endpoints: all routes having 30 m of stored arc is not a cost.
        def arc_at_x(x):
            query=torch.full_like(xy[:,:1,0],x) if isinstance(x,float) else x[:,None]
            i=torch.searchsorted(xy[:,:,0].contiguous(),query.contiguous(),right=True).sub(1).clamp(0,xy.shape[1]-2)
            left=xy[:,:,0].gather(1,i);f=((query-left)/(xy[:,:,0].gather(1,i+1)-left)).clamp(0,1)
            return (self.routes.s[i]+f*self.routes.ds).squeeze(1)
        path_length=arc_at_x(float(c['exit_x_m']+c['tail_distance_m']))-arc_at_x(start[:,0])
        cost=path_length+.1*k.square().sum(1)*self.routes.ds
        return xy,h,k,prediction,violation,cost,clearance,errors

    def __call__(self,pose,xi,previous,bounds,obstacles,noise,initial_mean,initial_std):
        c=self.contract;mean=initial_mean;std=initial_std;parameters=[];results=[]
        for iteration in range(c['iterations']):
            samples=(mean+std*noise[iteration]).clamp(-c['parameter_limit_m'],c['parameter_limit_m'])
            samples[0]=mean;samples[1]=0.  # Mean and valid straight incumbent share the same 64-candidate budget.
            result=self.evaluate(samples,pose,xi,previous,bounds,obstacles)
            elite=feasibility_order(result[4],result[5])[:c['elites']]
            mean=samples[elite].mean(0);std=samples[elite].std(0,unbiased=False).clamp_min(c['minimum_std_m'])
            parameters.append(samples);results.append(result)
        values=[torch.cat([r[j] for r in results],0) for j in range(8)]
        selected=feasibility_order(values[4],values[5])[0]
        return torch.cat(parameters).index_select(0,selected.reshape(1)).squeeze(0),*[value.index_select(0,selected.reshape(1)).squeeze(0) for value in values]
