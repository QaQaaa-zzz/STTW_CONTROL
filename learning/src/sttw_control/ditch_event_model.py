"""Small contact-event world model over the measured ditch/STTW task.

Ground entry, flight/contact/handback, and post-handback continuation have
separate heads. Predictions are estimates, never physical feasibility proofs.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn


def route_parameters(xy,speed,alpha):
    y=xy[:,1];active=np.flatnonzero(np.abs(y)>1e-5)
    return np.array([y.max(),y.min(),xy[np.abs(y).argmax(),0],xy[active[0],0] if len(active) else 0.,xy[active[-1],0] if len(active) else 0.,speed,alpha],np.float32)


class KnownHeightMap:
    """Sample the actual two-bank triangle mesh, with an explicit open-gap mask."""
    def __init__(self,vertices,spec):
        self.spec=spec;self.banks=[]
        intervals=spec.get('rough_x_intervals_m',[[2.,8.]])
        start=min(x[0] for x in intervals);end=max(x[1] for x in intervals)
        for lo,hi in ((max(-5.,start),min(spec['front_x_m'],end)),(max(start,spec['front_x_m']+spec['width_m']),min(end,spec['ground_end_x_m']))):
            keep=(vertices[:,0]>=np.float32(lo))&(vertices[:,0]<=np.float32(hi))&(np.abs(vertices[:,1])<=(4. if spec.get("full_width_rough",False) else 1.))
            points=vertices[keep];xs,ix=np.unique(points[:,0],return_inverse=True);ys,iy=np.unique(points[:,1],return_inverse=True)
            z=np.full((len(xs),len(ys)),-np.inf,np.float32);np.maximum.at(z,(ix,iy),points[:,2])
            if len(xs)<2 or len(ys)!=(321 if spec.get("full_width_rough",False) else 81) or not np.isfinite(z).all():raise ValueError('Expected the committed regular bank mesh')
            self.banks.append((xs,ys,z))
        xx,yy=np.meshgrid(np.linspace(-.2,1.8,8),np.linspace(-.5,.5,8),indexing='ij')
        self.local=np.stack((xx.ravel(),yy.ravel()),-1)
    def crop(self,root):
        x,y,z,w=root[:,3:7].T
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        cs,sn=np.cos(yaw)[:,None],np.sin(yaw)[:,None]
        qx=root[:,0,None]+cs*self.local[None,:,0]-sn*self.local[None,:,1]
        qy=root[:,1,None]+sn*self.local[None,:,0]+cs*self.local[None,:,1]
        near=self.spec['front_x_m'];far=near+self.spec['width_m']
        valid=(((qx>=-5)&(qx<=near))|((qx>=far)&(qx<=self.spec['ground_end_x_m'])))&(np.abs(qy)<=4)
        if 'land_bridge_y_interval_m' in self.spec:
            lo,hi=self.spec['land_bridge_y_interval_m'];valid|=(qx>=near)&(qx<=far)&(qy>=lo)&(qy<=hi)
        height=np.where(qx>=far,self.spec.get('far_bank_height_m',0.),0.)
        for xs,ys,grid in self.banks:
            ux=(qx-xs[0])/(xs[1]-xs[0]);uy=(qy-ys[0])/(ys[1]-ys[0])
            ix=np.floor(ux).astype(int).clip(0,len(xs)-2);iy=np.floor(uy).astype(int).clip(0,len(ys)-2)
            fx=(ux-ix).clip(0,1);fy=(uy-iy).clip(0,1)
            a,b,c,d=grid[ix,iy],grid[ix+1,iy],grid[ix,iy+1],grid[ix+1,iy+1]
            value=np.where(fx+fy<=1,a+(b-a)*fx+(c-a)*fy,d+(c-d)*(1-fx)+(b-d)*(1-fy))
            inside=(qx>=xs[0])&(qx<=xs[-1])&(qy>=ys[0])&(qy<=ys[-1])
            height=np.where(inside,value,height)
        return np.concatenate((np.where(valid,height*1000,0.),valid.astype(float)),-1).astype(np.float32)


class EventTransitionNetwork(nn.Module):
    def __init__(self,dimension,state_mean,state_std,plan_mean,plan_std,residual_mean,residual_std,conditional_event_targets=False,plan_dimension=10,ground_height_hold=False,shared_event_head=False):
        super().__init__();self.dimension=dimension;self.conditional_event_targets=conditional_event_targets;self.ground_height_hold=ground_height_hold;self.shared_event_head=shared_event_head
        for name,value in dict(state_mean=state_mean,state_std=state_std,plan_mean=plan_mean,plan_std=plan_std,residual_mean=residual_mean,residual_std=residual_std).items():self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.terrain=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.shared=nn.Sequential(nn.Linear(dimension+plan_dimension+6+32,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.heads=nn.ModuleList([nn.Linear(64,2*dimension+2) for _ in range(1 if shared_event_head else 3)])
    def forward(self,state,plan,crop,mode):
        x=torch.cat(((state-self.state_mean)/self.state_std,(plan-self.plan_mean)/self.plan_std,nn.functional.one_hot(mode,6).float(),self.terrain(crop)),-1)
        encoded=self.shared(x);all_heads=torch.stack([head(encoded) for head in self.heads],1)
        head=torch.zeros_like(mode) if self.shared_event_head else torch.where(mode==0,0,torch.where(mode==4,2,1))
        value=all_heads[torch.arange(len(mode),device=mode.device),head]
        d=self.dimension
        return value[:,:d],value[:,d:2*d].clamp(-4,3),value[:,-2],value[:,-1]
    def physical_baseline(self,state,dt,mode=None):
        result=state.clone();result[:,:3]+=state[:,7:10]*dt[:,None]
        if self.ground_height_hold:
            ground=(mode==0)|(mode==4)
            result[:,2]=torch.where(ground,state[:,2],result[:,2])
        return result
    def loss(self,state,plan,crop,mode,next_state,dt,hit):
        mean,logvar,logtime,logit=self(state,plan,crop,mode)
        target=(next_state-self.physical_baseline(state,dt,mode)-self.residual_mean)/self.residual_std
        weights=hit if self.conditional_event_targets else torch.ones_like(hit)
        denominator=weights.sum().clamp_min(1.)
        state_loss=(.5*((mean-target).square()*(-logvar).exp()+logvar).mean(-1)*weights).sum()/denominator
        time_loss=((logtime-torch.log1p(dt)).square()*weights).sum()/denominator
        event_loss=nn.functional.binary_cross_entropy_with_logits(logit,hit)
        return state_loss+time_loss+event_loss


class EventPlanner:
    def __init__(self,directory):
        directory=Path(directory);self.meta=json.loads((directory/'model.json').read_text())
        self.codec=dict(np.load(directory/'codec.npz'));weights=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
        self.model=EventTransitionNetwork(**weights['constructor']);self.model.load_state_dict(weights['state_dict']);self.model.eval()
        self.map=KnownHeightMap(np.load(self.meta['terrain_file'])['vertices'],self.meta['terrain_spec'])
    def encode(self,root,observation,history,road_violation):
        hidden=np.concatenate((observation,history),-1)
        latent=((hidden-self.codec['hidden_mean'])/self.codec['hidden_std'])@self.codec['components'].T
        return np.concatenate((root,latent,np.asarray(road_violation)[:,None]),-1).astype(np.float32)
    def plan_features(self,state,parameters):
        spec=self.meta['terrain_spec'];x=state[:,0]
        return np.column_stack((parameters,spec['front_x_m']-x,spec['front_x_m']+spec['width_m']-x,self.meta['goal']['x_min_m']-x)).astype(np.float32)
    @torch.no_grad()
    def score(self,state,parameters,particles=16,seed=20261110):
        # All particles are model futures; no measured future is queried.
        rng=np.random.default_rng(seed);n=len(state);s=np.repeat(state,particles,0);params=np.repeat(parameters,particles,0)
        probability=np.ones(len(s));elapsed=np.zeros(len(s));events=[]
        jump=np.ones(n,dtype=bool) if parameters.shape[1]==7 else parameters[:,-1]>.5
        jump_particles=np.repeat(jump,particles)
        for mode in range(5):
            active=jump_particles if mode<4 else np.ones(len(s),dtype=bool)
            inp=torch.from_numpy(s);plan=torch.from_numpy(self.plan_features(s,params));crop=torch.from_numpy(self.map.crop(s[:,:13]))
            mean,logvar,logtime,logit=self.model(inp,plan,crop,torch.full((len(s),),mode,dtype=torch.long))
            dt=torch.expm1(logtime.clamp(0,np.log(11.)))
            noise_shape=(particles,mean.shape[1]) if self.meta.get('inference_common_random_numbers',False) else mean.shape
            noise_np=rng.standard_normal(noise_shape).astype(np.float32)
            noise=torch.from_numpy(np.tile(noise_np,(n,1)) if self.meta.get('inference_common_random_numbers',False) else noise_np)
            residual=(mean+noise*torch.exp(.5*logvar))*self.model.residual_std+self.model.residual_mean
            nxt=(self.model.physical_baseline(inp,dt,torch.full((len(s),),mode,dtype=torch.long))+residual).numpy()
            q=nxt[:,3:7];nxt[:,3:7]=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
            nxt[:,-1]=nxt[:,-1].clip(0,1)
            if self.model.conditional_event_targets:nxt[:,-1]=np.maximum(s[:,-1],nxt[:,-1])
            nxt=np.where(active[:,None],nxt,s)
            probability*=np.where(active,torch.sigmoid(logit).numpy(),1.);elapsed+=np.where(active,dt.numpy(),0.)
            events.append(dict(mode=mode,active_candidates=(jump if mode<4 else np.ones(n,dtype=bool)).tolist(),mean_root=nxt[:,:13].reshape(n,particles,13).mean(1).tolist(),mean_event_probability=torch.sigmoid(logit).numpy().reshape(n,particles).mean(1).tolist()))
            s=nxt.astype(np.float32)
            if mode==0:entry=s.copy()
        mean,var,logtime,logit=self.model(torch.from_numpy(entry),torch.from_numpy(self.plan_features(entry,params)),torch.from_numpy(self.map.crop(entry[:,:13])),torch.full((len(entry),),5,dtype=torch.long))
        events.append(dict(mode=5,active_candidates=jump.tolist(),name='source_scorer_clearance_auxiliary',mean_event_probability=torch.sigmoid(logit).numpy().reshape(n,particles).mean(1).tolist(),mean_delay_s=torch.expm1(logtime.clamp(0,np.log(11.))).numpy().reshape(n,particles).mean(1).tolist()))
        # Road history is an absorbing task condition, separate from survival.
        probability*=1-s[:,-1]
        q=probability.reshape(n,particles).mean(1)-.001*elapsed.reshape(n,particles).mean(1)
        return dict(score=q.tolist(),predicted_completion=probability.reshape(n,particles).mean(1).tolist(),predicted_time_s=elapsed.reshape(n,particles).mean(1).tolist(),events=events,scope='Learned event rollout estimate, not a feasibility certificate; physical execution decides completion')
