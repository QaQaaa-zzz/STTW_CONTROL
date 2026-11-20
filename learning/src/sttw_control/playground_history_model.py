"""History-initialized nominal dynamics correction; never controls the robot."""
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class GroundPredictionState:
    """Predicted ground state; hidden is model memory, not real controller history."""
    pose: torch.Tensor
    velocity: torch.Tensor
    previous_velocity: torch.Tensor
    hidden: torch.Tensor
    elapsed_steps: int = 0

from .playground_prediction import multiply,rotate


class HistoryDynamics(nn.Module):
    def __init__(self,nominal,controller_mean,controller_scale,input_mean,input_scale,correction_scale,terrain=None,model_kind="residual",alpha_conditioned=False):
        super().__init__()
        self.nominal=nominal;self.terrain=terrain;self.model_kind=model_kind;self.alpha_conditioned=alpha_conditioned
        for name,value in [('controller_mean',controller_mean),('controller_scale',controller_scale),
                           ('input_mean',input_mean),('input_scale',input_scale),('correction_scale',correction_scale)]:
            self.register_buffer(name,torch.as_tensor(value,device=nominal.mean.device,dtype=nominal.mean.dtype))
        self.initial=nn.Sequential(nn.Linear(len(controller_mean),128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.memory=nn.GRUCell(len(input_mean),64)
        self.correction=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,9))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
        self.to(nominal.mean.device)

    def initialize(self, pose, velocity, previous_velocity, controller):
        return GroundPredictionState(pose,velocity,previous_velocity,
            self.initial((controller-self.controller_mean)/self.controller_scale))

    def rollout_state(self,state,s,xy,heading,curvature,steps,speed_request=None,terrain_ids=None,alpha_request=None):
        pose,velocity,domain,next_state=self.rollout(state.pose,state.velocity,state.previous_velocity,None,
            s,xy,heading,curvature,steps,speed_request,terrain_ids,True,state.hidden,True,alpha_request)
        next_state.elapsed_steps=state.elapsed_steps+steps
        return dict(pose=pose,velocity=velocity,domain=domain,duration_s=steps*.02,next_state=next_state)

    def step(self,pose,xi,previous,hidden,s,xy,heading,curvature,speed_request,terrain_ids,alpha_request):
        eta=self.nominal.features(pose,xi,s,xy,heading,curvature)
        if speed_request is not None:eta[:,13]=speed_request
        features=torch.cat((xi,eta,pose[:,10:12]),1)
        if self.terrain is not None:features=torch.cat((features,self.terrain(pose,terrain_ids)),1)
        if self.alpha_conditioned:features=torch.cat((features,alpha_request[:,None]),1)
        hidden=self.memory((features-self.input_mean)/self.input_scale,hidden)
        x=torch.cat((xi,previous,eta),1)
        if self.model_kind=="direct":nxt=xi
        else:
            nxt=((x-self.nominal.mean)/self.nominal.scale)@self.nominal.weight+self.nominal.bias
            if self.nominal.identity:nxt=nxt+xi
        nxt=nxt+self.correction(hidden)*self.correction_scale
        mid=(xi+nxt)*.5;rv=mid[:,3:6]*.02;angle=rv.norm(dim=-1,keepdim=True)
        dq=torch.cat(((angle*.5).cos(),rv*.5*torch.sinc(angle/(2*torch.pi))),1)
        q=multiply(pose[:,3:7],dq);q=q/q.norm(dim=-1,keepdim=True)
        xyz=pose[:,:3]+.01*(rotate(pose[:,3:7],xi[:,:3])+rotate(q,nxt[:,:3]))
        pose=torch.cat((xyz,q,pose[:,7:10]+mid[:,6:9]*.02,pose[:,10:12]),1)
        return pose,nxt,hidden,eta[:,9:12]

    def rollout(self,pose,xi,previous,controller,s,xy,heading,curvature,steps,speed_request=None,terrain_ids=None,return_domain=False,hidden=None,return_state=False,alpha_request=None):
        if not self.alpha_conditioned and alpha_request is not None:raise ValueError("Unconditioned ground model does not support alpha queries")
        if self.alpha_conditioned and alpha_request is None:raise ValueError("Conditional ground prediction requires explicit stage alpha")
        if hidden is None:hidden=self.initial((controller-self.controller_mean)/self.controller_scale)
        poses=[pose];velocities=[xi];domain=[]
        for _ in range(steps):
            pose,nxt,hidden,route_domain=self.step(pose,xi,previous,hidden,s,xy,heading,curvature,speed_request,terrain_ids,alpha_request)
            if return_domain:domain.append(route_domain)
            previous,xi=xi,nxt;poses.append(pose);velocities.append(xi)
        if return_domain:
            domain.append(self.nominal.features(pose,xi,s,xy,heading,curvature)[:,9:12])
            result=(torch.stack(poses,1),torch.stack(velocities,1),torch.stack(domain,1))
        else:result=(torch.stack(poses,1),torch.stack(velocities,1))
        if return_state:return (*result,GroundPredictionState(pose,xi,previous,hidden,steps))
        return result


def load_history_model(root,checkpoint='best.pt',device='cuda'):
    import json
    import numpy as np
    from .playground_prediction import NominalRollout
    contract=json.loads((root/'contract.json').read_text())
    state=torch.load(root/checkpoint,map_location=device,weights_only=True)
    nominal=NominalRollout(dict(np.load(root/'nominal.npz')),contract['follower'],device)
    terrain=None
    if contract.get("terrain_features"):
        from .playground_terrain_features import TerrainFeatures
        terrain=TerrainFeatures(**dict(np.load(root/"terrain.npz")),device=device)
    model=HistoryDynamics(nominal,*[state[key] for key in
        ['controller_mean','controller_scale','input_mean','input_scale','correction_scale']],terrain=terrain,model_kind=contract.get('model_kind','residual'),alpha_conditioned=contract.get('alpha_conditioned',False))
    model.load_state_dict(state);model.eval()
    return model
