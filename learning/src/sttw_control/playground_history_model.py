"""History-initialized nominal dynamics correction; never controls the robot."""
import torch
from torch import nn
from .playground_prediction import multiply,rotate


class HistoryDynamics(nn.Module):
    def __init__(self,nominal,controller_mean,controller_scale,input_mean,input_scale,correction_scale,terrain=None,model_kind="residual"):
        super().__init__()
        self.nominal=nominal;self.terrain=terrain;self.model_kind=model_kind
        for name,value in [('controller_mean',controller_mean),('controller_scale',controller_scale),
                           ('input_mean',input_mean),('input_scale',input_scale),('correction_scale',correction_scale)]:
            self.register_buffer(name,torch.as_tensor(value,device=nominal.mean.device,dtype=nominal.mean.dtype))
        self.initial=nn.Sequential(nn.Linear(len(controller_mean),128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.memory=nn.GRUCell(len(input_mean),64)
        self.correction=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,9))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
        self.to(nominal.mean.device)

    def rollout(self,pose,xi,previous,controller,s,xy,heading,curvature,steps,speed_request=None):
        hidden=self.initial((controller-self.controller_mean)/self.controller_scale)
        poses=[pose];velocities=[xi]
        for _ in range(steps):
            eta=self.nominal.features(pose,xi,s,xy,heading,curvature)
            if speed_request is not None:eta[:,13]=speed_request
            features=torch.cat((xi,eta,pose[:,10:12]),1)
            if self.terrain is not None:features=torch.cat((features,self.terrain(pose)),1)
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
            previous,xi=xi,nxt;poses.append(pose);velocities.append(xi)
        return torch.stack(poses,1),torch.stack(velocities,1)


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
        ['controller_mean','controller_scale','input_mean','input_scale','correction_scale']],terrain=terrain,model_kind=contract.get('model_kind','residual'))
    model.load_state_dict(state);model.eval()
    return model
