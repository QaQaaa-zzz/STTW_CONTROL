"""Decode estimated execution histories from frozen ground-model predictions."""
import torch
from torch import nn


class SkillHistoryBridge(nn.Module):
    def __init__(self,mean,scale,target_mean,target_scale):
        super().__init__()
        for name,value in [('mean',mean),('scale',scale),('target_mean',target_mean),('target_scale',target_scale)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.network=nn.Sequential(nn.Linear(len(mean),256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,len(target_mean)))

    def forward(self,features):
        return self.network((features-self.mean)/self.scale)*self.target_scale+self.target_mean


def bridge_features(state,initial_history):
    age=torch.as_tensor(state.elapsed_steps,device=state.pose.device,dtype=state.pose.dtype).expand(len(state.pose))[:,None]*.02
    return torch.cat((state.pose,state.velocity,state.hidden,age,initial_history),-1)
