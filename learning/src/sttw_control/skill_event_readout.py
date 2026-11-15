"""Causal cumulative event readout over frozen predicted skill states."""
import numpy as np
import torch
from torch import nn


def event_labels(times,event_times):
    """A5ms event stays visible on later20ms samples; no interpolation of motion."""
    events=np.asarray(event_times,dtype=float)
    return ((np.asarray(times)[:,None]>=events[None,:])&np.isfinite(events)[None,:]).astype('float32')


class SkillEventReadout(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.network=nn.Sequential(nn.Linear(len(mean),64),nn.SiLU(),nn.Linear(64,2))

    def forward(self,features):
        return self.network((features-self.mean)/self.scale)

    def predict(self,features,previous_probability):
        probability=self(features).sigmoid().cummax(dim=1).values
        probability=torch.maximum(probability,previous_probability[:,None,:])
        return probability,probability[:,-1]


class EventWorldModel(nn.Module):
    """Frozen physical world model plus causal event memory; no control action."""
    def __init__(self,world,readout):
        super().__init__()
        self.world=world
        self.readout=readout

    def initialize(self,*args):
        state=self.world.initialize(*args)
        return dict(world=state,event_probability=state.qpos.new_zeros(len(state.qpos),2))

    def rollout(self,state,context,steps):
        physical=state['world'];previous=state['event_probability']
        poses=[physical.qpos];velocities=[physical.qvel];events=[];phases=[]
        for _ in range(steps):
            physical,_,phase=self.world.step(physical,context)
            age=torch.as_tensor(physical.elapsed_steps,device=physical.qpos.device,dtype=physical.qpos.dtype).expand(len(physical.qpos))[:,None]*self.world.dt
            features=torch.cat((physical.qpos,physical.qvel,physical.hidden,age),-1)
            probability,previous=self.readout.predict(features[:,None,:],previous)
            poses.append(physical.qpos);velocities.append(physical.qvel);events.append(probability[:,0]);phases.append(phase)
        return dict(qpos=torch.stack(poses,1),qvel=torch.stack(velocities,1),event_probability=torch.stack(events,1),phase_logits=torch.stack(phases,1),duration_s=steps*self.world.dt,next_state=dict(world=physical,event_probability=previous))
