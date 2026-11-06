"""Causal wheel-contact event state for frozen-policy branch experiments."""
import math
import torch


class WheelClearanceEvents:
    """Latch clearance only after observed ridge contacts and rear exit/support.

    Inputs refer to the just-solved physics step. ``rear_beyond_support`` must
    include wheel geometry, not merely chassis route progress. Call
    ``take_ready`` at the following50Hz decision boundary, never inside a step.
    """
    def __init__(self,count,device,physics_dt,confirmation_s=.02):
        self.dt=physics_dt
        self.required=math.ceil(confirmation_s/physics_dt)
        self.step=0
        self.first_contact=torch.full((count,2),-1,device=device,dtype=torch.long)
        self.last_contact=torch.full_like(self.first_contact,-1)
        self.quiet=torch.zeros(count,device=device,dtype=torch.long)
        self.confirmed=torch.full((count,),-1,device=device,dtype=torch.long)
        self.dispatched=torch.zeros(count,device=device,dtype=torch.bool)

    def update(self,ridge_contact,rear_beyond_support,rear_ground_loaded,alive):
        self.step+=1
        contact=ridge_contact&alive[:,None]
        self.first_contact=torch.where(contact&(self.first_contact<0),self.step,self.first_contact)
        self.last_contact=torch.where(contact,self.step,self.last_contact)
        self.quiet=torch.where(~contact[:,1]&alive,self.quiet+1,0)
        seen=(self.first_contact>=0).all(1)
        ready=seen&rear_beyond_support&rear_ground_loaded&alive&(self.quiet>=self.required)
        self.confirmed=torch.where(ready&(self.confirmed<0),self.step,self.confirmed)

    def take_ready(self,alive):
        ready=(self.confirmed>=0)&~self.dispatched&alive
        self.dispatched|=ready
        return ready
