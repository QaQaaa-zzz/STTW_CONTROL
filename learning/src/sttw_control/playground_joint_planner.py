"""Finite full-motion planning on a fixed map, with a segmented comparator."""
import torch


class MotionLibraryPlanner:
    def __init__(self,lower,upper,valid,prefix,labels,cost,prefix_cost,eligible,margin):
        self.lower=lower;self.upper=upper;self.valid=valid;self.prefix=prefix
        self.labels=labels;self.cost=cost;self.prefix_cost=prefix_cost
        self.eligible=eligible;self.margin=margin

    def __call__(self,bounds,boxes):
        low,high=self.lower,self.upper
        road=torch.stack((low[:,:,0]-bounds[0],bounds[1]-high[:,:,0],low[:,:,1]-bounds[2],bounds[3]-high[:,:,1]),-1).amin(-1)
        separation=torch.maximum(boxes[None,None,:,0:3]-high[:,:,None,:],low[:,:,None,:]-boxes[None,None,:,3:6]).amax(-1).amin(-1)
        clearance=torch.minimum(road,separation)
        full_clear=torch.where(self.valid,clearance,torch.inf).amin(1)
        prefix_clear=torch.where(self.valid&self.prefix,clearance,torch.inf).amin(1)
        feasible=(full_clear>=self.margin)&self.eligible
        # H0 solves its first segment against the canonical straight downstream.
        prefix_ok=(prefix_clear>=self.margin)&self.eligible&(self.labels[:,2]==0)
        first=torch.where(prefix_ok,self.prefix_cost,torch.inf).argmin()
        conditional=feasible&(self.labels[:,:2]==self.labels[first,:2]).all(1)
        sequential_ok=prefix_ok.any()&conditional.any()
        sequential=torch.where(conditional,self.cost,torch.inf).argmin()
        joint=torch.where(feasible,self.cost,torch.inf).argmin()
        # No-plan retains the incumbent; never publish a least-violating motion.
        indices=torch.stack((torch.where(sequential_ok,sequential,torch.zeros_like(first)),torch.where(feasible.any(),joint,torch.zeros_like(first))))
        return indices,torch.stack((sequential_ok,feasible.any())),full_clear,prefix_clear,first
