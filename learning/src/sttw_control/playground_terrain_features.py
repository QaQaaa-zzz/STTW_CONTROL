"""Differentiable local height/support descriptors for a prediction model.

This raster is a model input; collision checking keeps the physical geometry.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .playground_prediction import euler
from .ditch_terrain import build_ditch_terrain


class TerrainFeatures(nn.Module):
    def __init__(self,height,support,bounds,spacing,device='cpu'):
        super().__init__()
        height=torch.as_tensor(height,dtype=torch.float32,device=device)
        support=torch.as_tensor(support,dtype=torch.float32,device=device)
        gy,gx=torch.gradient(height,spacing=(float(spacing),float(spacing)))
        padded=F.pad(height[None,None],(1,1,1,1),mode='replicate')
        mean=F.avg_pool2d(padded,3,stride=1)[0,0]
        variance=(F.avg_pool2d(padded.square(),3,stride=1)[0,0]-mean.square()).clamp_min(0)
        variation=(variance+1e-8).sqrt()
        self.register_buffer('field',torch.stack((height,gx,gy,variation,support))[None])
        self.register_buffer('bounds',torch.as_tensor(bounds,dtype=torch.float32,device=device))
        x,y=torch.meshgrid(torch.tensor([-.4,0.,.4,.8,1.2],device=device),torch.tensor([-.2,0.,.2],device=device),indexing='ij')
        self.register_buffer('offsets',torch.stack((x.flatten(),y.flatten()),1))
        self.width=self.offsets.shape[0]*self.field.shape[1]

    def forward(self,pose):
        yaw=euler(pose[:,3:7])[:,2];cs=yaw.cos()[:,None];sn=yaw.sin()[:,None]
        x=pose[:,0:1]+cs*self.offsets[:,0]-sn*self.offsets[:,1]
        y=pose[:,1:2]+sn*self.offsets[:,0]+cs*self.offsets[:,1]
        grid=torch.stack((2*(x-self.bounds[0])/(self.bounds[1]-self.bounds[0])-1,2*(y-self.bounds[2])/(self.bounds[3]-self.bounds[2])-1),-1)
        sampled=F.grid_sample(self.field,grid.reshape(1,-1,1,2),align_corners=True,padding_mode='zeros')
        return sampled[0,:,:,0].T.reshape(len(pose),-1)


def raster_from_spec(spec,spacing=.025):
    """Sample original fixed patches, explicitly retaining the trench support mask."""
    _,_,patches=build_ditch_terrain(spec,return_patches=True)
    bounds=[-5.,spec['ground_end_x_m'],-4.,4.]
    xs=np.arange(bounds[0],bounds[1]+spacing*.5,spacing);ys=np.arange(bounds[2],bounds[3]+spacing*.5,spacing)
    height=np.zeros((len(ys),len(xs)),np.float32);support=np.zeros_like(height)
    for px,py,pz in patches:
        ix=np.flatnonzero((xs>=px[0]-1e-6)&(xs<=px[-1]+1e-6));iy=np.flatnonzero((ys>=py[0]-1e-6)&(ys<=py[-1]+1e-6))
        along=np.stack([np.interp(xs[ix],px,pz[:,j]) for j in range(len(py))],1)
        values=np.stack([np.interp(ys[iy],py,row) for row in along],1)
        height[np.ix_(iy,ix)]=values;support[np.ix_(iy,ix)]=1
    return dict(height=height,support=support,bounds=np.array(bounds),spacing=spacing)
