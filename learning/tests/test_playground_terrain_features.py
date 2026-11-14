import unittest
import numpy as np
import torch
from sttw_control.playground_terrain_features import TerrainFeatures

class TerrainFeatureTests(unittest.TestCase):
    def test_plane_gradient_and_rotated_queries(self):
        x=np.linspace(-2,2,81);y=x.copy();xx,yy=np.meshgrid(x,y)
        terrain=TerrainFeatures(.1*xx+.2*yy,np.ones_like(xx),[-2,2,-2,2],.05)
        pose=torch.zeros(2,12);pose[0,3]=1;pose[1,3]=pose[1,6]=2**-.5
        values=terrain(pose).reshape(2,15,5)
        self.assertTrue(torch.allclose(values[:,:,1],torch.full((2,15),.1),atol=1e-5))
        self.assertTrue(torch.allclose(values[:,:,2],torch.full((2,15),.2),atol=1e-5))
        ox,oy=terrain.offsets.T
        self.assertTrue(torch.allclose(values[0,:,0],.1*ox+.2*oy,atol=1e-5))
        self.assertTrue(torch.allclose(values[1,:,0],-.1*oy+.2*ox,atol=1e-5))
    def test_hole_is_not_flat_supported_ground(self):
        height=np.zeros((81,81),np.float32);mask=np.ones_like(height);mask[30:51,30:51]=0
        terrain=TerrainFeatures(height,mask,[-2,2,-2,2],.05)
        pose=torch.zeros(1,12,requires_grad=True);q=pose+torch.tensor([[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.]])
        values=terrain(q).reshape(1,15,5)
        self.assertEqual(float(values[0,4,4].detach()),0.)
        values.sum().backward();self.assertTrue(torch.isfinite(pose.grad).all())

if __name__=='__main__':unittest.main()
