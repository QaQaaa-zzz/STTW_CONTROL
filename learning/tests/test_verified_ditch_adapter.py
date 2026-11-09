"""CPU checks of solved-contact routing and conservative mesh geometry."""
import unittest
import numpy as np
import torch
from sttw_control.ditch_jump_adapter import BankEnvelope, classify_patches
from sttw_control.ditch_terrain import build_ditch_terrain


class WiringTests(unittest.TestCase):
    def test_many_valid_contacts_use_actual_allocated_capacity(self):
        n=100
        point=torch.tensor([[3.2,0.,0.]]).repeat(n,1)
        raw=(torch.ones(n,1),point,torch.tensor([[0.,0.,1.]]).repeat(n,1),None,
             torch.tensor([[n]]),torch.tensor([[0]]))
        args=(raw,torch.zeros(1,3),torch.tensor([.07]),3.3,3.6,8.,1.)
        self.assertFalse(classify_patches(*args,64)[0].item())
        self.assertTrue(classify_patches(*args,256)[0].item())
        raw[-2][0,0]=256
        self.assertFalse(classify_patches(*args,256)[0].item())

    def test_patch_support_must_stay_on_one_bank_and_in_requested_strip(self):
        # Local coordinates: lanes 0/1 valid, lane 2 touches both banks, lane 3
        # straddles the smooth shoulder, lane 4 has no registered terrain load.
        n=5
        origin=torch.tensor([[0.,0.,0.],[20.,3.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        point=torch.tensor([[3.2,0.,0.],[23.7,3.,0.],[3.2,0.,0.],
                            [3.7,0.,0.],[3.7,.95,0.],[3.7,1.02,0.]])
        count=torch.tensor([[1],[1],[2],[2],[0]])
        start=torch.tensor([[0],[1],[2],[4],[0]])
        normal=torch.tensor([[0.,0.,1.]]).repeat(len(point),1)
        raw=(torch.full((len(point),1),5.),point,normal,None,count,start)
        near,far=classify_patches(raw,origin,torch.full((n,),.07),3.3,3.6,8.,1.,16)
        self.assertEqual(near.tolist(),[True,False,False,False,False])
        self.assertEqual(far.tolist(),[False,True,False,False,False])
        # An exhausted or malformed solver buffer cannot certify a landing.
        count[1]=16
        self.assertFalse(classify_patches(raw,origin,torch.full((n,),.07),3.3,3.6,8.,1.,16)[1][1])

    def test_rough_envelope_and_virtual_gap(self):
        spec=dict(front_x_m=3.3,width_m=.3,ground_end_x_m=25.,
                  stones_per_m2=30,seed=20261109,stone_height_min_m=.01,stone_height_max_m=.04)
        vertices,_=build_ditch_terrain(spec)
        bounds=BankEnvelope(vertices,spec,.07,.025,'cpu')
        peak=vertices[np.argmax(vertices[:,2])]
        points=torch.tensor([[[peak[0],peak[1],.20],[3.45,0.,.20]],
                             [[3.45,2.,.20],[10.,0.,.20]]])
        clearance=bounds.clearance(points)
        self.assertLessEqual(float(clearance[0,0]),.20-bounds.bound-float(peak[2])+1e-6)
        np.testing.assert_allclose(clearance.numpy()[[0,1,1],[1,0,1]],.20-bounds.bound,atol=1e-6)
        # Local cylinder Z horizontal (upright wheel) or vertical (sideways).
        q=torch.tensor([[[.5**.5,0.,0.,.5**.5],[0.,0.,0.,1.]]]*2)
        high=bounds.clearance_upper(points,q)
        np.testing.assert_allclose(high.numpy(),[[.13,.175]]*2,atol=1e-6)
        self.assertTrue(bool((high>=clearance).all()))


if __name__=='__main__':
    unittest.main()
