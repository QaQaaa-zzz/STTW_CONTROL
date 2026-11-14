import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sttw_control.playground_rolling import RollingMotionLibrary

class RollingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        path=Path(self.tmp.name)/'bank.npz'
        np.savez(path,x=np.arange(4,dtype=np.float32),low=np.full((2,4,22),-1,np.float32),high=np.ones((2,4,22),np.float32),valid=np.ones((2,4),bool),phase_low=np.zeros((2,4),np.float32),phase_high=np.zeros((2,4),np.float32),edges=np.array([0,4],np.float32),margin=np.ones((1,22),np.float32),routes=np.array([0,1]),activation=np.array([2.5,2.7],np.float32),cost=np.array([2,1],np.float32))
        xy=torch.stack((torch.stack((torch.arange(4),torch.zeros(4)),1),torch.tensor([[0.,0.],[1.,0.],[2.,0.],[3.,1.]]))).float()
        self.route=SimpleNamespace(xy=xy,heading=torch.zeros(2,4),curvature=torch.zeros(2,4),s=torch.arange(4).float(),route_ids=torch.tensor([0]))
        self.planner=RollingMotionLibrary(path,self.route,[1.,1.],'cpu')
        self.q=torch.zeros(1,12);self.q[:,0]=.5;self.v=torch.zeros(1,11);self.h=torch.zeros(1,2)
    def select(self,required=1.,phase=0):
        return self.planner.select(self.q,self.v,self.h,torch.tensor([phase]),torch.tensor([required]),self.route,torch.tensor([2.5]))
    def test_prefix_excludes_cheaper_future_change(self):
        chosen,valid,_=self.select();self.assertEqual(chosen.item(),1);self.assertTrue(valid.item())
        chosen,valid,_=self.select(required=3.);self.assertEqual(chosen.item(),0);self.assertTrue(valid.item())
    def test_actual_history_outside_envelope_rejects_all(self):
        self.h[:]=5
        _,valid,_=self.select();self.assertFalse(valid.item())
    def test_activation_cannot_change_after_jump(self):
        self.planner.phase_low[:]=2;self.planner.phase_high[:]=2
        chosen,valid,_=self.select(phase=2);self.assertEqual(chosen.item(),0);self.assertTrue(valid.item())
    def test_rejected_proposal_keeps_original_route(self):
        route,ids=self.planner.proposal(self.route,torch.tensor([1]),torch.tensor([False]))
        self.assertTrue(torch.equal(route.xy,self.route.xy));self.assertEqual(ids.item(),0)

if __name__=='__main__':unittest.main()
