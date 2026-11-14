import unittest
import numpy as np
import torch
from sttw_control.playground_planner import SplineRoutes,feasibility_order

class PlannerTest(unittest.TestCase):
    def test_feasible_precedes_cheap_collision(self):
        order=feasibility_order(torch.tensor([.1,0.,0.,.01]),torch.tensor([0.,10.,5.,1.]))
        self.assertEqual(order.tolist(),[2,1,3,0])

    def test_cubic_routes_preserve_straight_prefix_and_arc(self):
        routes=SplineRoutes(4.,8.,device='cpu',dtype=torch.float64)
        xy,h,k=routes(torch.tensor([[.4,.6,.3,-.2,-.3,-.1],[0.,0.,0.,0.,0.,0.]],dtype=torch.float64))
        before=xy[0,:,0]<3.9
        self.assertTrue(torch.equal(xy[0,before],xy[1,before]))
        np.testing.assert_allclose(xy[0,before,1].numpy(),0,atol=1e-12)
        np.testing.assert_allclose(h[0,before].numpy(),0,atol=1e-12)
        np.testing.assert_allclose(k[0,before].numpy(),0,atol=1e-12)
        np.testing.assert_allclose(torch.linalg.vector_norm(xy.diff(dim=1),dim=-1).numpy(),.02,atol=2e-5)
        np.testing.assert_allclose(xy[1,:,0].numpy(),routes.s.numpy()-5,atol=1e-12)

if __name__=='__main__':unittest.main()
