import unittest
import numpy as np
import torch
from sttw_control.playground_planner import SplineRoutes,feasibility_order,GroundCEM

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

    def test_goal_hold_rejects_temporary_departure_despite_good_endpoint(self):
        class Predictor:
            def __call__(self,pose,xi,previous,s,xy,h,k,steps,return_domain=False):
                q=pose[:,None,:].expand(-1,steps+1,-1).clone();q[:,:,0]=torch.linspace(8.2,8.8,steps+1);q[:,10,1]=1.
                return q,xi[:,None,:].expand(-1,steps+1,-1)
        envelope=lambda q:(q[:,:,:2,None].transpose(2,3)-.1,q[:,:,:2,None].transpose(2,3)+.1)
        routes=SplineRoutes(3.3,8.,device='cpu');c=dict(steps=30,iterations=4,elites=8,parameter_limit_m=1.4,minimum_std_m=.04,margin_m=.2,roll_limit_rad=.7,pitch_limit_rad=1.309,min_speed_m_s=.5,curvature_limit_per_m=1.4,exit_x_m=6.2,tail_distance_m=1.8,exit_y_m=0.,exit_y_tolerance_m=.3,exit_heading_tolerance_rad=.2,exit_speed_tolerance_m_s=.4,wheelbase_m=.27903)
        pose=torch.zeros((1,12));pose[:,0]=8.2;pose[:,3]=1;xi=torch.zeros((1,9));xi[:,0]=1.8
        args=(torch.zeros((1,6)),pose,xi,xi,torch.tensor([-5.,25.,-10.,10.]),torch.tensor([[20.,21.,5.,6.]]))
        self.assertEqual(float(GroundCEM(routes,Predictor(),envelope,c,'dynamic').evaluate(*args)[4]),0.)
        c.update(goal_hold_steps=25,goal_x_min_m=8.,goal_x_max_m=9.7)
        self.assertGreater(float(GroundCEM(routes,Predictor(),envelope,c,'dynamic').evaluate(*args)[4]),0.)

if __name__=='__main__':unittest.main()
