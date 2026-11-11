"""Independent dynamics and geometry checks for shared margin calculations."""
import json,unittest
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from sttw_control import margin_theory as m

class MarginTheoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads((Path(__file__).parent/'fixtures/paper_preference/controller.json').read_text())

    def test_impulse_matches_state_transition(self):
        for v in [.5,1.5,2.3,2.6,3.]:
            c,b,a=m.coefficients(v,self.c);kd,kphi,kp=m.gains(v,self.c)
            A=np.array([[0,1,0],[a-c*kphi,-c*kp,b-c*kd],[-kphi,-kp,-kd]])
            for t in [0,.05,.2,.6,2.]:
                got=np.array(m.impulse_response(v,t,self.c));x=expm(A*t)@np.array([0,1,0])
                np.testing.assert_allclose(got[:3],x,atol=1e-10)
                self.assertAlmostEqual(got[3],-np.array([kphi,kp,kd])@x,places=10)

    def test_rate_then_travel_envelope(self):
        for side in [-1,1]:
            x=side*np.array([.12,.8,.65]);v=2.6;t=np.array([.01,.05,.1,.4])
            c,b,a=m.coefficients(v,self.c);A=np.zeros((4,4));A[:3,:3]=[[0,1,0],[a,0,b],[0,0,0]];A[:3,3]=[0,c,1]
            got,orientation=m.optimal_envelope(v,x,t,self.c)
            for i,tt in enumerate(t):
                hit=(.8-.65)/3;first=min(hit,tt)
                exact=(expm(A*first)@np.r_[x,side*3])[:3]
                exact=(expm(A*max(tt-hit,0))@np.r_[exact,0])[:3]
                np.testing.assert_allclose(np.array(got)[:,i]*orientation,exact,atol=1e-11)

    def test_tracking_projection(self):
        for heading in [-1.2,-.2,.7,1.4]:
            for lateral in [-.5,.2]:
                world_target=np.array([1.6,-lateral]);forward=np.array([np.cos(heading),np.sin(heading)])
                self.assertAlmostEqual(m.local_tracking_margin(1.6,lateral,heading),m.tracking_margin(world_target@forward),places=12)

    def test_heading_is_separate_tracking_constraint(self):
        forward,heading=m.tracking_domain_margins(1.,np.pi)
        self.assertGreater(forward,0);self.assertLess(heading,0)

    def test_batch_margin_matches_scalar_braking_classification(self):
        rng=np.random.default_rng(42);v=rng.uniform(.5,3,40)
        states=rng.uniform([-.6,-8,-.8],[.6,8,.8],(40,3))
        margins=m.recovery_margin_batch(v,states,self.c)
        for speed,state,margin in zip(v,states,margins):
            result=m.recovery_outcome(speed,state,self.c)
            self.assertEqual(margin<0,result['outcome']=='unavoidable_nominal_limit')

    def test_known_capacity_and_cascade(self):
        low=m.impulse_capacity(1.8,self.c);high=m.impulse_capacity(2.6,self.c)
        self.assertAlmostEqual(low['p1_roll_rate'],.05291,places=4)
        self.assertAlmostEqual(high['p1_roll_rate'],.30161,places=4)
        self.assertAlmostEqual(high['J_sat_rad_s'],2.2186,places=3)

if __name__=='__main__':unittest.main()
