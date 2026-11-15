import unittest
import numpy as np
from sttw_control.playground_constraint_log import clearance_witness,dynamic_witnesses

class ConstraintLogTest(unittest.TestCase):
    def test_margin_rejects_clear_body_at_specific_obstacle(self):
        lo=np.array([[[0.,0.,0.]],[[1.,0.,0.]]]);hi=lo+.1
        row=clearance_witness(lo,hi,np.full((2,3),.2),[-5,5,-5,5],[[1.25,0.,0.,1.5,.1,.1]],['wheel'])
        self.assertEqual((row['step'],row['constraint'],row['geometry']),(1,'obstacle_0','wheel'))
        self.assertAlmostEqual(row['raw_clearance_m'],.15)
        self.assertAlmostEqual(row['expanded_clearance_m'],-.05)

    def test_road_witness_without_obstacles(self):
        lo=np.array([[[0.,.9,0.]]]);hi=lo+.1
        row=clearance_witness(lo,hi,np.full((1,3),.1),[-5,5,-1,1],[],['frame'])
        self.assertEqual(row['constraint'],'road_3')
        self.assertAlmostEqual(row['expanded_clearance_m'],-.1)

class DynamicLogTest(unittest.TestCase):
    def test_hold_is_consecutive_and_late_domain_failure_is_excluded(self):
        pose=np.zeros((31,12));pose[:,0]=8.5;v=np.zeros((31,9));v[:,0]=1.8
        domain=np.zeros((31,3));domain[:,2]=1.;angles=np.zeros((31,3));angles[28:,0]=1.
        goal=dict(x_min_m=8.,x_max_m=9.,abs_y_max_m=.3,min_vx_m_s=.5,path_tolerance_m=.3,heading_tolerance_rad=.2,speed_tolerance_m_s=.4)
        d=dynamic_witnesses(pose,v,angles,domain,v[:,0],1.8,goal,.6,10)
        self.assertTrue(d['valid']);self.assertEqual(d['goal_time_s'],.5)
        pose[10,1]=1.
        d=dynamic_witnesses(pose,v,angles,domain,v[:,0],1.8,goal,.6,10)
        self.assertFalse(d['valid']);self.assertEqual(d['events'][0]['constraint'],'roll')
        self.assertEqual(d['events'][-1]['constraint'],'goal_hold')
        self.assertEqual(d['events'][-1]['longest_consecutive_samples'],20)

if __name__=='__main__':unittest.main()
