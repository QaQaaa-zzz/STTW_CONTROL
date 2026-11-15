import unittest
import numpy as np
from sttw_control.playground_constraint_log import clearance_witness

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

if __name__=='__main__':unittest.main()
