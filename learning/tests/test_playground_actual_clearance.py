import unittest
import mujoco
import numpy as np
from sttw_control.playground_actual_clearance import obstacle_clearance

class ActualClearanceTest(unittest.TestCase):
    def test_closest_recorded_pose_and_time(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="planning_obstacle_0" type="sphere" size=".1" pos="1 0 0"/><body><freejoint/><geom name="robot" type="sphere" size=".1"/></body></worldbody></mujoco>')
        q=np.zeros((2,7));q[:,3]=1;q[1,0]=.5
        result=obstacle_clearance(m,q,np.array([0.,.005]))
        self.assertAlmostEqual(result['minimum']['distance_m'],.3)
        self.assertEqual(result['minimum']['time_s'],.005)
        self.assertEqual(result['minimum']['geometry'],'robot')

if __name__=='__main__':unittest.main()
