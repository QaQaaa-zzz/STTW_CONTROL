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


def test_rotated_robot_bounds_use_full_geometry_and_obstacle_extent():
    from sttw_control.playground_actual_clearance import obstacle_aabb_bounds
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="planning_obstacle_0" type="box" size=".1 .1 .1" pos="2 0 0"/><body><freejoint/><geom type="box" size=".4 .1 .2" pos=".2 0 0"/></body></worldbody></mujoco>')
    q=np.array([[0,0,0,1,0,0,0],[1.,0,0,np.sqrt(.5),0,0,np.sqrt(.5)]])
    envelope,clearance=obstacle_aabb_bounds(m,q)
    np.testing.assert_allclose(envelope,[[-.2,-.1,.6,.1],[.9,-.2,1.1,.6]],atol=1e-8)
    np.testing.assert_allclose(clearance,.8)
