import unittest
import torch
from sttw_control.skill_sequence_rollout import native_to_ground,ground_to_native

class SkillSequenceTest(unittest.TestCase):
    def test_native_joint_order_round_trip_for_locked_ground(self):
        schema=dict(qpos_names=['x','y','z','w','qx','qy','qz','rearwheel_joint','steering_joint','frontwheel_joint','hip_joint','knee_joint'],steer_dof=7,front_dof=8,rear_dof=6)
        q=torch.tensor([[1.,2.,.3,.70710678,0.,0.,.70710678,11.,.2,13.,.4,-.1]])
        v=torch.tensor([[1.,2.,3.,.1,.2,.3,21.,.4,23.,0.,0.]])
        pose,velocity=native_to_ground(q,v,schema)
        self.assertEqual(pose[0,7:].tolist(),q[0,[8,9,7,10,11]].tolist())
        restored_q,restored_v=ground_to_native(pose,velocity,schema)
        torch.testing.assert_close(q,restored_q);torch.testing.assert_close(v,restored_v)

if __name__=='__main__':unittest.main()
