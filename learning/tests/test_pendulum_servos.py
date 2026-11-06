import unittest
import torch
from sttw_control.torch_controller import TorchPositionServos

class PositionServoTests(unittest.TestCase):
    def test_target_and_force_limits_with_damping(self):
        rows=[dict(name=n,joint=n,enabled=True,gear=[1.,0.,0.,0.,0.,0.],gain=[100.],bias=[0.,-100.,-6.],ctrlrange=r,forcerange=[-30.,30.]) for n,r in [('hip',[-1.3,.5]),('knee',[-1.5,2.5])]]
        servos=TorchPositionServos({'actuators':rows},['wheel','hip','knee'],['hip','knee'])
        pos=torch.tensor([[0.,0.,2.5],[0.,.4,0.]])
        vel=torch.tensor([[0.,0.,1.],[0.,1.,0.]])
        result=servos.torque(torch.tensor([[2.,3.],[.5,-2.]]),pos,vel)
        torch.testing.assert_close(result,torch.tensor([[0.,30.,-6.],[0.,4.,-30.]]))
