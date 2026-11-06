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

    def test_selected_inactive_steering_preserves_native_effort(self):
        row=dict(name='steer_pos',joint='steer',enabled=False,gear=[1.,0.,0.,0.,0.,0.],gain=[10.],bias=[0.,-10.,-.1],ctrlrange=[-.8,.8],forcerange=[-1.,1.])
        servo=TorchPositionServos({'actuators':[row]},['steer','wheel'],['steer_pos'],allow_inactive=True)
        torque=servo.torque(torch.tensor([[.8],[.05]]),torch.zeros(2,2),torch.tensor([[0.,0.],[1.,0.]]))
        torch.testing.assert_close(torque,torch.tensor([[1.,0.],[.4,0.]]))
