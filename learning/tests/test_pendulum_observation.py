import math
import torch
from sttw_control.pendulum_skill import TorchPendulumObservation


def test_upright_frame_wheel_sign_and_real_fifo():
    names=['frontwheel_joint','steering_joint','rearwheel_joint','hip_joint','knee_joint']
    h=math.sqrt(.5)
    history=TorchPendulumObservation(names,[h,-h,0.,0.],1)
    root=torch.tensor([[2.,0.,.18,h,0.,0.,h,2.,0.,0.,0.,0.,0.]])
    pos=torch.tensor([[0.,.1,0.,-1.2,2.5]])
    vel=torch.tensor([[-20.,.3,-21.,.4,.5]])
    action=torch.tensor([[.1,.2,.3,.4]])
    for k in range(3):
        root[:,0]=2.+k
        obs=history.advance(root,pos,vel,torch.zeros(1,3),torch.tensor([5.])-root[:,0]-.4,action,torch.tensor([float(k==2)]))
    assert obs.shape==(1,76)
    frame=obs[0,50:75]
    torch.testing.assert_close(frame[:3],torch.tensor([0.,0.,-1.]),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(frame[6:9],torch.tensor([0.,0.,9.81]),atol=2e-6,rtol=1e-6)
    torch.testing.assert_close(frame[9:17],torch.tensor([.1,-1.2,2.5,.3,.4,.5,20.,21.]))
    torch.testing.assert_close(frame[17:21],action[0])
    torch.testing.assert_close(obs[0,[22,47,72]],torch.tensor([2.6,1.6,.6]))
    torch.testing.assert_close(obs[0,[24,49,74,75]],torch.ones(4))


def test_original_critic_order_omits_only_added_passive_suspension():
    names=['frontwheel_joint','steering_joint','rearwheel_joint','hip_joint','knee_joint','rearsuspension_joint']
    h=math.sqrt(.5);history=TorchPendulumObservation(names,[h,-h,0.,0.],1)
    root=torch.tensor([[2.,0.,.25,h,0.,0.,h,2.,.1,.2,.3,.4,.5]])
    position=torch.tensor([[3.,.1,4.,-1.2,2.5,.04]])
    velocity=torch.tensor([[-20.,.3,-21.,.4,.5,.6]])
    critic=history.privileged(torch.zeros(1,76),root,position,velocity,torch.tensor([True]))
    assert critic.shape==(1,106)
    torch.testing.assert_close(critic[0,76:88],torch.tensor([2.,0.,.25,1.,0.,0.,0.,-4.,.1,-3.,-1.2,2.5]),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(critic[0,88:99],torch.tensor([2.,.1,.2,.3,.4,.5,21.,.3,20.,.4,.5]),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(critic[0,99:106],torch.tensor([0.,0.,0.,2.,.1,.2,1.]),atol=1e-6,rtol=1e-6)
