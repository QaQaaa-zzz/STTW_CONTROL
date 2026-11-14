"""Four-channel ownership and fixed physical-boundary contracts."""
import unittest
from types import SimpleNamespace
import torch
import jax.numpy as jp
from sttw_control.playground_adapters import JumpAdapter
from sttw_control.playground_world_map import body_outside


class OwnershipMapTest(unittest.TestCase):
    def test_rb_keeps_wheels_and_homes_only_while_completing(self):
        phase=torch.tensor([0,1,3,4,2])
        rb=torch.arange(20,dtype=torch.float32).reshape(5,4)
        ground=-rb-1
        adapter=JumpAdapter(torch.tensor([0.,1.]),torch.tensor([2.,4.]),
                            torch.tensor([-1.,0.]),torch.tensor([1.,2.]),True)
        output=adapter.select(phase,ground,rb,torch.tensor([[.2,.4]]).expand(5,2))
        torch.testing.assert_close(output[[0,4]],ground[[0,4]],rtol=0,atol=0)
        torch.testing.assert_close(output[[1,2]],rb[[1,2]],rtol=0,atol=0)
        torch.testing.assert_close(output[3,:2],rb[3,:2],rtol=0,atol=0)
        torch.testing.assert_close(output[3,2:],torch.tensor([.1,1.1]))

    def test_finite_completion_targets_keep_actuator_limits(self):
        adapter=JumpAdapter(torch.tensor([0.,1.]),torch.ones(2),
                            torch.tensor([-1.,0.]),torch.tensor([1.,2.]),True)
        output=adapter.targets(torch.zeros(1,4),torch.tensor([4]),torch.tensor([[100.,-100.]]))
        torch.testing.assert_close(output[0,2:],torch.tensor([1.,0.]),rtol=0,atol=0)

    def test_whole_body_boundary_not_just_center(self):
        data=SimpleNamespace(geom_xpos=jp.array([[[0.,.8,0.]],[[0.,.6,0.]]]),
                             geom_xmat=jp.broadcast_to(jp.eye(3),(2,1,3,3)))
        geometry=SimpleNamespace(robot_geom_ids=jp.array([0]),robot_geom_types=jp.array([2]),
                                 robot_geom_sizes=jp.array([[.3,0.,0.]]))
        self.assertEqual(body_outside(data,geometry,[-2.,2.,-1.,1.]).tolist(),[True,False])


if __name__=='__main__':unittest.main()
