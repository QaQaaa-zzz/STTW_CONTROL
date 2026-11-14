"""Ordinary route publication preserves closed-loop state and committed geometry."""
import unittest
from types import SimpleNamespace
import torch
from sttw_control.torch_middle_control import TorchMiddleControl
from sttw_control.torch_path_reference import TorchPathReference


def route(turn=False):
    s=torch.linspace(0,10,101)
    y=.05*(s-5).clamp_min(0)**3 if turn else torch.zeros_like(s)
    return TorchPathReference(dict(s=s,xy=torch.stack((s,y),-1),
        heading=torch.atan(.15*(s-5).clamp_min(0)**2) if turn else y,
        curvature=torch.zeros_like(s)),{})


class PlanUpdateTest(unittest.TestCase):
    def setUp(self):
        self.middle=TorchMiddleControl.__new__(TorchMiddleControl)
        self.middle.route=route()
        self.middle.tick=100
        self.middle.lane_ticks=torch.tensor([20,22])
        self.middle.projection={'progress':torch.tensor([2.,3.])}
        self.middle.follower={'lookahead':torch.tensor([.5,.5])}
        self.middle.v_user=torch.tensor([1.8,1.8])
        self.middle.frames=torch.randn(2,16,20)
        self.middle.policy=SimpleNamespace(filtered=torch.randn(2,2))

    def test_publish_preserves_all_other_state(self):
        state=self.middle.__dict__.copy()
        frames=self.middle.frames.clone()
        filtered=self.middle.policy.filtered.clone()
        candidate=route(True)
        self.middle.update_plan(candidate,torch.tensor([4.,4.]))
        self.assertIs(self.middle.route,candidate)
        for name,value in state.items():
            if name!='route':self.assertIs(self.middle.__dict__[name],value)
        torch.testing.assert_close(self.middle.frames,frames,rtol=0,atol=0)
        torch.testing.assert_close(self.middle.policy.filtered,filtered,rtol=0,atol=0)
        self.assertGreater(float(self.middle.route.at(torch.tensor([8.]))[0][0,1]),0.)

    def test_rejects_changed_committed_prefix(self):
        candidate=route(True);candidate.xy[20,1]=.1
        with self.assertRaisesRegex(ValueError,'committed prefix'):
            self.middle.update_plan(candidate,torch.tensor([4.,4.]))

    def test_rejects_short_prefix_and_mid_tick(self):
        with self.assertRaisesRegex(ValueError,'preview'):
            self.middle.update_plan(route(True),torch.tensor([2.,3.]))
        self.middle.tick=101
        with self.assertRaisesRegex(ValueError,'20 ms'):
            self.middle.update_plan(route(True),torch.tensor([4.,4.]))


if __name__=='__main__':unittest.main()
