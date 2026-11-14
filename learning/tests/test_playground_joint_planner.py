import unittest
import torch
from sttw_control.playground_joint_planner import MotionLibraryPlanner


class JointPlannerTest(unittest.TestCase):
    def make(self):
        center=torch.tensor([[[0.,0.,.2],[2.,0.,.2]],[[0.,0.,.2],[2.,.05,.2]],[[0.,1.,.2],[2.,1.,.2]],[[0.,1.,.2],[2.,1.1,.2]]])
        return MotionLibraryPlanner(center-.05,center+.05,torch.ones(4,2,dtype=torch.bool),
            torch.tensor([[True,False]]*4),torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1]]),
            torch.tensor([3.,2.,4.,5.]),torch.tensor([1.,1.,2.,2.]),torch.ones(4,dtype=torch.bool),.03)

    def test_joint_keeps_feasible_upstream_alternative(self):
        selected,valid,*_=self.make()(torch.tensor([-1.,4.,-2.,2.]),torch.tensor([[1.9,-.2,0.,2.1,.2,.5]]))
        self.assertEqual(valid.tolist(),[False,True]);self.assertEqual(selected.tolist(),[0,2])

    def test_same_choice_when_first_segment_does_not_restrict_completion(self):
        selected,valid,*_=self.make()(torch.tensor([-1.,4.,-2.,2.]),torch.tensor([[1.9,-1.4,0.,2.1,-1.2,.5]]))
        self.assertEqual(valid.tolist(),[True,True]);self.assertEqual(selected.tolist(),[1,1])

    def test_no_plan_keeps_incumbent_instead_of_least_violation(self):
        selected,valid,*_=self.make()(torch.tensor([-1.,4.,-2.,2.]),torch.tensor([[1.9,-2.,0.,2.1,2.,.5]]))
        self.assertEqual(valid.tolist(),[False,False]);self.assertEqual(selected.tolist(),[0,0])


if __name__=='__main__':unittest.main()
