import unittest
import torch
from sttw_control.playground_rolling import RollingMotionLibrary

class RollingSnapshotTest(unittest.TestCase):
    def test_restores_selection_and_failure_without_aliasing(self):
        p=RollingMotionLibrary.__new__(RollingMotionLibrary)
        p.current_route=torch.tensor([1,2]);p.active=torch.tensor([3,4]);p.failed=torch.tensor([False,True])
        state=p.snapshot();p.active[0]=9;p.current_route[1]=8;p.failed[:]=False
        p.restore(state)
        self.assertEqual(p.active.tolist(),[3,4]);self.assertEqual(p.current_route.tolist(),[1,2]);self.assertEqual(p.failed.tolist(),[False,True])
        p.active[0]=7;self.assertEqual(state['active'].tolist(),[3,4])

if __name__=='__main__':unittest.main()
