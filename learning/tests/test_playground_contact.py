"""World-local constraint addresses and global contact buffer reduction."""
import unittest
from types import SimpleNamespace
import jax.numpy as jp
from sttw_control.playground_contact import normal_loads


class ContactTest(unittest.TestCase):
    def test_world_addresses_masks_and_multiple_points(self):
        c=SimpleNamespace(contact__geom=jp.array([[0,1],[0,1],[0,2],[0,3],[0,3],[1,2],[0,1],[0,3]]),
            contact__worldid=jp.array([0,0,1,1,0,0,0,0]),
            contact__efc_address=jp.array([[0],[1],[0],[1],[-1],[2],[2],[2]]),
            contact__dist=jp.array([-.01,-.01,-.02,-.01,-.01,-.01,.01,-.01]),
            efc__force=jp.array([[10.,20.,30.],[40.,50.,60.]]),nacon=jp.array([7]))
        result=normal_loads(SimpleNamespace(_impl=c),jp.array([True,False,False,False]),(1,2),jp.array([3]))
        # A positive-distance contact can still exert force within its margin.
        self.assertEqual(result.tolist(),[[60.,0.,0.,0.],[0.,40.,50.,1.]])


if __name__=='__main__':unittest.main()
