import unittest
import numpy as np
import warp as wp
from sttw_control.playground_tree_reductions import subtree_com,composite_inertia,vec10

class TreeReductionTest(unittest.TestCase):
    def test_sibling_sums_preserve_parent_and_separate_worlds(self):
        parent=wp.array(np.array([0,0,1,1],np.int32),dtype=int,device='cpu')
        tree=wp.array(np.array([2,3],np.int32),dtype=int,device='cpu')
        for width,dtype,kernel in [(3,wp.vec3,subtree_com),(10,vec10,composite_inertia)]:
            values=np.arange(2*4*width,dtype=np.float32).reshape(2,4,width)/10
            expected=values.copy();expected[:,1]+=values[:,2]+values[:,3]
            data=wp.array(values,dtype=dtype,device='cpu')
            wp.launch(kernel,dim=(2,2),inputs=[parent,data,tree],outputs=[data],device='cpu')
            np.testing.assert_allclose(data.numpy(),expected,rtol=1e-6,atol=1e-6)
            # A world-root node never contributes to itself; inertia also skips parent zero.
            root_tree=wp.array(np.array([0],np.int32),dtype=int,device='cpu')
            wp.launch(kernel,dim=(2,1),inputs=[parent,data,root_tree],outputs=[data],device='cpu')
            np.testing.assert_allclose(data.numpy(),expected,rtol=1e-6,atol=1e-6)

if __name__=='__main__':unittest.main()
