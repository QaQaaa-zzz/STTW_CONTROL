"""A constraint row permutation must preserve each equation and its contact address."""
import numpy as np
import warp as wp

from sttw_control.playground_constraint_order import order_active


def test_canonical_rows_retain_contact_axes_and_coefficients():
    def array(values, dtype):
        return wp.array(np.asarray(values), dtype=dtype, device='cpu')

    # First two contacts share geoms and differ in position. Six elliptic
    # rows arrive interleaved in a noncanonical axis order.
    dim = array([3,3], int)
    geom = array([[2,5],[2,5]],wp.vec2i)
    pos = array([[1,0,0],[0,0,0]],wp.vec3)
    frame = array([np.eye(3), np.eye(3)], wp.mat33)
    address = array([[3,6,1],[5,2,4]],int)
    kind = array([[0,7,7,7,7,7,7,99]],int)
    owner = array([[2,0,1,0,1,1,0,-1]],int)
    jac = array(np.arange(16).reshape(1,8,2),float)
    values = [array([np.arange(8)+i*10],float) for i in range(6)]
    old_jac = jac.numpy().copy()
    wp.launch(order_active,dim=1,device='cpu',inputs=[
        array([7],int),dim,geom,pos,frame,address,kind,owner,jac,*values])
    perm = [0,5,2,4,3,6,1,7]
    np.testing.assert_array_equal(jac.numpy(),old_jac[:,perm])
    for i,v in enumerate(values):
        np.testing.assert_array_equal(v.numpy()[0],np.array(perm)+i*10)
    np.testing.assert_array_equal(kind.numpy(),[[0,7,7,7,7,7,7,99]])
    np.testing.assert_array_equal(owner.numpy(),[[2,1,1,1,0,0,0,-1]])
    np.testing.assert_array_equal(address.numpy(),[[4,5,6],[1,2,3]])
