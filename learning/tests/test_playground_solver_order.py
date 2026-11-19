"""Compare deterministic reductions to the native constraint cost and the active DOFs."""
import numpy as np
import warp as wp
from mujoco.mjx.third_party.mujoco_warp._src import solver, types
from sttw_control.playground_solver_order import gradient, ordered_cost


def arr(values, dtype=float):
    return wp.array(np.asarray(values), dtype=dtype, device='cpu')


def test_gradient_uses_active_dofs_only():
    grad = arr([[0,0,17,18]])
    norm = arr([0])
    wp.launch(gradient,dim=(1,2),device='cpu',inputs=[
        arr([[1,2]]),arr([[3,4]]),arr([[2,9,0,0]]),arr([False],bool),grad,norm])
    np.testing.assert_array_equal(grad.numpy(),[[-2,3,17,18]])
    np.testing.assert_array_equal(norm.numpy(),[13])


def test_cost_matches_native_for_elliptic_contact_and_joint_constraints():
    # two saturated joint friction rows, one equality, one limit and a cone with three axes
    kind=arr([[0,1,1,3,7,7,7]],int)
    ids=arr([[0,0,1,0,0,0,0]],int)
    dim=arr([3],int)
    address=arr([[4,5,6]],int)
    friction=arr([[.8,.8,1,0,0]],types.vec5)
    mass=arr([[2]*7])
    loss=arr([[1]*7])
    done=arr([False],bool)
    ratio=arr([1])
    length=arr([7],int)
    for normal in [-8.,-.1,8.]:
        residual=arr([[-1,-2,2,-1,normal,1,1]])
        states=arr([[0]*7],int)
        native=arr([0])
        args=[ratio,arr([1],int),arr([2],int),length,friction,dim,address,kind,ids,mass,loss,
              arr([1],int),residual,done,arr([[0]*7]),states,native]
        wp.launch(solver.update_constraint_efc,dim=(1,7),device='cpu',inputs=args)
        candidate=arr([0])
        wp.launch(ordered_cost,dim=1,device='cpu',inputs=[length,states,ids,mass,loss,residual,
            dim,address,friction,ratio,done,arr([.125]),candidate])
        np.testing.assert_allclose(candidate.numpy(),native.numpy()+.125,rtol=1e-6)


def test_cone_hessian_matches_native_for_two_contacts():
    from sttw_control.playground_solver_order import cone_hessian
    owner=arr([[1,1,1,0,0,0]],int)
    state=arr([[int(types.ConstraintState.CONE)]*6],int)
    dim=arr([3,3],int)
    address=arr([[3,4,5],[0,1,2]],int)
    friction=arr([[.8,.8,1,0,0],[.7,.7,1,0,0]],types.vec5)
    jac=arr(np.arange(12).reshape(1,6,2)/11)
    mass=arr([[2]*6])
    residual=arr([[0,1,1,.25,.5,.5]])
    tri=[arr([0,1,1],int),arr([0,0,1],int)]
    ratio,done=arr([1]),arr([False],bool)
    native=arr(np.zeros((1,2,2)))
    wp.launch(solver.update_gradient_JTCJ,dim=(2,3),device='cpu',inputs=[ratio,*tri,arr([-.01,-.01]),
        arr([0,0]),friction,dim,address,arr([0,0],int),jac,mass,state,2,arr([2],int),residual,done,1,2,native])
    ordered=arr(np.zeros((1,2,2)))
    wp.launch(cone_hessian,dim=(1,3),device='cpu',inputs=[arr([6],int),owner,ratio,*tri,friction,dim,
        address,jac,mass,state,residual,done,ordered])
    assert np.linalg.norm(native.numpy()) > 0
    np.testing.assert_allclose(ordered.numpy(),native.numpy(),atol=1e-6)
