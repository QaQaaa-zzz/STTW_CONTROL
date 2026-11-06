from types import SimpleNamespace
import torch
from sttw_control.contact_torque_intervention import DelayedContactTorque


def test_compensation_is_pending_next_step_and_submission_is_measured_separately():
    plant=SimpleNamespace(servo_mode='explicit',count=4,device='cpu',physics_dt=.0002,dofs=torch.zeros(4,2,2),signals=SimpleNamespace(steer=0))
    control=DelayedContactTorque(plant,['none','cancel_negative','restore_reference','pulse'],1.)
    assert not plant.diagnostic_joint_torque.any()
    plant.last_diagnostic_joint_torque=plant.diagnostic_joint_torque.clone()
    result=control.after_solve(torch.full((4,),.3),torch.full((4,),2.),torch.full((4,),-.8),torch.ones(4,dtype=torch.bool))
    torch.testing.assert_close(plant.diagnostic_joint_torque[:,0],torch.tensor([0.,-.3,1.,-.8]))
    assert not result['submitted_previous_torque'].any()
    assert not control.submitted_impulse.any()
    # The following physical solve consumes the pending contribution exactly once.
    plant.last_diagnostic_joint_torque=plant.diagnostic_joint_torque.clone()
    control.after_solve(torch.zeros(4),torch.zeros(4),torch.zeros(4),torch.ones(4,dtype=torch.bool))
    torch.testing.assert_close(control.submitted_impulse,torch.tensor([0.,-.3,1.,-.8])*.0002)
    assert not plant.diagnostic_joint_torque.any()
