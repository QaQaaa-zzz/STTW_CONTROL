"""One-solved-step causal diagnostic torque channel, separate from the motor."""
import torch


class DelayedContactTorque:
    """Schedule an explicitly bounded external generalized torque for step k+1.

    This is a causal simulation intervention, not a new deployable actuator.
    Call ``after_solve`` only after reading the completed step's contact data.
    The plant consumes the pending tensor on its next ``step``. Submitted
    impulses are integrated from the plant's copied previous-step contribution.
    """
    def __init__(self,plant,modes,limit_Nm):
        if plant.servo_mode!='explicit' or len(modes)!=plant.count or limit_Nm<=0:
            raise ValueError('Explicit plant, one mode per lane and positive intervention limit required')
        self.plant=plant;self.limit=limit_Nm
        mapping={'none':0,'cancel_negative':1,'restore_reference':2,'pulse':3}
        self.mode=torch.tensor([mapping[x] for x in modes],device=plant.device)
        plant.diagnostic_joint_torque=torch.zeros_like(plant.dofs[:,:,0])
        self.requested_impulse=torch.zeros(plant.count,device=plant.device)
        self.submitted_impulse=torch.zeros_like(self.requested_impulse)
        self.submitted_absolute_impulse=torch.zeros_like(self.requested_impulse)

    def after_solve(self,negative_lateral_moment,reference_delta,pulse,alive):
        previous=self.plant.last_diagnostic_joint_torque[:,self.plant.signals.steer]
        dt=self.plant.physics_dt
        self.submitted_impulse+=previous*dt
        self.submitted_absolute_impulse+=previous.abs()*dt
        request=torch.where(self.mode==1,-negative_lateral_moment,torch.where(self.mode==2,reference_delta,torch.where(self.mode==3,pulse,0.)))
        request=torch.where(alive,request,0.)
        self.requested_impulse+=request*dt
        self.plant.diagnostic_joint_torque[:,self.plant.signals.steer]=request.clamp(-self.limit,self.limit)
        return dict(requested_next_torque=request,submitted_previous_torque=previous)
