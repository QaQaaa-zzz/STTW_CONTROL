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


class ContactTorqueExperiment:
    """GPU solved-step diagnostics and next-step intervention, compact chunk output."""
    def __init__(self,env,config,output):
        import json
        from pathlib import Path
        if not env.plant.track_contact_wrench:
            raise ValueError('Torque intervention requires native contact wrench tracking')
        self.channel=DelayedContactTorque(env.plant,config['modes'],config['limit_Nm'])
        self.output=Path(output)/'contact_intervention';self.output.mkdir()
        self.config=config;self.dt=env.physics_dt;self.total=0;self.used=0;self.chunks=[];self.buffers=None
        self.chunk_steps=int(config.get('chunk_steps',250))
        self.window=tuple(config['active_window_s'])
        self.pulse_window=tuple(config.get('pulse_window_s',self.window))
        self.pulse=torch.tensor(config.get('pulse_Nm',[0.]*env.n),device=env.device)
        if len(self.window)!=2 or not 0<=self.window[0]<self.window[1] or self.pulse.shape!=(env.n,):
            raise ValueError('Declare increasing intervention window and one pulse amplitude per lane')
        self.zero_torques={key:torch.zeros_like(self.pulse) for key in ['normal_moment','friction_moment','lateral_moment','negative_lateral','reference_delta']}
        self.metadata=dict(config=config,physics_dt_s=self.dt,body_order=env.plant.bodies,
            semantics='post-solve wrench k requests bounded external steering torque at k+1; unchanged motor limits',
            impulse_scope='submitted generalized torque, not direct measurement of cancelled contact impulse',
            clock='first sampled state at physics_dt; active window gates next-step application time',
            contact_sampling='geometry_observed marks actual contact calculation; zero placeholders outside active window are unobserved, not zero measured contact')
        (self.output/'protocol.json').write_text(json.dumps(self.metadata,indent=2))

    def record(self,env,lower_tick,physics_tick):
        from .contact_torque_geometry import front_contact_torques
        next_time=(self.total+1)*self.dt
        active=self.window[0]<=next_time<self.window[1]
        torques=front_contact_torques(env.plant.read_wheel_contact_samples(),env.plant.bodies) if active else self.zero_torques
        submitted=self.channel.after_solve(torques['negative_lateral'],torques['reference_delta'],self.pulse if self.pulse_window[0]<=next_time<self.pulse_window[1] else torch.zeros_like(self.pulse),env.alive&active)
        values=dict(**torques,**submitted,geometry_observed=torch.full((env.n,),active,device=env.device,dtype=torch.bool),bounded_next_torque=env.plant.diagnostic_joint_torque[:,env.plant.signals.steer],alive_before=env.alive.clone(),alpha=env.alpha.clone(),
            steer_velocity=env.plant.dofs[:,env.plant.signals.steer,1],
            steer_position=env.plant.dofs[:,env.plant.signals.steer,0],
            motor_steering_torque=env.plant.last_motor_torque[:,env.plant.signals.steer])
        if self.buffers is None:
            self.buffers={k:torch.empty((self.chunk_steps,)+v.shape,device=v.device,dtype=v.dtype) for k,v in values.items()}
        for key,value in values.items():self.buffers[key][self.used].copy_(value)
        self.used+=1;self.total+=1
        if self.used==self.chunk_steps:self.flush()

    def flush(self):
        import numpy as np
        if not self.used:return
        name=f'chunk_{len(self.chunks):05d}.npz'
        arrays={k:v[:self.used].cpu().numpy() for k,v in self.buffers.items()}
        arrays['time_s']=np.arange(self.total-self.used+1,self.total+1)*self.dt
        np.savez_compressed(self.output/name,**arrays)
        self.chunks.append(dict(file=name,first_step=self.total-self.used+1,last_step=self.total));self.used=0

    def close(self):
        import json
        self.flush()
        self.metadata.update(total_physics_steps=self.total,chunks=self.chunks,
            requested_impulse_Nms=self.channel.requested_impulse.cpu().tolist(),
            submitted_impulse_Nms=self.channel.submitted_impulse.cpu().tolist(),
            submitted_absolute_impulse_Nms=self.channel.submitted_absolute_impulse.cpu().tolist())
        (self.output/'index.json').write_text(json.dumps(self.metadata,indent=2)+'\n')
