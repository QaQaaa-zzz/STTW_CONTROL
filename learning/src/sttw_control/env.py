"""Deterministic CPU MuJoCo / MJX environment with a shared control tick.

No ROS dependency. CPU reset copies the model initial pose and starts at a
declared forward velocity (engineering reset, not a proven reached state).
"""
from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path
from flax import struct
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from .model import load_model
from .controller import ControllerConfig,initial_controller,controller_step
from .actuator import ActuatorConfig,initial_actuator,apply_residual
from .observation import ObservationConfig,initial_history,advance_history,make_frame
from .recovery import RecoveryConfig,initial_recovery,update_recovery
from .path import CircleConfig,circle_command


@dataclass(frozen=True)
class TaskConfig:
    controller: ControllerConfig=field(default_factory=ControllerConfig)
    actuator: ActuatorConfig=field(default_factory=ActuatorConfig)
    observation: ObservationConfig=field(default_factory=ObservationConfig)
    recovery: RecoveryConfig=field(default_factory=RecoveryConfig)
    circle: CircleConfig | None=None
    horizon_seconds: float=8.
    speed_reference: float=2.
    steer_reference: float=0.
    steer_amplitude: float=0.
    steer_frequency: float=.25
    eso_start: float=3.
    disturbance_start: float=4.
    disturbance_duration: float=.1
    disturbance_force: float=0.
    initial_roll_range: float=0.
    roll_failure: float=.7

    def __post_init__(self):
        scalars=(self.horizon_seconds,self.speed_reference,self.steer_reference,self.steer_amplitude,
                 self.steer_frequency,self.eso_start,self.disturbance_start,self.disturbance_duration,
                 self.disturbance_force,self.initial_roll_range,self.roll_failure)
        if not all(math.isfinite(x) for x in scalars) or self.roll_failure<=0 or self.steer_frequency<0 or self.eso_start<0:
            raise ValueError('task parameters must be finite with positive failure limit and nonnegative timing')
        if self.horizon_seconds<=0 or self.disturbance_start<0 or self.disturbance_duration<=0 or self.initial_roll_range<0:
            raise ValueError('invalid episode or perturbation interval')
        if self.controller.dt!=self.actuator.dt or self.controller.dt!=self.recovery.dt:
            raise ValueError('controller/actuator/recovery dt mismatch')
        if any(not math.isclose(value/self.controller.dt,round(value/self.controller.dt),abs_tol=1e-8)
               for value in (self.disturbance_start,self.disturbance_duration)):
            raise ValueError('disturbance start and duration must align to control ticks')
        if self.speed_reference<=0 or self.speed_reference>.1*self.actuator.rear_rate_limit:
            raise ValueError('initial engineering task requires forward speed within limits')
        if abs(self.steer_reference)+abs(self.steer_amplitude)>self.actuator.steer_limit:
            raise ValueError('steer reference exceeds position limit')
        if self.circle is not None and self.circle.max_steer>self.actuator.steer_limit:
            raise ValueError('circle steer bound exceeds actuator position limit')


def load_config(path):
    raw=json.loads(Path(path).read_text())
    for name,cls in [('controller',ControllerConfig),('actuator',ActuatorConfig),('observation',ObservationConfig),('recovery',RecoveryConfig),('circle',CircleConfig)]:
        if name in raw and raw[name] is not None: raw[name]=cls(**raw[name])
    return TaskConfig(**raw)


@struct.dataclass
class EnvState:
    data: object
    controller: object
    actuator: object
    history: object
    recovery: object
    measurement: object
    pose: object
    reference: object
    base: object
    obs: object
    tick: object
    reward: object
    done: object
    terminated: object
    truncated: object
    end_code: object

    @property
    def balance_recovered(self): return self.recovery.balance_recovered
    @property
    def task_recovered(self): return self.recovery.task_recovered


class RecoveryEnv:
    action_size=2

    def __init__(self,config=TaskConfig(),*,backend='cpu'):
        if backend not in ('cpu','mjx'): raise ValueError('backend must be cpu or mjx')
        self.config=config
        self.backend=backend
        self.bundle=load_model()
        self.model=self.bundle.model
        c=config.actuator
        if c.steer_limit>self.model.jnt_range[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'steering_joint'),1] or c.steer_rate_limit>self.model.actuator_ctrlrange[2,1] or c.rear_rate_limit>self.model.actuator_ctrlrange[1,1]:
            raise ValueError('configured actuator limits exceed the authoritative XML')
        ratio=config.controller.dt/self.model.opt.timestep
        if not math.isclose(ratio,round(ratio),abs_tol=1e-9): raise ValueError('noninteger physics substeps')
        self.substeps=int(round(ratio))
        self.horizon=int(math.ceil(config.horizon_seconds/config.controller.dt))
        self.event_start=int(round(config.disturbance_start/config.controller.dt))
        self.event_end=self.event_start+int(round(config.disturbance_duration/config.controller.dt))
        self.observation_size=config.observation.history_steps*16
        if backend=='mjx':
            from mujoco import mjx
            self.mjx_model=mjx.put_model(self.model,impl='jax')
            self._mjx=mjx
        self._prepare_jit=jax.jit(self._prepare)
        self._advance_jit=jax.jit(self._advance)

    def command(self,tick,pose=None):
        c=self.config
        if c.circle is not None:
            if pose is None: raise ValueError('circle tracking requires XY/yaw localization')
            return jp.array([circle_command(pose,c.circle,c.controller.wheelbase,c.controller.caster),c.speed_reference])
        return jp.array([c.steer_reference+c.steer_amplitude*jp.sin(2*jp.pi*c.steer_frequency*tick*c.controller.dt),c.speed_reference])

    def pose(self,data):
        matrix=jp.asarray(data.xmat[self.bundle.chassis]).reshape(3,3)
        return jp.array([data.qpos[0],data.qpos[1],jp.arctan2(matrix[1,0],matrix[0,0])])

    def measure(self,data):
        b=self.bundle
        w,x,y,z=data.qpos[3:7]
        raw=jp.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))-jp.pi/2
        gyro=data.sensordata[b.imu_gyro:b.imu_gyro+3]
        return jp.array([-raw,-gyro[0],data.qpos[b.steer_qpos],data.qvel[b.steer_dof],gyro[2],-data.qvel[b.rear_dof],-data.qvel[b.front_dof]])

    def _prepare(self,controller,actuator,history,measurement,tick,pose):
        c=self.config
        command=self.command(tick,pose)
        roll,rate,steer,steer_rate,_,rear,_=measurement
        row=jp.array([rear*.1,steer,steer_rate,roll,rate,command[0]])
        controller,out=controller_step(controller,row,tick*c.controller.dt>c.eso_start,c.controller)
        frame=make_frame(measurement,command,out.reference_roll,out.steer_rate,actuator.previous,out.disturbance)
        history,obs=advance_history(history,frame,c.observation)
        return controller,history,obs,jp.array([out.steer_rate,command[1]/.1]),out.reference_roll

    def reset(self,seed=0):
        key=jax.random.PRNGKey(seed) if isinstance(seed,int) else seed
        c=self.config
        qpos=jp.asarray(self.model.qpos0)
        angle=jax.random.uniform(key,(),minval=-c.initial_roll_range,maxval=c.initial_roll_range)
        # Left-positive physical roll = negative world-x rotation.
        w,x,y,z=qpos[3:7]
        ca,sa=jp.cos(-angle/2),jp.sin(-angle/2)
        qpos=qpos.at[3:7].set(jp.array([ca*w-sa*x,ca*x+sa*w,ca*y-sa*z,ca*z+sa*y]))
        qvel=jp.zeros(self.model.nv).at[0].set(c.speed_reference)
        qvel=qvel.at[self.bundle.rear_dof].set(-c.speed_reference/.1).at[self.bundle.front_dof].set(-c.speed_reference/.1)
        ctrl=jp.zeros(self.model.nu).at[1].set(-c.speed_reference/.1)
        if self.backend=='cpu':
            data=mujoco.MjData(self.model)
            data.qpos[:]=np.asarray(qpos); data.qvel[:]=np.asarray(qvel); data.ctrl[:]=np.asarray(ctrl)
            mujoco.mj_forward(self.model,data)
        else:
            data=self._mjx.make_data(self.model).replace(qpos=qpos,qvel=qvel,ctrl=ctrl)
            data=self._mjx.forward(self.mjx_model,data)
        actuator=initial_actuator(c.actuator,c.speed_reference/.1)
        measurement=self.measure(data)
        pose=self.pose(data)
        controller,history,obs,base,reference=self._prepare(initial_controller(c.controller),actuator,initial_history(c.observation),measurement,jp.int32(0),pose)
        return EnvState(data,controller,actuator,history,initial_recovery(),measurement,pose,reference,base,obs,
                        jp.int32(0),jp.asarray(0.),jp.bool_(False),jp.bool_(False),jp.bool_(False),jp.int32(0))

    def _advance(self,state,measurement,actuator,action,physical_contact,physics_finite,true_speed,pose):
        c=self.config
        tick=state.tick+1
        command=self.command(tick,pose)
        controller,history,obs,base,reference=self._prepare(state.controller,actuator,state.history,measurement,tick,pose)
        leaves=jax.tree_util.tree_leaves((controller,actuator,history,obs,base,measurement,action,true_speed))
        invalid=~jp.all(jp.stack([jp.all(jp.isfinite(leaf)) for leaf in leaves])) | ~physics_finite
        fallen=jp.abs(measurement[0])>c.roll_failure
        failed=invalid|fallen|physical_contact
        # Count only complete stable intervals after the force has ended.
        event_finished=(c.disturbance_force!=0)&(state.tick>=self.event_end)
        errors=jp.array([measurement[0]-reference,measurement[1],true_speed-command[1],measurement[2]-command[0]])
        recovery=update_recovery(state.recovery,*errors,event_finished,failed,c.recovery)
        timeout=(tick>=self.horizon)&~failed
        reward=c.controller.dt*(1.-10*errors[0]**2-errors[1]**2-errors[2]**2-errors[3]**2-.01*jp.sum(action**2))
        reward=jp.where(failed,-10.,reward)+jp.where(recovery.task_recovered&~state.recovery.task_recovered,5.,0.)
        code=jp.where(invalid,3,jp.where(physical_contact,4,jp.where(fallen,1,jp.where(timeout,2,0))))
        return state.replace(controller=controller,actuator=actuator,history=history,recovery=recovery,measurement=measurement,pose=pose,
                             reference=reference,base=base,obs=jp.nan_to_num(obs,nan=0.,posinf=0.,neginf=0.),tick=tick,reward=reward,done=failed|timeout,
                             terminated=failed,truncated=timeout,end_code=code)

    def _contact_failure(self,data):
        # A floor contact with a non-wheel body counts as physical failure.
        if self.backend=='cpu':
            for contact in data.contact[:data.ncon]:
                names=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[g])) for g in contact.geom]
                if 'world' in names and any(n not in ('world','frontwheel','rearwheel') for n in names): return True
            return False
        geom=data.contact.geom
        body=jp.asarray(self.model.geom_bodyid)[jp.maximum(geom,0)]
        wheelids=jp.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('frontwheel','rearwheel')])
        other=jp.where(body[:,0]==0,body[:,1],body[:,0])
        floor=jp.any(body==0,axis=1)
        active=(data.contact.dist<=0)&jp.all(geom>=0,axis=1)
        return jp.any(active&floor&~jp.any(other[:,None]==wheelids[None,:],axis=1))

    def _step(self,state,action):
        c=self.config
        actuator,command=apply_residual(state.actuator,state.base,action,state.measurement[2],c.actuator)
        ctrl=jp.array([0.,-command[1],command[0],command[0]])
        force=jp.where((state.tick>=self.event_start)&(state.tick<self.event_end),c.disturbance_force,0.)
        if self.backend=='cpu':
            data=mujoco.MjData(self.model)
            mujoco.mj_copyData(data,self.model,state.data)
            data.ctrl[:]=np.asarray(ctrl)
            data.xfrc_applied[:]=0
            data.xfrc_applied[self.bundle.chassis,1]=float(force)
            physical_contact=False
            for _ in range(self.substeps):
                mujoco.mj_step(self.model,data)
                physical_contact=physical_contact or self._contact_failure(data)
            # Sensors from mj_step can lag the final integration state.
            mujoco.mj_forward(self.model,data)
            physics_finite=jp.asarray(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.act).all())
            true_speed=jp.dot(jp.asarray(data.qvel[:3]),jp.asarray(data.xmat[self.bundle.chassis]).reshape(3,3)[:,0])
            new=self._advance_jit(state.replace(data=None),self.measure(data),actuator,jp.asarray(action),physical_contact or self._contact_failure(data),physics_finite,true_speed,self.pose(data))
        else:
            data=state.data.replace(ctrl=ctrl,xfrc_applied=jp.zeros_like(state.data.xfrc_applied).at[self.bundle.chassis,1].set(force))
            def substep(_,carry):
                d,contact=carry
                d=self._mjx.step(self.mjx_model,d)
                return d,contact|self._contact_failure(d)
            data,physical_contact=jax.lax.fori_loop(0,self.substeps,substep,(data,jp.bool_(False)))
            data=self._mjx.forward(self.mjx_model,data)
            physics_finite=jp.all(jp.isfinite(data.qpos))&jp.all(jp.isfinite(data.qvel))&jp.all(jp.isfinite(data.act))
            true_speed=jp.dot(data.qvel[:3],data.xmat[self.bundle.chassis,:,0])
            new=self._advance(state,self.measure(data),actuator,action,physical_contact|self._contact_failure(data),physics_finite,true_speed,self.pose(data))
        return new.replace(data=data)

    def step(self,state,action):
        if self.backend=='cpu':
            return state if bool(state.done) else self._step(state,jp.asarray(action))
        return jax.lax.cond(state.done,lambda _:state,lambda _:self._step(state,action),operand=None)
