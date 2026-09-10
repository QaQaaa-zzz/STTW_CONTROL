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
from .observation import ObservationConfig,initial_history,advance_history,make_frame,observation_fields
from .recovery import RecoveryConfig,initial_recovery,update_recovery
from .path import CircleConfig,circle_command,FigureEightConfig,eight_command,eight_features


from .events import RandomEvents,sample_event,profile
from .action_mapping import MappingConfig,map_action

@dataclass(frozen=True)
class TaskConfig:
    controller: ControllerConfig=field(default_factory=ControllerConfig)
    actuator: ActuatorConfig=field(default_factory=ActuatorConfig)
    observation: ObservationConfig=field(default_factory=ObservationConfig)
    recovery: RecoveryConfig=field(default_factory=RecoveryConfig)
    action_mapping: MappingConfig | None=None
    circle: CircleConfig | None=None
    figure_eight: FigureEightConfig | None=None
    horizon_seconds: float=8.
    speed_reference: float=2.
    steer_reference: float=0.
    steer_amplitude: float=0.
    steer_frequency: float=.25
    eso_start: float=3.
    disturbance_start: float=4.
    disturbance_duration: float=.1
    random_events: RandomEvents | None=None
    disturbance_waveform: str="constant"
    disturbance_force: float=0.
    disturbance_steer_rate: float=0.
    disturbance_force_frame: str="world_y"
    disturbance_force_point: str="chassis_com"
    initial_roll_range: float=0.
    roll_failure: float=.7
    failure_penalty: float=10.
    path_excess_weight: float=0.
    path_soft_limit: float=.2
    path_error_weight: float=0.
    heading_error_weight: float=0.
    speed_error_weight: float=1.

    def __post_init__(self):
        if self.action_mapping is not None:
            if self.action_mapping.authority_aware and self.actuator.delay_steps:
                raise ValueError('authority allocation does not predict delayed command headroom')
            if self.action_mapping.horizon<=self.actuator.delay_steps*self.actuator.dt:
                raise ValueError('mapping horizon must exceed command delay')
            if self.actuator.steer_acceleration is not None or self.actuator.rear_acceleration is not None:
                raise ValueError('mapping predictor does not model command slew limits')
        if not math.isfinite(self.failure_penalty) or self.failure_penalty<=0 or not math.isfinite(self.path_soft_limit) or self.path_soft_limit<=0:
            raise ValueError("invalid failure penalty/path soft limit")
        if any(not math.isfinite(x) or x<0 for x in (self.path_error_weight,self.heading_error_weight,self.speed_error_weight,self.path_excess_weight)):
            raise ValueError("invalid reward weights")
        if self.circle is not None and self.figure_eight is not None:
            raise ValueError('choose one reference path')
        if self.figure_eight is not None and self.figure_eight.max_steer>self.actuator.steer_limit:
            raise ValueError('figure eight steer bound exceeds actuator limit')
        if self.observation.include_path and self.circle is None and self.figure_eight is None:
            raise ValueError("path observations require a reference path")
        if self.disturbance_waveform not in ("constant","half_sine"):
            raise ValueError("invalid event waveform")
        if self.random_events and self.random_events.start_max+self.random_events.duration_max>=self.horizon_seconds:
            raise ValueError("random event must finish inside episode")
        if self.disturbance_force_frame not in ("world_y","heading_lateral") or self.disturbance_force_point not in ("chassis_com","vehicle_com"):
            raise ValueError("invalid force frame/application point")
        scalars=(self.horizon_seconds,self.speed_reference,self.steer_reference,self.steer_amplitude,
                 self.steer_frequency,self.eso_start,self.disturbance_start,self.disturbance_duration,
                 self.disturbance_force,self.disturbance_steer_rate,self.initial_roll_range,self.roll_failure)
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
    for name,cls in [('controller',ControllerConfig),('actuator',ActuatorConfig),('observation',ObservationConfig),('recovery',RecoveryConfig),('circle',CircleConfig),('figure_eight',FigureEightConfig),('random_events',RandomEvents),('action_mapping',MappingConfig)]:
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
    event: object

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
        self.observation_size=config.observation.history_steps*(len(observation_fields(config.observation))+1)
        if backend=='mjx':
            from mujoco import mjx
            self.mjx_model=mjx.put_model(self.model,impl='jax')
            # MJX 3.6 JAX does not implement opt.disableactuator. Materialize
            # the existing CPU contract for disabled stateless fixed/affine
            # servos; zero ctrl alone leaves their position/velocity bias active.
            disabled=np.flatnonzero((self.model.opt.disableactuator >> self.model.actuator_group)&1)
            for index in disabled:
                if (self.model.actuator_dyntype[index]!=mujoco.mjtDyn.mjDYN_NONE
                    or self.model.actuator_gaintype[index]!=mujoco.mjtGain.mjGAIN_FIXED
                    or self.model.actuator_biastype[index] not in (mujoco.mjtBias.mjBIAS_NONE,mujoco.mjtBias.mjBIAS_AFFINE)):
                    raise ValueError('unsupported disabled actuator semantics for MJX')
                self.mjx_model=self.mjx_model.replace(
                    actuator_gainprm=self.mjx_model.actuator_gainprm.at[index].set(0),
                    actuator_biasprm=self.mjx_model.actuator_biasprm.at[index].set(0))
            self._mjx=mjx
        self._prepare_jit=jax.jit(self._prepare)
        self._advance_jit=jax.jit(self._advance)

    @property
    def has_disturbance(self):
        return self.config.random_events is not None or self.config.disturbance_force!=0 or self.config.disturbance_steer_rate!=0

    def fixed_event(self):
        return jp.array([self.event_start,self.event_end,self.config.disturbance_steer_rate,self.config.disturbance_force,float(self.config.disturbance_waveform=='half_sine')])

    def disturbance_active(self,tick,event=None):
        event=self.fixed_event() if event is None else event
        return (tick>=event[0])&(tick<event[1])

    def disturbance_wrench(self,data,tick,event=None):
        c=self.config
        xp=np if self.backend=='cpu' else jp
        event=xp.asarray(self.fixed_event() if event is None else event)
        magnitude=event[3]*profile(tick,event,xp)
        force=xp.array([0.,magnitude,0.])
        if c.disturbance_force_frame=='heading_lateral':
            matrix=xp.asarray(data.xmat[self.bundle.chassis]).reshape(3,3)
            yaw=xp.arctan2(matrix[1,0],matrix[0,0])
            force=magnitude*xp.array([-xp.sin(yaw),xp.cos(yaw),0.])
        torque=xp.zeros(3)
        if c.disturbance_force_point=='vehicle_com':
            offset=xp.asarray(data.subtree_com[self.bundle.chassis])-xp.asarray(data.xipos[self.bundle.chassis])
            torque=xp.cross(offset,force)
        return xp.concatenate([force,torque])

    def command(self,tick,pose=None):
        c=self.config
        if c.figure_eight is not None:
            if pose is None:raise ValueError("figure eight requires localization")
            return jp.array([eight_command(pose,c.figure_eight,c.controller.wheelbase,c.controller.caster),c.speed_reference])
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

    def path_features(self,pose):
        if self.config.figure_eight is not None:return eight_features(pose,self.config.figure_eight)
        c=self.config.circle
        if c is None:
            return jp.zeros(3)
        dx,dy=pose[0]-c.center_x,pose[1]-c.center_y
        tangent=jp.arctan2(dy,dx)+c.direction*jp.pi/2
        heading=jp.arctan2(jp.sin(pose[2]-tangent),jp.cos(pose[2]-tangent))
        return jp.array([jp.sqrt(dx*dx+dy*dy)-c.radius,heading,c.direction/c.radius])

    def _prepare(self,controller,actuator,history,measurement,tick,pose):
        c=self.config
        command=self.command(tick,pose)
        roll,rate,steer,steer_rate,_,rear,_=measurement
        row=jp.array([rear*.1,steer,steer_rate,roll,rate,command[0]])
        controller,out=controller_step(controller,row,tick*c.controller.dt>c.eso_start,c.controller)
        frame=make_frame(measurement,command,out.reference_roll,out.steer_rate,actuator.previous,out.disturbance)
        if c.observation.include_path:
            frame=jp.concatenate([frame,self.path_features(pose)])
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
                        jp.int32(0),jp.asarray(0.),jp.bool_(False),jp.bool_(False),jp.bool_(False),jp.int32(0),
                        sample_event(jax.random.fold_in(key,17),c.random_events,c.controller.dt) if c.random_events else self.fixed_event())

    def _advance(self,state,measurement,actuator,action,physical_contact,physics_finite,true_speed,pose):
        c=self.config
        tick=state.tick+1
        command=self.command(tick,pose)
        controller,history,obs,base,reference=self._prepare(state.controller,actuator,state.history,measurement,tick,pose)
        leaves=jax.tree_util.tree_leaves((controller,actuator,history,obs,base,measurement,action,true_speed))
        invalid=~jp.all(jp.stack([jp.all(jp.isfinite(leaf)) for leaf in leaves])) | ~jp.asarray(physics_finite)
        fallen=jp.abs(measurement[0])>c.roll_failure
        failed=invalid|fallen|physical_contact
        # Count only complete stable intervals after the force has ended.
        event_finished=((state.event[2]!=0)|(state.event[3]!=0))&(state.tick>=state.event[1])
        errors=jp.array([measurement[0]-reference,measurement[1],true_speed-command[1],measurement[2]-command[0]])
        recovery=update_recovery(state.recovery,*errors,event_finished,failed,c.recovery)
        timeout=(tick>=self.horizon)&~failed
        reward=c.controller.dt*(1.-10*errors[0]**2-errors[1]**2-c.speed_error_weight*errors[2]**2-errors[3]**2-.01*jp.sum(action**2))
        path=self.path_features(pose)
        reward-=c.controller.dt*(c.path_error_weight*path[0]**2+c.heading_error_weight*path[1]**2+c.path_excess_weight*jp.maximum(jp.abs(path[0])-c.path_soft_limit,0.)**2)
        reward=jp.where(failed,-c.failure_penalty,reward)+jp.where(recovery.task_recovered&~state.recovery.task_recovered,5.,0.)
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

    def prepare_action(self,state,action):
        """Shared pure preparation for physics and pre-limit diagnostics."""
        c=self.config
        if c.action_mapping is not None:
            mapped=map_action(action,state.measurement[5]*c.action_mapping.wheel_radius_proxy,state.measurement[2],c.action_mapping,c.actuator,c.controller,base=state.base)
            # Keep malformed input visible to failure detection, while the shared
            # actuator disables its residual exactly as in the direct branch.
            action=jp.where(jp.all(jp.isfinite(action)),mapped,action)
        offset=state.event[2]*profile(state.tick,state.event)
        return state.base.at[0].add(offset),action

    def _step(self,state,action):
        c=self.config
        base,action=self.prepare_action(state,action)
        actuator,command=apply_residual(state.actuator,base,action,state.measurement[2],c.actuator)
        ctrl=jp.array([0.,-command[1],command[0],command[0]])
        if self.backend=='cpu':
            data=mujoco.MjData(self.model)
            mujoco.mj_copyData(data,self.model,state.data)
            data.ctrl[:]=np.asarray(ctrl)
            data.xfrc_applied[:]=0
            physical_contact=False
            for _ in range(self.substeps):
                data.xfrc_applied[self.bundle.chassis,:]=np.asarray(self.disturbance_wrench(data,state.tick,state.event))
                mujoco.mj_step(self.model,data)
                physical_contact=physical_contact or self._contact_failure(data)
            # Sensors from mj_step can lag the final integration state.
            mujoco.mj_forward(self.model,data)
            physics_finite=jp.asarray(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.act).all())
            true_speed=jp.dot(jp.asarray(data.qvel[:3]),jp.asarray(data.xmat[self.bundle.chassis]).reshape(3,3)[:,0])
            new=self._advance_jit(state.replace(data=None),self.measure(data),actuator,jp.asarray(action),physical_contact or self._contact_failure(data),physics_finite,true_speed,self.pose(data))
        else:
            data=state.data.replace(ctrl=ctrl,xfrc_applied=jp.zeros_like(state.data.xfrc_applied))
            def substep(_,carry):
                d,contact=carry
                d=d.replace(xfrc_applied=d.xfrc_applied.at[self.bundle.chassis].set(self.disturbance_wrench(d,state.tick,state.event)))
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
