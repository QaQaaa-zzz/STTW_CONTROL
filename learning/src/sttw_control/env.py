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
from .actuator import ActuatorConfig,initial_actuator,apply_residual,effective_base
from .observation import ObservationConfig,initial_history,advance_history,make_frame,observation_fields
from .recovery import RecoveryConfig,initial_recovery,update_recovery
from .path import BendConfig,bend_table,bend_command,bend_features,ReferencePaths,integrated_reference,features_at_progress
from .path import CircleConfig,circle_command,FigureEightConfig,eight_command,eight_features


from .events import RandomEvents,sample_event,profile
from .priority import PriorityConfig,priority_weights
from .action_mapping import MappingConfig,map_action
from .motion_commands import MotionCommands
from .timed_reference import TimedReferenceConfig
from .tracking_reward import TrackingConfig, initial_return, return_observation
from .tracking_reward import transition as tracking_transition

@dataclass(frozen=True)
class TaskConfig:
    controller: ControllerConfig=field(default_factory=ControllerConfig)
    actuator: ActuatorConfig=field(default_factory=ActuatorConfig)
    observation: ObservationConfig=field(default_factory=ObservationConfig)
    recovery: RecoveryConfig=field(default_factory=RecoveryConfig)
    action_mapping: MappingConfig | None=None
    motion_commands: MotionCommands | None=None
    tracking: TrackingConfig | None=None
    reference_paths: ReferencePaths | None=None
    timed_reference: TimedReferenceConfig | None=None
    bend: BendConfig | None=None
    rear_disturbance_mode: str="torque"  # event[5]: signed Nm or signed rad/s bias magnitude; positive opposes forward motion
    disturbance_rear_torque: float=0.
    circle: CircleConfig | None=None
    figure_eight: FigureEightConfig | None=None
    priority: PriorityConfig | None=None
    horizon_seconds: float=8.
    speed_reference: float=2.
    speed_schedule: tuple | None=None  # geometric tracking: (seconds, speed m/s) rows
    learning_roll_reference: float | None=None  # rad; does not change ECBC target.
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
    alive_reward_rate: float=1.  # Reward per simulated second, only before termination.
    path_excess_weight: float=0.
    path_soft_limit: float=.2
    path_error_weight: float=0.
    heading_error_weight: float=0.
    speed_error_weight: float=1.
    preparation_seconds: float=0.  # closed-loop zero-residual straight preparation before task time zero
    preparation_base_output_scale: float=1.

    def __post_init__(self):
        is_time = self.timed_reference is not None and self.timed_reference.mode == 'time'
        if is_time != self.observation.include_timed:raise ValueError("timed reference and observation must agree")
        if self.timed_reference is not None:
            if self.tracking is None or self.tracking.timed != is_time or any(x is not None for x in (self.reference_paths,self.bend,self.circle,self.figure_eight,self.speed_schedule,self.motion_commands)):raise ValueError("timed tracking needs its own independent reference")
            tr=self.timed_reference
            if tr.mode == 'geometry' and self.tracking.geometric:
                raise ValueError('full-curve geometry and committed geometry are distinct modes')
            if tr.mode == 'geometry' and self.tracking.objective not in ('geometric_huber', 'asymmetric_geometric_huber'):
                raise ValueError('geometry mode requires an explicit geometric huber objective')
            if tr.training_mix and not math.isclose(self.speed_reference, 2.3):
                raise ValueError('declared mixed task preparation starts at 2.3 m/s')
            if tr.speed_max>.1*self.actuator.rear_rate_limit or tr.max_steer>self.actuator.steer_limit:raise ValueError("timed reference exceeds actuator contract")
            if tr.fixed is not None:
                if not math.isclose(tr.fixed[0][1],self.speed_reference) or tr.fixed[0][2]!=0 or any(row[0]>=self.horizon_seconds or row[1]>.1*self.actuator.rear_rate_limit for row in tr.fixed):raise ValueError("invalid timed fixed initial/time/speed contract")
            elif not tr.training_mix and tr.switch_windows and tr.switch_windows[-1][1]>=self.horizon_seconds:raise ValueError("timed switches must fit horizon")
            if tr.fixed is None and not tr.training_mix and tr.recovery_probability and max(tr.recovery_start,tr.switch_windows[-1][1])+.001>=self.horizon_seconds:
                raise ValueError("recovery mixture switches must fit horizon")
        elif self.tracking is not None and self.tracking.timed:raise ValueError("timed reward requires timed reference")
        if self.reference_paths is not None:
            if self.tracking is None or self.bend is None or self.speed_schedule is not None:
                raise ValueError('reference bank needs geometry tracking and lookahead settings, without a second speed schedule')
            for case in self.reference_paths.cases:
                if not math.isclose(case['commands'][0][1],self.speed_reference):raise ValueError('reference initial speed mismatch')
                for t,v,w in case['commands']:
                    if t>=self.horizon_seconds or v>.1*self.actuator.rear_rate_limit or not math.isclose(t/self.controller.dt,round(t/self.controller.dt),abs_tol=1e-7):raise ValueError('reference exceeds time/speed contract')
        if self.speed_schedule is not None:
            if self.tracking is None or not self.speed_schedule:
                raise ValueError("speed schedule requires geometric tracking")
            last=-1.
            for row in self.speed_schedule:
                if (len(row)!=2 or not all(math.isfinite(x) for x in row) or row[0]<=last
                        or row[0]<0 or not 0<row[1]<=.1*self.actuator.rear_rate_limit):
                    raise ValueError("invalid geometric speed schedule")
                if not math.isclose(row[0]/self.controller.dt,round(row[0]/self.controller.dt),abs_tol=1e-7):
                    raise ValueError("speed changes must align with control ticks")
                last=row[0]
            if self.speed_schedule[0][0]!=0 or not math.isclose(self.speed_schedule[0][1],self.speed_reference):
                raise ValueError("first speed request must match physical reset speed")
        if (self.tracking is not None) != self.observation.include_tracking:
            raise ValueError("tracking config and observation flag must agree")
        if self.tracking is not None:
            if (self.motion_commands is not None or not self.observation.include_path
                    or self.priority is None or not self.observation.include_priority
                    or self.action_mapping is not None or self.priority.risk_gate
                    or self.observation.include_attitude_risk
                    or self.actuator.composition != "additive"):
                raise ValueError("geometric tracking requires visible alpha, path and fixed additive authority")
            if self.learning_roll_reference is not None:
                raise ValueError("geometric tracking must preserve dynamic roll reference")
            if self.horizon_seconds > 10.:
                raise ValueError("new tracking episodes must not exceed 10 seconds")
            if self.tracking.roll_working_limit >= self.roll_failure:
                raise ValueError("working roll limit must lie below failure threshold")
        if self.action_mapping is not None and self.actuator.composition!="additive":
            raise ValueError("physical action mapping requires additive composition")
        if (self.motion_commands is not None)!=self.observation.include_motion:raise ValueError("motion observation contract mismatch")
        if self.motion_commands is not None:
            if any(x is not None for x in (self.bend,self.circle,self.figure_eight,self.random_events,self.action_mapping)) or self.observation.include_path or self.priority is None:raise ValueError("independent commands require direct priority policy without a path or random force events")
            if self.motion_commands.fixed is not None and (not math.isclose(self.motion_commands.fixed[0][1],self.speed_reference) or any(row[1]>.1*self.actuator.rear_rate_limit for row in self.motion_commands.fixed)):raise ValueError("fixed initial/requested speed contract")
            if self.motion_commands.roll_working_limit>=self.roll_failure:raise ValueError("working roll limit must be below failure threshold")
            if self.motion_commands.speed_max>.1*self.actuator.rear_rate_limit:raise ValueError("command speed exceeds actuator command bound")
        if self.learning_roll_reference is not None and (not math.isfinite(self.learning_roll_reference) or abs(self.learning_roll_reference)>=self.roll_failure):
            raise ValueError("learning roll reference must be finite and inside roll failure bound")
        if (self.priority is not None)!=self.observation.include_priority:
            raise ValueError('priority config and observation flag must agree')
        if self.priority is not None and (self.action_mapping is not None or not (self.observation.include_path or self.observation.include_motion)):
            raise ValueError('priority conditioning requires direct residual and path observations')
        if self.action_mapping is not None:
            if self.action_mapping.authority_aware and self.actuator.delay_steps:
                raise ValueError('authority allocation does not predict delayed command headroom')
            if self.action_mapping.horizon<=self.actuator.delay_steps*self.actuator.dt:
                raise ValueError('mapping horizon must exceed command delay')
            if self.actuator.steer_acceleration is not None or self.actuator.rear_acceleration is not None:
                raise ValueError('mapping predictor does not model command slew limits')
        if not math.isfinite(self.failure_penalty) or self.failure_penalty<=0 or not math.isfinite(self.path_soft_limit) or self.path_soft_limit<=0:
            raise ValueError("invalid failure penalty/path soft limit")
        if not math.isfinite(self.alive_reward_rate) or self.alive_reward_rate<0:
            raise ValueError("invalid alive reward rate")
        if any(not math.isfinite(x) or x<0 for x in (self.path_error_weight,self.heading_error_weight,self.speed_error_weight,self.path_excess_weight)):
            raise ValueError("invalid reward weights")
        if self.rear_disturbance_mode not in ('torque','command'):raise ValueError('invalid rear disturbance mode')
        if not math.isfinite(self.disturbance_rear_torque):raise ValueError('rear disturbance amplitude must be finite in declared mode units')
        if self.bend is not None and self.bend.max_steer>self.actuator.steer_limit:raise ValueError('bend steering exceeds actuator limit')
        if sum(x is not None for x in (self.circle,self.figure_eight,self.bend))>1:
            raise ValueError('choose one reference path')
        if self.figure_eight is not None and self.figure_eight.max_steer>self.actuator.steer_limit:
            raise ValueError('figure eight steer bound exceeds actuator limit')
        if self.observation.include_path and self.circle is None and self.figure_eight is None and self.bend is None and self.timed_reference is None:
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
        if not math.isfinite(self.preparation_seconds) or self.preparation_seconds<0 or not math.isfinite(self.preparation_base_output_scale) or self.preparation_base_output_scale<=0:
            raise ValueError('invalid preparation duration')
        if self.preparation_seconds and self.timed_reference is None:
            raise ValueError('closed-loop task preparation is currently defined for timed references')
        if not math.isclose(self.preparation_seconds/self.controller.dt,round(self.preparation_seconds/self.controller.dt),abs_tol=1e-8):
            raise ValueError('preparation duration must align to control ticks')
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
    path=Path(path)
    raw=json.loads(path.read_text())
    if 'extends' in raw:
        parent=path.parent/raw.pop('extends')
        base=json.loads(parent.read_text())
        if 'extends' in base:
            raise ValueError('nested config inheritance is not supported')
        for key,value in raw.items():
            if isinstance(value,dict) and isinstance(base.get(key),dict):
                base[key]={**base[key],**value}
            else:base[key]=value
        raw=base
    return config_from_dict(raw)


def config_from_dict(raw):
    raw=dict(raw)
    for name,cls in [('timed_reference',TimedReferenceConfig),('reference_paths',ReferencePaths),('tracking',TrackingConfig),('motion_commands',MotionCommands),('bend',BendConfig),('controller',ControllerConfig),('actuator',ActuatorConfig),('observation',ObservationConfig),('recovery',RecoveryConfig),('circle',CircleConfig),('figure_eight',FigureEightConfig),('priority',PriorityConfig),('random_events',RandomEvents),('action_mapping',MappingConfig)]:
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
    priority_alpha: object
    command_schedule: object=None
    yaw_rate: object=0.
    priority_locked: object=False
    tracking_state: object=None
    tracking_components: object=None
    path_id: object=0
    path_progress: object=0.
    reference_pose: object=None
    reference_command: object=None
    raw_reference_request: object=None
    geometric_table: object=None
    geometric_features: object=None
    path_segment: object=0
    reference_geometry: object=None
    eso_enabled: object=False
    preparation_failed: object=False
    active_base_output_scale: object=1.

    @property
    def balance_recovered(self): return self.recovery.balance_recovered
    @property
    def task_recovered(self): return self.recovery.task_recovered


class RecoveryEnv:
    action_size=2

    def __new__(cls,config=TaskConfig(),**kwargs):
        if cls is RecoveryEnv and config.timed_reference is not None:
            from .timed_env import TimedRecoveryEnv
            return object.__new__(TimedRecoveryEnv)
        if cls is RecoveryEnv and config.motion_commands is not None:
            from .command_env import CommandRecoveryEnv
            return object.__new__(CommandRecoveryEnv)
        return object.__new__(cls)

    def __init__(self,config=TaskConfig(),*,backend='cpu'):
        if backend not in ('cpu','mjx'): raise ValueError('backend must be cpu or mjx')
        self.config=config
        self.backend=backend
        self.bundle=load_model()
        self.model=self.bundle.model
        self._wheel_body_ids=tuple(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('frontwheel','rearwheel'))
        self._allowed_floor_bodies=(0,*self._wheel_body_ids)
        c=config.actuator
        if c.steer_limit>self.model.jnt_range[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'steering_joint'),1] or c.steer_rate_limit>self.model.actuator_ctrlrange[2,1] or c.rear_rate_limit>self.model.actuator_ctrlrange[1,1]:
            raise ValueError('configured actuator limits exceed the authoritative XML')
        ratio=config.controller.dt/self.model.opt.timestep
        if not math.isclose(ratio,round(ratio),abs_tol=1e-9): raise ValueError('noninteger physics substeps')
        self.substeps=int(round(ratio))
        self.horizon=int(math.ceil(config.horizon_seconds/config.controller.dt))
        self.bend_table=jp.asarray(bend_table(config.bend)) if config.bend is not None else None
        self.reference_tables=None
        if config.reference_paths is not None:
            self.reference_tables=jp.asarray(np.stack([integrated_reference(x['commands'],config.horizon_seconds,config.controller.dt) for x in config.reference_paths.cases]))
            length=max(len(x['commands']) for x in config.reference_paths.cases)
            self.reference_schedules=jp.asarray([list(x['commands'])+[[1e9,*x['commands'][-1][1:]]]*(length-len(x['commands'])) for x in config.reference_paths.cases])
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
        return self.config.disturbance_rear_torque!=0 or self.config.random_events is not None or self.config.disturbance_force!=0 or self.config.disturbance_steer_rate!=0

    def fixed_event(self):
        return jp.array([self.event_start,self.event_end,self.config.disturbance_steer_rate,self.config.disturbance_force,float(self.config.disturbance_waveform=='half_sine'),self.config.disturbance_rear_torque])

    def disturbance_active(self,tick,event=None):
        event=self.fixed_event() if event is None else event
        return (tick>=event[0])&(tick<event[1])

    def disturbance_wrench(self,data,tick,event=None,*,magnitude=None):
        c=self.config
        xp=np if self.backend=='cpu' else jp
        if magnitude is None:
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

    def speed_command(self,tick,path_id=0):
        if self.reference_tables is not None:
            rows=self.reference_schedules[path_id]
            index=jp.maximum(jp.sum(tick*self.config.controller.dt>=rows[:,0])-1,0)
            return rows[index,1]
        if self.config.speed_schedule is None:return jp.asarray(self.config.speed_reference)
        rows=jp.asarray(self.config.speed_schedule)
        index=jp.maximum(jp.sum(tick*self.config.controller.dt>=rows[:,0])-1,0)
        return rows[index,1]

    def command(self,tick,pose=None,schedule=None,path_id=0,path_progress=None):
        c=self.config
        speed=self.speed_command(tick,path_id)
        if self.reference_tables is not None:return jp.array([bend_command(pose,c.bend,self.reference_tables[path_id],c.controller.wheelbase,c.controller.caster,progress=path_progress if c.reference_paths.continuous_projection else None),speed])
        if c.bend is not None:return jp.array([bend_command(pose,c.bend,self.bend_table,c.controller.wheelbase,c.controller.caster),speed])
        if c.figure_eight is not None:
            if pose is None:raise ValueError("figure eight requires localization")
            return jp.array([eight_command(pose,c.figure_eight,c.controller.wheelbase,c.controller.caster),speed])
        if c.circle is not None:
            if pose is None: raise ValueError('circle tracking requires XY/yaw localization')
            return jp.array([circle_command(pose,c.circle,c.controller.wheelbase,c.controller.caster),speed])
        return jp.array([c.steer_reference+c.steer_amplitude*jp.sin(2*jp.pi*c.steer_frequency*tick*c.controller.dt),speed])

    def pose(self,data):
        matrix=jp.asarray(data.xmat[self.bundle.chassis]).reshape(3,3)
        return jp.array([data.qpos[0],data.qpos[1],jp.arctan2(matrix[1,0],matrix[0,0])])

    def measure(self,data):
        b=self.bundle
        w,x,y,z=data.qpos[3:7]
        raw=jp.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))-jp.pi/2
        gyro=data.sensordata[b.imu_gyro:b.imu_gyro+3]
        return jp.array([-raw,-gyro[0],data.qpos[b.steer_qpos],data.qvel[b.steer_dof],gyro[2],-data.qvel[b.rear_dof],-data.qvel[b.front_dof]])

    def path_features(self,pose,path_id=0,path_progress=None):
        if self.reference_tables is not None and self.config.reference_paths.continuous_projection and path_progress is not None:return features_at_progress(pose,self.reference_tables[path_id],path_progress)
        if self.reference_tables is not None:return bend_features(pose,self.reference_tables[path_id])[0]
        if self.config.bend is not None:return bend_features(pose,self.bend_table)[0]
        if self.config.figure_eight is not None:return eight_features(pose,self.config.figure_eight)
        c=self.config.circle
        if c is None:
            return jp.zeros(3)
        dx,dy=pose[0]-c.center_x,pose[1]-c.center_y
        tangent=jp.arctan2(dy,dx)+c.direction*jp.pi/2
        heading=jp.arctan2(jp.sin(pose[2]-tangent),jp.cos(pose[2]-tangent))
        return jp.array([jp.sqrt(dx*dx+dy*dy)-c.radius,heading,c.direction/c.radius])

    def _prepare(self,controller,actuator,history,measurement,tick,pose,alpha=.5,command_override=None,extra_frame=None,tracking_state=None,path_id=0,path_progress=None,path_features_override=None,timed_frame=None,eso_enabled=False,base_output_scale=None):
        c=self.config
        command=(self.command(tick,pose,path_id=path_id,path_progress=path_progress) if c.reference_paths is not None else self.command(tick,pose)) if command_override is None else command_override
        roll,rate,steer,steer_rate,_,rear,_=measurement
        row=jp.array([rear*.1,steer,steer_rate,roll,rate,command[0]])
        controller,out=controller_step(controller,row,(tick*c.controller.dt>c.eso_start)|jp.asarray(eso_enabled),c.controller)
        learning_reference=out.reference_roll if c.learning_roll_reference is None else jp.asarray(c.learning_roll_reference)
        raw_base=jp.array([out.steer_rate,command[1]/.1])
        visible_base=effective_base(raw_base,c.actuator,base_output_scale)
        frame=make_frame(measurement,command,learning_reference,visible_base[0],actuator.previous,out.disturbance)
        if c.observation.include_path:
            frame=jp.concatenate([frame,self.path_features(pose,path_id,path_progress) if path_features_override is None else path_features_override])
        if c.observation.include_motion:
            frame=jp.concatenate([frame,jp.zeros(2) if extra_frame is None else extra_frame])
        if c.priority is not None:
            risk=priority_weights(alpha,roll-learning_reference,roll,rate,c.priority)[0]
            frame=jp.concatenate([frame,jp.array([alpha,risk]) if c.observation.include_attitude_risk else jp.array([alpha])])
        if c.tracking is not None:
            frame=jp.concatenate([frame,return_observation(initial_return() if tracking_state is None else tracking_state,c.tracking)])
        if c.observation.include_timed:frame=jp.concatenate([frame,jp.zeros(3) if timed_frame is None else timed_frame])
        history,obs=advance_history(history,frame,c.observation)
        return controller,history,obs,raw_base,learning_reference

    def reset(self,seed=0,*,reference_id=None):
        key=jax.random.PRNGKey(seed) if isinstance(seed,int) else seed
        c=self.config
        path_id=jp.int32(0)
        if c.reference_paths is not None:
            chosen=c.reference_paths.selected if reference_id is None else reference_id
            path_id=jax.random.randint(jax.random.fold_in(key,79),(),0,len(c.reference_paths.cases)) if chosen is None else jp.asarray(chosen,jp.int32)
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
        alpha=(jax.random.uniform(jax.random.fold_in(key,31)) if c.priority.randomize_alpha else jp.asarray(c.priority.fixed_alpha)) if c.priority is not None else jp.asarray(.5)
        controller,history,obs,base,reference=self._prepare(initial_controller(c.controller),actuator,initial_history(c.observation),measurement,jp.int32(0),pose,alpha,path_id=path_id,path_progress=jp.asarray(0.))
        state=EnvState(data,controller,actuator,history,initial_recovery(),measurement,pose,reference,base,obs,
                        jp.int32(0),jp.asarray(0.),jp.bool_(False),jp.bool_(False),jp.bool_(False),jp.int32(0),
                        sample_event(jax.random.fold_in(key,17),c.random_events,c.controller.dt) if c.random_events else self.fixed_event(),alpha,path_id=path_id,path_progress=jp.asarray(0.))
        state=state.replace(active_base_output_scale=jp.asarray(c.actuator.base_output_scale))
        if c.tracking is not None:
            tracking_state=initial_return()
            _,components=tracking_transition(tracking_state,roll=0.,roll_rate=0.,speed_error=0.,
                lateral_error=0.,heading_error=0.,action=jp.zeros(2),alpha=alpha,dt=c.controller.dt,
                alive_rate=c.alive_reward_rate,failure_penalty=c.failure_penalty,failed=False,
                enabled=False,config=c.tracking,**({"longitudinal_error":0.,"yaw_rate_error":0.} if c.tracking.timed else {}))
            state=state.replace(tracking_state=tracking_state,
                                tracking_components=jax.tree.map(jp.zeros_like,components))
        return state

    def _advance(self,state,measurement,actuator,action,physical_contact,physics_finite,true_speed,pose):
        c=self.config
        if c.tracking is not None:
            return self._advance_tracking(state,measurement,actuator,action,physical_contact,
                                          physics_finite,true_speed,pose)
        tick=state.tick+1
        command=self.command(tick,pose)
        eso_enabled=state.eso_enabled|(tick*c.controller.dt>c.eso_start)
        controller,history,obs,base,reference=self._prepare(state.controller,actuator,state.history,measurement,tick,pose,state.priority_alpha,eso_enabled=eso_enabled,base_output_scale=state.active_base_output_scale)
        leaves=jax.tree_util.tree_leaves((controller,actuator,history,obs,base,measurement,action,true_speed))
        invalid=~jp.all(jp.stack([jp.all(jp.isfinite(leaf)) for leaf in leaves])) | ~jp.asarray(physics_finite)
        fallen=jp.abs(measurement[0])>c.roll_failure
        failed=invalid|fallen|physical_contact
        # Count only complete stable intervals after the force has ended.
        event_finished=((state.event[2]!=0)|(state.event[3]!=0)|(state.event[5]!=0))&(state.tick>=state.event[1])
        errors=jp.array([measurement[0]-reference,measurement[1],true_speed-command[1],measurement[2]-command[0]])
        recovery=update_recovery(state.recovery,*errors,event_finished,failed,c.recovery)
        timeout=(tick>=self.horizon)&~failed
        weights=priority_weights(state.priority_alpha,errors[0],measurement[0],measurement[1],c.priority) if c.priority is not None else jp.array([0.,1.,1.,1.])
        reward=c.controller.dt*(c.alive_reward_rate-weights[3]*(10*errors[0]**2+errors[1]**2)-weights[1]/(c.priority.speed_cost_scale if c.priority else 1.)*c.speed_error_weight*errors[2]**2-errors[3]**2-.01*jp.sum(action**2))
        path=self.path_features(pose)
        reward-=c.controller.dt*weights[2]/(c.priority.path_cost_scale if c.priority else 1.)*(c.path_error_weight*path[0]**2+c.heading_error_weight*path[1]**2+c.path_excess_weight*jp.maximum(jp.abs(path[0])-c.path_soft_limit,0.)**2)
        if c.priority is None:
            reward=c.controller.dt*(c.alive_reward_rate-10*errors[0]**2-errors[1]**2-c.speed_error_weight*errors[2]**2-errors[3]**2-.01*jp.sum(action**2))
            reward-=c.controller.dt*(c.path_error_weight*path[0]**2+c.heading_error_weight*path[1]**2+c.path_excess_weight*jp.maximum(jp.abs(path[0])-c.path_soft_limit,0.)**2)
        reward=jp.where(failed,-c.failure_penalty,reward)+jp.where(recovery.task_recovered&~state.recovery.task_recovered,5.,0.)
        code=jp.where(invalid,3,jp.where(physical_contact,4,jp.where(fallen,1,jp.where(timeout,2,0))))
        return state.replace(controller=controller,actuator=actuator,history=history,recovery=recovery,measurement=measurement,pose=pose,
                             reference=reference,base=base,obs=jp.nan_to_num(obs,nan=0.,posinf=0.,neginf=0.),tick=tick,reward=reward,done=failed|timeout,
                             terminated=failed,truncated=timeout,end_code=code,eso_enabled=eso_enabled)

    def _advance_tracking(self,state,measurement,actuator,action,physical_contact,physics_finite,true_speed,pose):
        c=self.config;tick=state.tick+1
        # The reward uses the request and alpha that generated this transition.
        command=self.command(state.tick,state.pose,path_id=state.path_id,path_progress=state.path_progress)
        progress=state.path_progress
        if c.reference_paths is not None and c.reference_paths.continuous_projection:
            window=c.reference_paths.projection_margin+2*jp.linalg.norm(pose[:2]-state.pose[:2])
            _,progress=bend_features(pose,self.reference_tables[state.path_id],progress,window)
        path=self.path_features(pose,state.path_id,progress)
        leaves=jax.tree.leaves((measurement,actuator,action,true_speed,pose,path))
        invalid=(~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in leaves]))
                 | ~jp.asarray(physics_finite))
        fallen=jp.abs(measurement[0])>c.roll_failure
        failed=invalid|fallen|physical_contact
        tracking_state,parts=tracking_transition(state.tracking_state,
            roll=measurement[0],roll_rate=measurement[1],speed_error=true_speed-command[1],
            lateral_error=path[0],heading_error=path[1],action=action,alpha=state.priority_alpha,
            dt=c.controller.dt,alive_rate=c.alive_reward_rate,failure_penalty=c.failure_penalty,
            failed=failed,enabled=state.tick*c.controller.dt>=c.tracking.start_seconds,
            config=c.tracking)
        eso_enabled=state.eso_enabled|(tick*c.controller.dt>c.eso_start)
        controller,history,obs,base,reference=self._prepare(state.controller,actuator,state.history,
            measurement,tick,pose,state.priority_alpha,tracking_state=tracking_state,path_id=state.path_id,path_progress=progress,eso_enabled=eso_enabled,base_output_scale=state.active_base_output_scale)
        controller_invalid=~jp.all(jp.stack([jp.all(jp.isfinite(x)) for x in jax.tree.leaves((controller,obs,base))]))
        invalid=invalid|controller_invalid;failed=failed|controller_invalid
        parts={k:jp.where(controller_invalid,-c.failure_penalty if k=='failure' else 0.,v) for k,v in parts.items()}
        timeout=(tick>=self.horizon)&~failed
        code=jp.where(invalid,3,jp.where(physical_contact,4,jp.where(fallen,1,jp.where(timeout,2,0))))
        recovery=state.recovery.replace(balance_recovered=tracking_state.hold>=c.tracking.hold_seconds,
                                         task_recovered=tracking_state.credited & ~tracking_state.pending & ~tracking_state.deadline_missed & ~failed)
        return state.replace(controller=controller,actuator=actuator,history=history,measurement=measurement,
            pose=pose,reference=reference,base=base,obs=jp.nan_to_num(obs),tick=tick,reward=sum(parts.values()),
            done=failed|timeout,terminated=failed,truncated=timeout,end_code=code,recovery=recovery,
            tracking_state=tracking_state,tracking_components=parts,path_progress=progress,eso_enabled=eso_enabled)

    def _contact_failure(self,data):
        # A floor contact with a non-wheel body counts as physical failure.
        if self.backend=='cpu':
            for contact in data.contact[:data.ncon]:
                first,second=self.model.geom_bodyid[contact.geom]
                if (first==0 and second not in self._allowed_floor_bodies) or (second==0 and first not in self._allowed_floor_bodies): return True
            return False
        geom=data.contact.geom
        body=jp.asarray(self.model.geom_bodyid)[jp.maximum(geom,0)]
        wheelids=jp.array(self._wheel_body_ids)
        other=jp.where(body[:,0]==0,body[:,1],body[:,0])
        floor=jp.any(body==0,axis=1)
        active=(data.contact.dist<=0)&jp.all(geom>=0,axis=1)
        return jp.any(active&floor&~jp.any(other[:,None]==wheelids[None,:],axis=1))

    def set_priority(self,state,alpha):
        """Explicit intervention on current alpha; retain truthful past frames."""
        if self.config.priority is None:raise ValueError('policy is not priority conditioned')
        alpha=jp.clip(jp.asarray(alpha),0.,1.)
        history=state.history.replace(frames=state.history.frames.at[-1,observation_fields(self.config.observation).index("speed_priority")].set(alpha))
        obs=jp.concatenate([history.frames.flatten(),history.mask])
        return state.replace(priority_alpha=alpha,history=history,obs=obs,priority_locked=jp.bool_(True))

    def prepare_action(self,state,action):
        """Shared pure preparation for physics and pre-limit diagnostics."""
        c=self.config
        if c.action_mapping is not None:
            mapped=map_action(action,state.measurement[5]*c.action_mapping.wheel_radius_proxy,state.measurement[2],c.action_mapping,c.actuator,c.controller,base=state.base)
            # Keep malformed input visible to failure detection, while the shared
            # actuator disables its residual exactly as in the direct branch.
            action=jp.where(jp.all(jp.isfinite(action)),mapped,action)
        offset=state.event[2]*profile(state.tick,state.event)
        base=effective_base(state.base,c.actuator,state.active_base_output_scale).at[0].add(offset)
        if c.rear_disturbance_mode=='command':base=base.at[1].add(-state.event[5]*profile(state.tick,state.event))
        return base,action

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
            data.qfrc_applied[:]=0
            data.qfrc_applied[self.bundle.rear_dof]=float(state.event[5]*profile(state.tick,state.event)) if c.rear_disturbance_mode=='torque' else 0.
            physical_contact=False
            # The event envelope is held over a control tick. Heading and COM
            # still change at each physics substep while a force is applied.
            event=np.asarray(state.event)
            magnitude=event[3]*profile(state.tick,event,np)
            for _ in range(self.substeps):
                if magnitude!=0:
                    data.xfrc_applied[self.bundle.chassis,:]=self.disturbance_wrench(data,state.tick,magnitude=magnitude)
                mujoco.mj_step(self.model,data)
                physical_contact=physical_contact or self._contact_failure(data)
            # Sensors from mj_step can lag the final integration state.
            mujoco.mj_forward(self.model,data)
            physics_finite=jp.asarray(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.act).all())
            true_speed=jp.dot(jp.asarray(data.qvel[:3]),jp.asarray(data.xmat[self.bundle.chassis]).reshape(3,3)[:,0])
            new=self._advance_jit(state.replace(data=None),self.measure(data),actuator,jp.asarray(action),physical_contact or self._contact_failure(data),physics_finite,true_speed,self.pose(data))
        else:
            data=state.data.replace(ctrl=ctrl,xfrc_applied=jp.zeros_like(state.data.xfrc_applied),qfrc_applied=jp.zeros_like(state.data.qfrc_applied).at[self.bundle.rear_dof].set(state.event[5]*profile(state.tick,state.event) if c.rear_disturbance_mode=='torque' else 0.))
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
