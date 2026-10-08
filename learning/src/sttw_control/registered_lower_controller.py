"""Registered frozen310-dim controllers; exact validated causal Transfer semantics.

Migrated from runs/tracking_candidate_review_20261008/command_replay.py.
No old Actor weights, normalization, field order or return rules are modified.
"""
import json,hashlib
from pathlib import Path
import jax.numpy as jp
from flax import struct
from .frozen_lower_controller import FrozenLowerController,path_features
from .observation import ObservationConfig,observation_fields,make_frame,advance_history
from .tracking_reward import TrackingConfig,return_observation,transition
from .timed_reference import errors
from .network import load_policy
from .teleop_reference import integrate

@struct.dataclass
class AdaptedState:
    inner:object
    yaw_rate:object
    previous_yaw:object

class RegisteredLowerController(FrozenLowerController):
    def __init__(self,alias,registry_path=None,path_capacity=3201):
        registry_path=Path(registry_path) if registry_path else Path(__file__).resolve().parents[2]/'configs/frozen_lower_registry.json'
        registry=json.loads(registry_path.read_text());entry=registry['models'][alias]
        if alias not in ('STTW_R196_ALPHA1','STTW_R244_ALPHA1') or entry['alpha']!=1.:raise ValueError('frozen ALPHA1 identity required')
        ck=Path(entry['checkpoint']);declaration=json.loads(Path(entry['source_declaration']).read_text());meta=json.loads((ck/'identity.json').read_text())
        for file,key in [('actor.msgpack','actor_sha256'),('identity.json','identity_sha256')]:
            if hashlib.sha256((ck/file).read_bytes()).hexdigest()!=entry[key]:raise ValueError('frozen lower SHA mismatch '+file)
        self.source=declaration['task'];self.identity=declaration['policy_identity'];self.lower_alpha=1.
        assert meta['identity']==self.identity
        self.observation_config=ObservationConfig(**self.source['observation'])
        assert list(observation_fields(self.observation_config))==meta['fields']==entry['frame_fields']
        assert self.observation_config.history_steps==10 and len(meta['fields'])==30 and meta['hidden_sizes']==[128,128,128]
        assert entry['actor_input_size']==310 and entry['action_scales']==[1.5,10.] and entry['control_dt']==.005
        self.policy=load_policy(ck,expected=self.identity);self.tracking=TrackingConfig(**self.source['tracking'])
        self.dt=self.source['controller']['dt'];self.wheelbase=self.source['controller']['wheelbase'];self.caster=self.source['controller']['caster']
        self.capacity=int(path_capacity)
        assert self.capacity>=2001 and self.dt==.005 and self.source.get('learning_roll_reference') is None
        self.provenance=dict(alias=alias,checkpoint=str(ck),identity=self.identity,payload_sha256=entry['actor_sha256'],sidecar_sha256=entry['identity_sha256'],fields=meta['fields'],lower_alpha=1.,path_capacity=self.capacity,transfer='validated causal historical path / endpoint ray + source return rules; body and world yaw kept distinct')
    def initial(self,pose):return AdaptedState(super().initial(pose),jp.asarray(0.),pose[2])
    def features(self,s,pose,governed):
        curvature=jp.cos(self.caster)*jp.tan(governed[1])/self.wheelbase
        timed=errors(pose,s.reference_pose,jp.array([governed[0],governed[0]*curvature]))
        path,extension=path_features(s,pose)
        if self.observation_config.include_timed and not self.tracking.geometric:path=timed[:3]
        return path,timed,extension
    def prepare(self,wrap,measurement,pose,governed,controller_output,previous_final):
        s=wrap.inner;curvature=jp.cos(self.caster)*jp.tan(governed[1])/self.wheelbase
        s=s.replace(path_points=s.path_points.at[0,3].set(jp.where(s.valid_count==1,curvature,s.path_points[0,3])))
        path,timed,extension=self.features(s,pose,governed)
        frame=make_frame(measurement,jp.array([governed[1],governed[0]]),controller_output.reference_roll,controller_output.steer_rate,previous_final,controller_output.disturbance)
        frame=jp.concatenate([frame,path,jp.ones(1),return_observation(s.return_state,self.tracking)])
        if self.observation_config.include_timed:frame=jp.concatenate([frame,jp.array([timed[3],governed[0]*curvature,wrap.yaw_rate])])
        history,obs=advance_history(s.history,frame,self.observation_config);action=self.policy(obs)
        return wrap.replace(inner=s.replace(history=history)),action,obs,dict(finite=jp.all(jp.isfinite(obs))&jp.all(jp.isfinite(action))&~s.overflow,path_endpoint_extension=extension,path_features=path)
    def after_step(self,wrap,governed,pose,measurement,true_speed,action,failed,tick):
        s=wrap.inner;ref=integrate(s.reference_pose,governed,self.dt,self.wheelbase,self.caster)
        curvature=jp.cos(self.caster)*jp.tan(governed[1])/self.wheelbase
        idx=jp.minimum(s.valid_count,self.capacity-1);overflow=s.overflow|(s.valid_count>=self.capacity)
        points=s.path_points.at[idx].set(jp.concatenate([ref,curvature[None]]))
        updated=s.replace(reference_pose=ref,path_points=jp.where(overflow,s.path_points,points),valid_count=jp.minimum(s.valid_count+1,self.capacity),overflow=overflow)
        path,timed,_=self.features(updated,pose,governed)
        dy=pose[2]-wrap.previous_yaw;yaw=jp.arctan2(jp.sin(dy),jp.cos(dy))/self.dt
        kw=dict(longitudinal_error=timed[3],yaw_rate_error=yaw-governed[0]*curvature) if self.tracking.timed else {}
        returned,_=transition(s.return_state,roll=measurement[0],roll_rate=measurement[1],speed_error=true_speed-governed[0],lateral_error=path[0],heading_error=path[1],action=action,alpha=1.,dt=self.dt,alive_rate=self.source['alive_reward_rate'],failure_penalty=self.source['failure_penalty'],failed=failed|overflow,enabled=tick*self.dt>=self.tracking.start_seconds,config=self.tracking,**kw)
        return AdaptedState(updated.replace(return_state=returned),yaw,pose[2])
