"""256/128 LeakyReLU residual Actor and identity-bound inference export.

This is an Actor architecture, not an implementation of the paper's TD3.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from flax import linen as nn, serialization
import jax
import jax.numpy as jp
from .observation import FIELDS, PATH_FIELDS, TRACKING_FIELDS, TIMED_FIELDS
from .action_mapping import ACTION_FIELDS


def make_policy_identity(model_identity,config,history_steps):
    """Bind mesh assets, loader transform and runtime as well as XML/config."""
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    config=dict(config)
    if config.get("timed_reference") is None:config.pop("timed_reference",None)
    else:
        config['timed_reference']=dict(config['timed_reference'])
        if config['timed_reference'].get('recovery_probability',0.)==0.:
            for key in ('recovery_probability','recovery_start','conflict_start_window'):
                config['timed_reference'].pop(key,None)
    if config.get('tracking') is not None:
        config['tracking']=dict(config['tracking'])
        for key,default in [('geometric',False),('reward_mode','gaussian'),('shrink_tolerances',False),('deadline_penalty',0.),('over_deadline_rate',0.)]:
            if config['tracking'].get(key,default)==default:config['tracking'].pop(key,None)

    if config.get("tracking") is not None and not config["tracking"].get("timed",False):
        config["tracking"]=dict(config["tracking"])
        for k in list(config["tracking"]):
            if k=="timed" or k.startswith(("longitudinal_","yaw_rate_","final_longitudinal_","final_yaw_rate_")):config["tracking"].pop(k)
    if config.get("reference_paths") is None:config.pop("reference_paths",None)
    else:
        config['reference_paths']=dict(config['reference_paths'])
        if not config['reference_paths'].get('continuous_projection',False):
            config['reference_paths'].pop('continuous_projection',None)
            config['reference_paths'].pop('projection_margin',None)
    if config.get("speed_schedule") is None:config.pop("speed_schedule",None)
    if config.get("tracking") is None:config.pop("tracking",None)
    if "actuator" in config:
        config["actuator"]=dict(config["actuator"])
        if not config["actuator"].get("project_base",False):config["actuator"].pop("project_base",None)
        if config["actuator"].get("composition","additive")=="additive":
            config["actuator"].pop("composition",None)
    if config.get("motion_commands") is None:config.pop("motion_commands",None)
    else:
        config['motion_commands']=dict(config['motion_commands'])
        if config['motion_commands'].get('residual_gate_min') is None:config['motion_commands'].pop('residual_gate_min',None)
        if config['motion_commands'].get('yaw_tracking_reward_rate',0.)==0.:
            for key in ('yaw_tracking_reward_rate','yaw_tracking_reward_scale'):
                config['motion_commands'].pop(key,None)
        if config['motion_commands'].get('tolerance_penalty_rate',0.)==0.:
            for key in ('tolerance_penalty_rate','speed_tolerance','yaw_tolerance','speed_excess_scale','yaw_excess_scale'):
                config['motion_commands'].pop(key,None)
        if config['motion_commands'].get('tracking_priority_ratio')==100.:
            config['motion_commands'].pop('tracking_priority_ratio')
    if config.get("learning_roll_reference") is None:config.pop("learning_roll_reference",None)
    if config.get("alive_reward_rate")==1.:config.pop("alive_reward_rate")
    if config.get('priority') is None:config.pop('priority',None)
    else:
        config['priority']=dict(config['priority'])
        for key,value in [('weight_schedule','linear'),('weight_base',10.),('tracking_weight_scale',1.),('risk_gate',True),('speed_cost_scale',1.),('path_cost_scale',1.)]:
            if config['priority'].get(key)==value:config['priority'].pop(key)
    if 'observation' in config:
        config['observation']=dict(config['observation'])
        if not config['observation'].get('include_timed',False):config['observation'].pop('include_timed',None)
        if not config['observation'].get('include_tracking',False):config['observation'].pop('include_tracking',None)
        if not config['observation'].get('include_motion',False):config['observation'].pop('include_motion',None)
        if config['observation'].get('include_attitude_risk') is True:config['observation'].pop('include_attitude_risk')
        if config['observation'].get('include_priority') is False:config['observation'].pop('include_priority')
    if config.get('rear_disturbance_mode','torque')=='torque':config.pop('rear_disturbance_mode',None)
    if config.get('bend') is None:config.pop('bend',None)
    if config.get('disturbance_rear_torque',0.)==0:config.pop('disturbance_rear_torque',None)
    if config.get('random_events') is not None:
        config['random_events']=dict(config['random_events'])
        for k,v in [('rear_probability',0.),('rear_min',.02),('rear_max',.08)]:
            if config['random_events'].get(k)==v:config['random_events'].pop(k)
    if config.get("figure_eight") is None:config.pop("figure_eight",None)
    # New optional behavior must not invalidate existing direct-action models.
    if config.get('action_mapping') is None:config.pop('action_mapping',None)
    if config.get('action_mapping') is not None:
        mapping=dict(config['action_mapping'])
        for key,value in [('authority_aware',False),('lateral_weight',1.),('speed_weight',1.),('regularization',1e-6)]:
            if mapping.get(key)==value:mapping.pop(key)
        config['action_mapping']=mapping
    identity={'model_sha256':digest(model_identity),'config_sha256':digest(config),'history_steps':history_steps}
    if config.get('observation',{}).get('include_path',False):
        identity['observation_fields']=list(FIELDS+PATH_FIELDS)
    if (config.get('figure_eight') is not None or config.get('bend') is not None) and config.get('observation',{}).get('include_path',False):
        identity['observation_fields']=list(FIELDS)+['path_right_error','heading_error','path_curvature']
    if config.get('observation',{}).get('include_motion',False):identity['observation_fields']=list(FIELDS)+['yaw_rate_reference','yaw_rate_world']
    if config.get('priority') is not None:
        identity['observation_fields']=identity.get('observation_fields',list(FIELDS))+['speed_priority']+(['attitude_risk'] if config.get('observation',{}).get('include_attitude_risk',True) else [])
    if config.get('observation',{}).get('include_tracking',False):
        identity['observation_fields'][15]='path_lateral_error'
        identity['observation_fields'] += list(TRACKING_FIELDS)
    if config.get('observation',{}).get('include_timed',False):identity['observation_fields'] += list(TIMED_FIELDS)
    if config.get('action_mapping') is not None:identity['action_fields']=ACTION_FIELDS
    if config.get("actuator",{}).get("composition")=="full_range":
        identity["action_fields"]=["steer_available_range_fraction","rear_available_range_fraction"]
    return identity


class ResidualActor(nn.Module):
    hidden_sizes: tuple=(256,128)
    negative_slope: float=.01
    activation: str="leaky_relu"

    @nn.compact
    def __call__(self,observation,return_logits=False):
        x=observation
        for width in self.hidden_sizes:
            x=nn.Dense(width)(x)
            if self.activation=="elu":x=nn.elu(x)
            elif self.activation=="leaky_relu":x=nn.leaky_relu(x,negative_slope=self.negative_slope)
            else:raise ValueError("unsupported actor activation")
        logits=nn.Dense(2)(x)
        return logits if return_logits else jp.tanh(logits)


def save_policy(path,params,mean,std,identity,*,hidden_sizes=(256,128),negative_slope=.01,activation="leaky_relu"):
    path=Path(path)
    mean,std=np.asarray(mean),np.asarray(std)
    fields=identity.get('observation_fields',list(FIELDS))
    size=int(identity['history_steps'])*(len(fields)+1)
    if mean.shape!=(size,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('invalid observation normalization')
    payload=serialization.to_bytes(params)
    meta={'schema':'sttw_actor_v1','identity':identity,'fields':fields,
          'actions':identity.get('action_fields',['steer_rate_residual','rear_rate_residual']),'hidden_sizes':list(hidden_sizes),
          'negative_slope':negative_slope,'activation':activation,'mean':mean.tolist(),'std':std.tolist(),
          'payload_sha256':hashlib.sha256(payload).hexdigest()}
    # Validate dimensions before publishing a checkpoint directory.
    ResidualActor(tuple(hidden_sizes),negative_slope,activation).apply(params,jp.zeros(size))
    path.mkdir(parents=True,exist_ok=False)
    (path/'actor.msgpack').write_bytes(payload)
    (path/'identity.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')


def load_policy(path,*,expected):
    path=Path(path)
    meta=json.loads((path/'identity.json').read_text())
    if meta.get('schema')!='sttw_actor_v1' or meta['identity']!=expected or meta['fields']!=expected.get('observation_fields',list(FIELDS)) or meta['actions']!=expected.get('action_fields',['steer_rate_residual','rear_rate_residual']):
        raise ValueError('policy observation/action/identity mismatch')
    payload=(path/'actor.msgpack').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=meta['payload_sha256']:
        raise ValueError('policy payload hash mismatch')
    mean,std=np.asarray(meta['mean']),np.asarray(meta['std'])
    size=int(expected['history_steps'])*(len(expected.get('observation_fields',FIELDS))+1)
    if mean.shape!=(size,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('invalid normalization')
    actor=ResidualActor(tuple(meta['hidden_sizes']),meta['negative_slope'],meta.get('activation','leaky_relu'))
    template=actor.init(jax.random.PRNGKey(0),jp.zeros(size))
    params=serialization.from_bytes(template,payload)
    return jax.jit(lambda obs: actor.apply(params,(obs-jp.asarray(mean))/jp.asarray(std)))
