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
from .observation import FIELDS, PATH_FIELDS
from .action_mapping import ACTION_FIELDS


def make_policy_identity(model_identity,config,history_steps):
    """Bind mesh assets, loader transform and runtime as well as XML/config."""
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    config=dict(config)
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
    if config.get('figure_eight') is not None and config.get('observation',{}).get('include_path',False):
        identity['observation_fields']=list(FIELDS)+['path_right_error','heading_error','path_curvature']
    if config.get('action_mapping') is not None:identity['action_fields']=ACTION_FIELDS
    return identity


class ResidualActor(nn.Module):
    hidden_sizes: tuple=(256,128)
    negative_slope: float=.01

    @nn.compact
    def __call__(self,observation,return_logits=False):
        x=observation
        for width in self.hidden_sizes:
            x=nn.leaky_relu(nn.Dense(width)(x),negative_slope=self.negative_slope)
        logits=nn.Dense(2)(x)
        return logits if return_logits else jp.tanh(logits)


def save_policy(path,params,mean,std,identity,*,hidden_sizes=(256,128),negative_slope=.01):
    path=Path(path)
    mean,std=np.asarray(mean),np.asarray(std)
    fields=identity.get('observation_fields',list(FIELDS))
    size=int(identity['history_steps'])*(len(fields)+1)
    if mean.shape!=(size,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('invalid observation normalization')
    payload=serialization.to_bytes(params)
    meta={'schema':'sttw_actor_v1','identity':identity,'fields':fields,
          'actions':identity.get('action_fields',['steer_rate_residual','rear_rate_residual']),'hidden_sizes':list(hidden_sizes),
          'negative_slope':negative_slope,'mean':mean.tolist(),'std':std.tolist(),
          'payload_sha256':hashlib.sha256(payload).hexdigest()}
    # Validate dimensions before publishing a checkpoint directory.
    ResidualActor(tuple(hidden_sizes),negative_slope).apply(params,jp.zeros(size))
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
    actor=ResidualActor(tuple(meta['hidden_sizes']),meta['negative_slope'])
    template=actor.init(jax.random.PRNGKey(0),jp.zeros(size))
    params=serialization.from_bytes(template,payload)
    return jax.jit(lambda obs: actor.apply(params,(obs-jp.asarray(mean))/jp.asarray(std)))
