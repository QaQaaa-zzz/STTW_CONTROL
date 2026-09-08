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
from .observation import FIELDS


def make_policy_identity(model_identity,config,history_steps):
    """Bind mesh assets, loader transform and runtime as well as XML/config."""
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return {'model_sha256':digest(model_identity),'config_sha256':digest(config),'history_steps':history_steps}


class ResidualActor(nn.Module):
    hidden_sizes: tuple=(256,128)
    negative_slope: float=.01

    @nn.compact
    def __call__(self,observation):
        x=observation
        for width in self.hidden_sizes:
            x=nn.leaky_relu(nn.Dense(width)(x),negative_slope=self.negative_slope)
        return jp.tanh(nn.Dense(2)(x))


def save_policy(path,params,mean,std,identity,*,hidden_sizes=(256,128),negative_slope=.01):
    path=Path(path)
    mean,std=np.asarray(mean),np.asarray(std)
    size=int(identity['history_steps'])*(len(FIELDS)+1)
    if mean.shape!=(size,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('invalid observation normalization')
    payload=serialization.to_bytes(params)
    meta={'schema':'sttw_actor_v1','identity':identity,'fields':list(FIELDS),
          'actions':['steer_rate_residual','rear_rate_residual'],'hidden_sizes':list(hidden_sizes),
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
    if meta.get('schema')!='sttw_actor_v1' or meta['identity']!=expected or meta['fields']!=list(FIELDS) or meta['actions']!=['steer_rate_residual','rear_rate_residual']:
        raise ValueError('policy observation/action/identity mismatch')
    payload=(path/'actor.msgpack').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=meta['payload_sha256']:
        raise ValueError('policy payload hash mismatch')
    mean,std=np.asarray(meta['mean']),np.asarray(meta['std'])
    size=int(expected['history_steps'])*(len(FIELDS)+1)
    if mean.shape!=(size,) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('invalid normalization')
    actor=ResidualActor(tuple(meta['hidden_sizes']),meta['negative_slope'])
    template=actor.init(jax.random.PRNGKey(0),jp.zeros(size))
    params=serialization.from_bytes(template,payload)
    return jax.jit(lambda obs: actor.apply(params,(obs-jp.asarray(mean))/jp.asarray(std)))
