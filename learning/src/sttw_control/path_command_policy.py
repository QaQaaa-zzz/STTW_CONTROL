"""Geometric schema v1; physical correction filter shared by train/evaluate."""
import jax.numpy as j
from flax import linen as nn
SCHEMA='sttw_geometric_path_actor351_v1'

def map_latent(z,config):
    c=config['residual_interface'];a=j.tanh(z)
    return j.stack([a[0]*j.where(a[0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),a[1]*c['steer_scale_rad']])

def correction_tick(filtered,offsets,z,nominal,config):
    c=config['residual_interface'];dt=config['timing']['lower_dt_s']
    target=map_latent(z,config);a=j.exp(-dt/j.array(c['filter_tau_s']))
    f=a*filtered+(1-a)*target
    increment=j.clip(f-offsets,j.array([-c['correction_speed_down_m_s2'],-c['correction_steer_rate_rad_s']])*dt,j.array([c['correction_speed_up_m_s2'],c['correction_steer_rate_rad_s']])*dt)
    before=nominal+offsets+increment
    governed=j.clip(before,j.array([c['speed_reference_bounds_m_s'][0],-c['steer_reference_bound_rad']]),j.array([c['speed_reference_bounds_m_s'][1],c['steer_reference_bound_rad']]))
    clipped=governed!=before;out=governed-nominal
    return j.where(clipped,out,f),out,governed,dict(reference_clipped=j.any(clipped),reference_clip_channels=clipped,rate_clipped=j.any(increment!=f-offsets),policy_fault=~j.all(j.isfinite(z)),target=target)

def validate_actor_input(obs):
    if obs.shape[-1]!=351:raise ValueError('geometric schema requires 351 inputs; legacy345 incompatible')

def assemble_observation(frames,mask,context,frame_scales,config):
    scales=j.array([x['scale'] for x in config['network']['context_fields']])
    obs=j.concatenate([(frames/frame_scales).reshape(-1),mask,context/scales]);validate_actor_input(obs)
    return j.clip(obs,-5,5),~j.all(j.isfinite(obs))

class PathCommandActor(nn.Module):
    @nn.compact
    def __call__(self,obs):
        validate_actor_input(obs)
        x=obs
        for width in (128,128,64):x=nn.elu(nn.Dense(width)(x))
        return nn.Dense(2,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(x)

class PathCommandCritic(nn.Module):
    @nn.compact
    def __call__(self,obs):
        if obs.shape[-1]!=352:raise ValueError('geometric critic352 required')
        x=obs
        for width in (128,128,64):x=nn.elu(nn.Dense(width)(x))
        return nn.Dense(1)(x)
