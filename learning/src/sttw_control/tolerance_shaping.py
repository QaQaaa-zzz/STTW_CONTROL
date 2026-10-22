"""Experimental analytic reserve cost, not a certified policy tolerance.

Scalar frozen-speed model only. UNKNOWN is explicit; callers must propagate
it as an engineering fault. This module does not modify existing rewards.
"""
import jax.numpy as j

def model_reserve(*,z,b,lam,remaining,effort_budget,gamma,duration,target_force,weight,static=False,enabled=True):
 if not enabled:
  return dict(plus_N=j.array(0.),minus_N=j.array(0.),cost_rate=j.array(0.),valid=j.array(True),enabled=j.array(False))
 inputs=j.stack([j.asarray(x) for x in [z,b,lam,remaining,effort_budget,gamma,duration,target_force,weight]])
 valid=j.all(j.isfinite(inputs))&(lam>0)&(remaining>0)&(remaining>=duration)&(effort_budget>=0)&(duration>0)&(target_force>0)&(weight>=0)&(b!=0)&(gamma!=0)
 # Safe denominators prevent incidental inf/NaN during tracing; valid is retained.
 safe_lam=j.where(lam>0,lam,1.);T=j.minimum(j.maximum(remaining,0.),1.)
 W=b*b*(-j.expm1(-2*safe_lam*T))/(2*safe_lam)
 K=gamma*(-j.expm1(-safe_lam*duration))/safe_lam
 valid=valid&j.isfinite(W)&(W>0)&j.isfinite(K)&(j.abs(K)>0)
 coordinate=j.asarray(0.) if static else j.asarray(z)
 radius=j.sqrt(j.maximum(effort_budget*W,0.));feasible=j.abs(coordinate)<=radius
 denom=j.where(j.abs(K)>0,j.abs(K),1.)
 plus=j.where(feasible,j.maximum(0.,(radius-j.sign(K)*coordinate)/denom),0.)
 minus=j.where(feasible,j.maximum(0.,(radius+j.sign(K)*coordinate)/denom),0.)
 cost=weight*j.clip(1-j.minimum(plus,minus)/j.where(target_force>0,target_force,1.),0.,1.)
 return dict(plus_N=j.where(valid,plus,j.nan),minus_N=j.where(valid,minus,j.nan),cost_rate=j.where(valid,cost,j.nan),valid=valid,enabled=j.array(True))


def tolerance_enabled(config):
 spec=config['path_reward'].get('tolerance')
 return bool(spec and spec.get('weight',0)!=0)

def validate_tolerance_config(config):
 import math
 if not tolerance_enabled(config):return
 c=config['path_reward'];s=c['tolerance']
 if s['method'] not in ('static','dynamic'):raise ValueError('invalid tolerance method')
 for k in ('weight','gamma','effort_budget','duration','target_force'):
  if not math.isfinite(s[k]):raise ValueError('nonfinite tolerance configuration')
 if min(s[k] for k in ('weight','effort_budget','duration','target_force'))<=0 or s['gamma']==0 or s['duration']>1:raise ValueError('invalid tolerance parameters')
 if c['component_caps'].get('tolerance')!=s['weight']:raise ValueError('tolerance cap must match weight')
 if not math.isclose(sum(c['component_caps'].values()),c['failure_cost_rate_bound'],rel_tol=0,abs_tol=1e-6):raise ValueError('failure cost bound must match caps')

def measurement_reserve(measurement,controller_state,controller_config,*,remaining,config):
 import jax
 from .controller import _system
 if not tolerance_enabled(config):
  return model_reserve(z=0.,b=0.,lam=1.,remaining=1.,effort_budget=0.,gamma=0.,duration=.1,target_force=1.,weight=0.,enabled=False)
 spec=config['path_reward']['tolerance'];m=measurement
 def active(_):
  a1,a2,a4,_,_=_system(m[5]*.1,controller_state.gains,controller_config)
  lam=j.sqrt(a4);z=a2/lam*m[2]+lam*m[0]+m[1]+controller_state.disturbance/lam
  return model_reserve(z=z,b=a1+a2/lam,lam=lam,remaining=remaining,static=spec['method']=='static',**{k:spec[k] for k in ('effort_budget','gamma','duration','target_force','weight')})
 def inactive(_):
  return model_reserve(z=0.,b=0.,lam=1.,remaining=1.,effort_budget=0.,gamma=0.,duration=.1,target_force=1.,weight=0.,enabled=False)
 result=jax.lax.cond(remaining>=spec['duration'],active,inactive,None)
 result['valid']=result['valid']&j.isfinite(remaining)&(remaining>=0)
 return result
