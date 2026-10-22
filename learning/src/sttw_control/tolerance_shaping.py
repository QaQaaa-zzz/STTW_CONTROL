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
