import math
import numpy as np
import jax
import jax.numpy as j
from sttw_control.tolerance_shaping import model_reserve

def kwargs():
 return dict(z=.2,b=-8.,lam=6.,remaining=3.,effort_budget=.03,gamma=.81556,duration=.1,target_force=6.,weight=10.,static=False,enabled=True)

def test_endpoints_match_quadratic_effort_boundary():
 k=kwargs();x=model_reserve(**k)
 W=k['b']**2*(-math.expm1(-2*k['lam']))/(2*k['lam']);K=k['gamma']*(-math.expm1(-k['lam']*k['duration']))/k['lam']
 assert bool(x['valid'])
 for sign,key in [(1,'plus_N'),(-1,'minus_N')]:
  f=float(x[key]);energy=(k['z']+sign*K*f)**2/W
  np.testing.assert_allclose(energy,k['effort_budget'],rtol=2e-6)
  assert (k['z']+sign*K*(f+.01))**2/W>k['effort_budget']

def test_direction_and_state_sign_symmetry():
 k=kwargs();a=model_reserve(**k);k['z']=-k['z'];b=model_reserve(**k)
 np.testing.assert_allclose(a['plus_N'],b['minus_N']);np.testing.assert_allclose(a['cost_rate'],b['cost_rate'])
 k['gamma']=-k['gamma'];c=model_reserve(**k);np.testing.assert_allclose(a['plus_N'],c['plus_N'])

def test_insufficient_budget_and_static_comparator():
 k=kwargs();k['z']=100.;x=model_reserve(**k)
 assert bool(x['valid']) and float(x['plus_N'])==0 and float(x['minus_N'])==0 and float(x['cost_rate'])==10
 k['static']=True;x=model_reserve(**k);assert float(x['plus_N'])>0

def test_unknown_is_not_safe_and_disabled_bypasses_input():
 for key,value in [('b',0.),('gamma',0.),('remaining',0.),('remaining',.05),('z',float('nan'))]:
  k=kwargs();k[key]=value;x=model_reserve(**k)
  assert not bool(x['valid']) and np.isnan(float(x['cost_rate']))
 k['enabled']=False;x=model_reserve(**k);assert bool(x['valid']) and float(x['cost_rate'])==0

def test_jit_batch_and_cost_bounds():
 k=kwargs();k.pop('z');f=jax.jit(jax.vmap(lambda z:model_reserve(z=z,**k)))
 x=f(j.linspace(-2,2,21));assert np.asarray(x['valid']).all()
 assert np.all(np.asarray(x['cost_rate'])>=0) and np.all(np.asarray(x['cost_rate'])<=10)
