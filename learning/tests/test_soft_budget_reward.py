import importlib.util
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
import jax
import jax.numpy as jp
from sttw_control.tracking_reward import TrackingConfig,initial_return,transition
s=importlib.util.spec_from_file_location('soft_reference',Path(__file__).parent/'fixtures/soft_budget_reference.py')
import sys
ref=importlib.util.module_from_spec(s);sys.modules[s.name]=ref;s.loader.exec_module(ref)

def config():
 return TrackingConfig(objective='soft_budget_v1',geometric=True,reward_mode='huber',tracking_rate=8.,speed_scale=.05,lateral_scale=.1,heading_scale=.15,tail_rate=0.,return_rate=0.,return_bonus=0.,speed_tight=.05,final_speed_tolerance=.05,final_overspeed_tolerance=.05,overspeed_band=.05,shrink_tolerances=True,deadline_penalty=5.,overdue_rate=2.)

def test_reference_trajectory_and_jax():
 c=config();rng=np.random.default_rng(7);state=initial_return(xp=np);reference=ref.initial_state()
 for i in range(1000):
  kw=dict(speed_error=rng.uniform(-.5,.2),lateral_error=rng.uniform(-.5,.5),heading_error=rng.uniform(-.3,.3),roll=rng.uniform(-.35,.35),roll_rate=rng.uniform(-.5,.5),action=rng.uniform(-1,1,2),alpha=float(rng.choice([0,.5,1])),enabled=i>10,recovery_trigger=i in [50,200],clock_from_departure=False)
  reference,reward,parts,raw,diag=ref.transition(reference,**kw)
  state,actual=transition(state,**kw,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,config=c,xp=np)
  np.testing.assert_allclose(sum(actual.values()),reward,atol=1e-9)
  for key in parts:np.testing.assert_allclose(actual[key],parts[key],atol=1e-9)
  np.testing.assert_allclose([state.elapsed,state.hold],[reference.elapsed_ticks*.005,reference.hold_ticks*.005],atol=1e-9)
  assert state.deadline_missed==reference.deadline_missed
 kw.update(alpha=1.)
 a,p=transition(state,**kw,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,config=c,xp=np)
 b,q=jax.jit(lambda x:transition(x,**kw,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,config=c,xp=jp))(state)
 np.testing.assert_allclose([p[k] for k in sorted(p)],[q[k] for k in sorted(p)],atol=1e-7)

def test_old_identity_excludes_new_defaults():
 from sttw_control.network import make_policy_identity
 c=asdict(TrackingConfig());old={k:v for k,v in c.items() if not k.startswith('soft_')}
 assert make_policy_identity({},dict(tracking=c),10)==make_policy_identity({},dict(tracking=old),10)

def test_candidates_grace_raw_and_failure():
 from sttw_control.tracking_reward import soft_budget_costs
 c=config();actual=[]
 for alpha in [0,.5,1]:
  actual.append([sum(soft_budget_costs(-u,y,0.,0.,0.,np.zeros(2),np.zeros(2),alpha,.5,True,c,xp=np)[0].values()) for u,y in [(.2,.04),(.09,.12),(.04,.28)]])
 np.testing.assert_allclose(actual,[[1.28,11.28,42],[28.64,16,21.14],[66,22.08,5.12]],atol=1e-8)
 for tau in [0.,1.,2.5]:
  _,bands=soft_budget_costs(0.,0.,0.,0.,0.,np.zeros(2),np.zeros(2),1.,tau,True,c,xp=np)
  assert bands['path_band']==(.4 if tau<=1 else .1)
 for bad in [np.nan,np.inf]:
  _,p=transition(initial_return(xp=np),speed_error=bad,lateral_error=0.,heading_error=0.,roll=0.,roll_rate=0.,action=np.zeros(2),alpha=1.,enabled=True,failed=False,dt=.005,alive_rate=1.,failure_penalty=200.,config=c,xp=np)
  assert sum(p.values())==-200 and all(v==0 for k,v in p.items() if k!='failure')


def test_final_hold_tick_boundary_and_failure_order():
 c=config();a=initial_return(xp=np);b=ref.initial_state()
 for tick in range(900):
  err=.2 if tick<602 else 0.
  kw=dict(speed_error=err,lateral_error=err,heading_error=0.,roll=0.,roll_rate=0.,action=np.zeros(2),alpha=1.,enabled=True,clock_from_departure=False,recovery_trigger=tick in [0,100,605])
  a,p=transition(a,**kw,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,config=c,xp=np)
  b,r,q,_,_=ref.transition(b,**kw)
  assert bool(a.pending)==bool(b.pending) and bool(a.deadline_missed)==bool(b.deadline_missed)
  np.testing.assert_allclose(sum(p.values()),r,atol=1e-8)
 gamma=.9995;discount=gamma**np.arange(2000)
 assert -.0495*discount.sum()-5 > max(.0005*discount[:i].sum()-200*discount[i] for i in range(2000))
