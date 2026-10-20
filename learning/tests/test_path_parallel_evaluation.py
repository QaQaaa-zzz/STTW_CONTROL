"""Parallel alpha dispatch preserves saved panel identity and exact lane inventory."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from test_path_conditional_reporting import config,panels,actor_parameters,ALPHAS
from sttw_control.path_command_evaluation import evaluate_conditional


def test_parallel_grid_uses_one_dispatch_and_preserves_every_trace(tmp_path):
 cfg=config();cfg['evaluation_execution']='alpha_case_vmap_v1';expected=panels()
 class Evaluator:
  config=cfg
  calls=[]
  def evaluate(self,params,alpha,output,**kwargs):
   self.calls.append(kwargs)
   assert kwargs['alpha_grid']==ALPHAS
   return {key+'/'+name:trace for key,cases in expected.items() for name,trace in cases.items()}
 e=Evaluator();actual=evaluate_conditional(e,actor_parameters(),ALPHAS,tmp_path/'panel')
 assert len(e.calls)==1 and actual.keys()==expected.keys()
 for key in actual:
  for name in actual[key]:
   for field in actual[key][name]:np.testing.assert_array_equal(actual[key][name][field],expected[key][name][field])


def test_lightweight_parallel_does_not_require_full_grid_or_save_full_score(tmp_path):
 cfg=config();cfg['evaluation_execution']='alpha_case_vmap_v1';expected=panels();names=['left90_R2_2.6','right90_R2_2.6']
 class Evaluator:
  config=cfg
  def evaluate(self,params,alpha,output,**kwargs):
   assert kwargs['case_names']==names
   return {key+'/'+name:expected[key][name] for key in ('alpha_0','alpha_1') for name in names}
 result=evaluate_conditional(Evaluator(),actor_parameters(),(0.,1.),tmp_path/'light',case_names=names,diagnostic_only=True)
 assert set(result)=={'alpha_0','alpha_1'}
 assert not (tmp_path/'light/task_metrics.json').exists()


def test_real_compiled_parallel_lanes_match_serial_and_resolve_speed_causally():
 import jax
 import jax.numpy as j
 from flax import struct
 from sttw_control.path_command_evaluation import BatchedPanelEvaluator
 from sttw_control.path_command_policy import make_path_actor
 @struct.dataclass
 class Physical:failed:object
 @struct.dataclass
 class State:
  alpha:object;tick:object;physical:object;fault:object;domain_exit:object
 reset_calls=[]
 class Env:
  terminal_contract='terminal_contract_v2'
  def reset(self,snapshot,path,v,a):
   jax.debug.callback(lambda _:reset_calls.append(1),a)
   return State(a,j.int32(0),Physical(j.bool_(False)),j.bool_(False),j.bool_(False))
  def observation(self,state,path,v,h):return j.zeros(351).at[336].set(state.alpha).at[337].set(v),j.zeros(352),j.bool_(False)
  def set_log_template(self,*args):pass
  def policy_step(self,state,z,path,v,h,*,evaluation=False):
   def tick(st,_):return st.replace(tick=st.tick+1),dict(active_tick=j.bool_(True),v_user=v,upper_alpha=st.alpha)
   return jax.lax.scan(tick,state,None,length=4)
 cfg={'network':{'architecture':'alpha_modulated_v1'},'timing':{'upper_dt_s':.02,'lower_dt_s':.005}}
 evaluator=BatchedPanelEvaluator(Env(),None,cfg);params=make_path_actor(cfg).init(jax.random.PRNGKey(0),j.zeros(351));paths=j.ones((3,2))
 speeds=j.ones(3)*2.;alpha=j.float32(.5)
 r,c=evaluator._compiled_group(paths,speeds,alpha,params,1.)
 jax.effects_barrier();assert reset_calls==[], 'compilation must not execute reset'
 _,serial=c(r(paths,speeds,alpha),paths,speeds,params)
 vector=j.ones(3)*.5;r,c=evaluator._compiled_group(paths,speeds,vector,params,1.)
 _,parallel=c(r(paths,speeds,vector),paths,speeds,params)
 for field in serial:np.testing.assert_array_equal(serial[field],parallel[field])
 scripts=j.tile(j.array([2.,2.8,2.3,.2,.6]),(3,1));r,c=evaluator._compiled_group(paths,scripts,vector,params,1.)
 _,events=c(r(paths,scripts,vector),paths,scripts,params)
 values=np.asarray(events['v_user'])[:,0,0]
 np.testing.assert_allclose(values[:10],2.);np.testing.assert_allclose(values[10:30],2.8);np.testing.assert_allclose(values[30:],2.3)
 # All four lower ticks share exactly the current policy request, including boundaries.
 np.testing.assert_array_equal(np.asarray(events['v_user']),np.broadcast_to(values[:,None,None],(50,3,4)))


def test_evaluation_path_cache_reuses_original_route_without_changing_knots(monkeypatch):
 from types import SimpleNamespace
 from sttw_control.path_command_evaluation import BatchedPanelEvaluator
 from sttw_control import geometric_path
 calls=[];original=object()
 monkeypatch.setattr(geometric_path,'build_path',lambda *a:(calls.append(a),original)[1])
 env=SimpleNamespace(physics=SimpleNamespace(helpers=SimpleNamespace(pose=lambda _:np.zeros(3))))
 e=BatchedPanelEvaluator(env,SimpleNamespace(data=None),{})
 assert e.path_for_route('left90_R2') is original
 assert e.path_for_route('left90_R2') is original
 assert len(calls)==1
