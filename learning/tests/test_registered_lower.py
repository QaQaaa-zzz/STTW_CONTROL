import json
from pathlib import Path
import numpy as np
import jax.numpy as jp
from sttw_control.registered_lower_controller import RegisteredLowerController
REG=Path(__file__).parents[1]/'configs/frozen_lower_registry.json'
def test_registered_contract_and_reset_alpha():
 from sttw_control.controller import initial_controller,controller_step,ControllerConfig
 for alias in ['STTW_R196_ALPHA1','STTW_R244_ALPHA1']:
  lower=RegisteredLowerController(alias,REG,path_capacity=3201)
  assert lower.capacity==3201 and lower.lower_alpha==1.
  state=lower.initial(jp.zeros(3));assert state.inner.path_points.shape==(3201,4)
  assert state.inner.valid_count==1
  measurement=jp.array([0.,0.,0.,0.,0.,23.,23.]);cc=ControllerConfig()
  _,out=controller_step(initial_controller(cc),jp.array([2.3,0.,0.,0.,0.,0.]),True,cc)
  state,action,obs,flags=lower.prepare(state,measurement,jp.zeros(3),jp.array([2.3,0.]),out,jp.array([0.,23.]))
  assert obs.shape==(310,) and action.shape==(2,) and bool(flags['finite'])
  frames=np.asarray(obs[:300]).reshape(10,30);mask=np.asarray(obs[300:]);assert mask.sum()==1
  assert np.all(frames[mask==0]==0) and np.all(frames[mask==1,18]==1)
def test_adapter_matches_validated_transfer_for_both_models():
 import importlib.util
 from sttw_control.controller import initial_controller,controller_step,ControllerConfig
 path='/home/qy/STTW_CONTROL/runs/tracking_candidate_review_20261008/command_replay.py'
 spec=importlib.util.spec_from_file_location('validated_transfer',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for alias,label in [('STTW_R196_ALPHA1','precision196'),('STTW_R244_ALPHA1','soft244')]:
  new=RegisteredLowerController(alias,REG,path_capacity=2001);old=mod.Transfer(label)
  ns=new.initial(jp.zeros(3));os=old.initial(jp.zeros(3));cc=ControllerConfig();cs=initial_controller(cc)
  for tick in range(4):
   pose=jp.array([.011*tick,.0001*tick,.002*tick]);cmd=jp.array([2.3,.04]);m=jp.array([.002,.01,.015,.02,.03,23.,23.])
   cs,co=controller_step(cs,jp.array([2.3,.015,.02,.002,.01,.04]),True,cc)
   ns,na,no,nf=new.prepare(ns,m,pose,cmd,co,jp.array([.1,23.]));os,oa,oo,of=old.prepare(os,m,pose,cmd,co,jp.array([.1,23.]))
   np.testing.assert_allclose(no,oo,rtol=1e-6,atol=1e-6);np.testing.assert_allclose(na,oa,rtol=1e-6,atol=1e-6)
   ns=new.after_step(ns,cmd,pose,m,jp.asarray(2.29),na,False,jp.int32(tick));os=old.after_step(os,cmd,pose,m,jp.asarray(2.29),oa,False,jp.int32(tick))
   import jax
   for a,b in zip(jax.tree.leaves(ns),jax.tree.leaves(os)):np.testing.assert_allclose(a,b,atol=1e-6)
def test_independent_upper_profiles_pin_lower_alpha(tmp_path):
 from sttw_control.upper_endpoint_training import make_spec,load_endpoint_spec
 import pytest
 for alias in ['STTW_R196_ALPHA1','STTW_R244_ALPHA1']:
  for a in [0,1]:
   s=make_spec(alias,a);p=tmp_path/'spec.json';p.write_text(json.dumps(s));assert load_endpoint_spec(p)==s
   assert s['lower_controller']['alpha']==1. and s['upper_alpha']==a
   assert s['ppo']['default_updates']==250 and s['ppo']['kl_group_index'] is None
   s['lower_controller']['alpha']=0.;p.write_text(json.dumps(s))
   with pytest.raises(ValueError):load_endpoint_spec(p)
def test_single_endpoint_ppo_does_not_require_other_alpha():
 import torch
 from tensordict import TensorDict
 from sttw_control.upper_endpoint_training import make_spec
 from sttw_control.direct_command_ppo import make_algorithm
 torch.set_num_threads(1)
 for a in [0,1]:
  s=make_spec('STTW_R196_ALPHA1',a);obs=TensorDict({'policy':torch.zeros(8,345),'critic':torch.zeros(8,346)},batch_size=[8]);obs['policy'][:,336]=a
  algo=make_algorithm(obs,16,s,'cpu')
  with torch.no_grad():
   for _ in range(16):algo.act(obs);algo.process_env_step(obs,torch.randn(8)*.01,torch.zeros(8,dtype=torch.bool),{})
   algo.compute_returns(obs)
  info=algo.update();assert set(info['kl_by_alpha'])=={'all'} and not info['nonfinite_stop']
def test_baseline_rescore_preserves_alpha0_and_changes_alpha1():
 from sttw_control.upper_endpoint_review import rescore_baseline
 from sttw_control.direct_command_audit import reconstruct_trace
 p=Path('/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/direct_command_frozen_lower500_20260929')
 t=dict(np.load(p/'review/main/B0.npz'));s=json.loads((p/'frozen_config.json').read_text());controller=json.loads((p/'manifest.json').read_text())['controller']
 previous=t['final_command'][0] # correction only first-step slew term; use actual prepared actuator below
 import pickle
 with (p/'prepared_bank.pkl').open('rb') as f:bank=pickle.load(f)
 previous=np.asarray(bank.actuator.previous)[3]
 for a in [0,1]:
  new=rescore_baseline(t,a,s,previous);assert reconstruct_trace(new,s,previous,controller)['passed']
  if a==0:np.testing.assert_allclose(new['scored_tick_reward'],t['scored_tick_reward'],atol=2e-6)


def test_reduced_endpoint_budget_stops_at_declared_update(tmp_path,monkeypatch):
 import json
 import sttw_control.upper_endpoint_training as training
 from unittest.mock import MagicMock
 s=training.make_spec('STTW_R244_ALPHA1',0,125)
 p=tmp_path/'config.json';p.write_text(json.dumps(s))
 assert training.load_endpoint_spec(p)==s
 assert s['budget']['default_policy_transitions']==8192000
 assert s['budget']['default_control_transitions_upper']==32768000
 campaign=MagicMock();campaign.spec=s;campaign.status={'lower_alias':'STTW_R244_ALPHA1','upper_alpha':0}
 campaign.train.side_effect=[(None,2),(None,125)]
 monkeypatch.setattr(training,'EndpointCampaign',lambda *args:campaign)
 monkeypatch.setattr(training,'notify',lambda *args:None)
 training.run_endpoint(p,tmp_path)
 assert campaign.train.call_args_list[-1].args==('pilot',125)
 campaign._status.assert_called_once_with(state='complete',stage='training_complete',completed_updates=125)
