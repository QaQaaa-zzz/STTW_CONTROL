import copy
import json
from pathlib import Path
import pytest
from sttw_control.path_command_fork import reward_validation_schedule, validate_reward_resume
from sttw_control.path_command_training import load_plan
ROOT=Path(__file__).parents[2]

def extended():
 p,c,s=load_plan(ROOT/'learning/configs/path_safety_control_run.json')
 p['updates']=200
 p['authorized_budget']={'total_updates':200,'checkpoint_updates':[50,100,150,200],'resume_boundary':50}
 return p,c,s

def test_legacy50_and_explicit200_schedule():
 p,c,s=load_plan(ROOT/'learning/configs/path_safety_control_run.json')
 assert reward_validation_schedule(p,c)==[50]
 p,c,s=extended();assert reward_validation_schedule(p,c)==[50,100,150,200]
 for bad in [100,201]:
  q=copy.deepcopy(p);q['updates']=bad
  with pytest.raises(ValueError):reward_validation_schedule(q,c)
 q=copy.deepcopy(p);q.pop('authorized_budget')
 with pytest.raises(ValueError):reward_validation_schedule(q,c)
 q=copy.deepcopy(p);q['authorized_budget']['checkpoint_updates']=[200]
 with pytest.raises(ValueError):reward_validation_schedule(q,c)

def test_resume_requires_completed_matching50_stage(tmp_path):
 p,c,s=extended();p['resume_stage']=str(tmp_path)
 old=copy.deepcopy(p);old.pop('authorized_budget');old.pop('resume_stage');old['updates']=50
 (tmp_path/'manifest.json').write_text(json.dumps({'plan':old,'config':c,'source':s}))
 (tmp_path/'status.json').write_text(json.dumps({'state':'completed','endpoint_updates':{'alpha0':50,'alpha1':50}}))
 for a in [0,1]:
  d=tmp_path/f'alpha{a}';d.mkdir();(d/'training_status.json').write_text(json.dumps({'state':'completed','completed_updates':50,'declared_updates':50}))
 validate_reward_resume(p,c,s,tmp_path)
 with pytest.raises(ValueError):validate_reward_resume(p,c,s,None)
 wrong=copy.deepcopy(p);wrong['experiment_arm']='candidate'
 with pytest.raises(ValueError):validate_reward_resume(wrong,c,s,tmp_path)
 (tmp_path/'status.json').write_text(json.dumps({'state':'running','endpoint_updates':{'alpha0':50,'alpha1':49}}))
 with pytest.raises(ValueError):validate_reward_resume(p,c,s,tmp_path)

def test_200_plans_preserve_checkpoint_identity_config_and_selection():
 for arm in ['control','candidate']:
  old,c,s=load_plan(ROOT/f'learning/configs/path_safety_{arm}_run.json')
  new,nc,ns=load_plan(ROOT/f'learning/configs/path_safety_{arm}_200_run.json')
  assert c==nc and s==ns
  for k in ['seed','initialization','sampler_resolution','best_resolution','experiment_arm']:assert old[k]==new[k]
  assert new['updates']==200 and reward_validation_schedule(new,nc)==[50,100,150,200]


def test_full_restore_retains_adam_lr_critic_rng_and_rejections(tmp_path):
 import random
 import numpy as np
 import torch
 from types import SimpleNamespace
 from sttw_control.path_command_training import restore_checkpoint
 from sttw_control.path_command_policy import SCHEMA
 class Policy(torch.nn.Module):
  def __init__(self):
   super().__init__();self.actor=torch.nn.Linear(3,2);self.critic=torch.nn.Linear(4,1);self.log_std=torch.nn.Parameter(torch.zeros(2))
 p=Policy();opt=torch.optim.Adam([{'params':p.actor.parameters(),'lr':3e-7},{'params':p.critic.parameters(),'lr':8e-5},{'params':[p.log_std],'lr':3e-7}])
 sum(v.square().sum() for v in p.parameters()).backward();opt.step()
 torch.manual_seed(911101);np.random.seed(911101);random.seed(911101)
 saved={'schema':SCHEMA,'identity':{'fixed':'identity'},'update':50,'policy':p.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'cuda_rng':None,'consecutive_rejections':2,'route_counters':np.arange(8)}
 f=tmp_path/'u50.pt';torch.save(saved,f)
 q=Policy();qo=torch.optim.Adam([{'params':q.actor.parameters(),'lr':.1},{'params':q.critic.parameters(),'lr':.1},{'params':[q.log_std],'lr':.1}])
 a=SimpleNamespace(policy=q,optimizer=qo,device='cpu');restored=restore_checkpoint(f,a,saved['identity'])
 for k,v in p.state_dict().items():assert torch.equal(v,q.state_dict()[k])
 assert qo.state_dict()['param_groups']==saved['optimizer']['param_groups']
 for k,v in saved['optimizer']['state'].items():
  for name,t in v.items():assert torch.equal(t,qo.state_dict()['state'][k][name])
 assert torch.equal(torch.get_rng_state(),saved['torch_rng']) and random.getstate()==saved['python_rng']
 np.testing.assert_array_equal(np.random.get_state()[1],saved['numpy_rng'][1])
 assert restored['consecutive_rejections']==2 and restored['update']==50
 np.testing.assert_array_equal(restored['route_counters'],np.arange(8))
 with pytest.raises(ValueError):restore_checkpoint(f,a,{'fixed':'wrong'})
