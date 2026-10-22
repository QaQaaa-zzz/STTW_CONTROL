import copy,json,pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sttw_control.tolerance_training import validate_training_config,restore_weights_only
from sttw_control.preference_best import digest
from sttw_control.path_command_policy import SCHEMA

BASE=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Coverage_V2.json').read_text())
def test_training_fork_rejects_unrelated_contract_changes():
 c=copy.deepcopy(BASE);c['training_future_phase_C'].update(updates_initial=50,updates_max=50,validation_updates=[],save_every=5)
 validate_training_config(c,BASE,'baseline')
 c['path_reward']['scale']=.2
 with pytest.raises(ValueError):validate_training_config(c,BASE,'baseline')

class Policy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.actor=torch.nn.Sequential(torch.nn.Linear(3,2));self.critic=torch.nn.Sequential(torch.nn.Linear(4,1));self.log_std=torch.nn.Parameter(torch.zeros(2))

def parent(tmp_path):
 p=Policy();o=torch.optim.Adam(p.parameters())
 sum(x.sum() for x in p.parameters()).backward();o.step()
 identity=dict(schema=SCHEMA,alpha=0,lower_actor_sha256='lower',config_sha256='cfg',sampler_sha256='sampler',best_sha256='best',seed=910101)
 saved=dict(schema=SCHEMA,identity=identity,update=200,terminal_contract='terminal_contract_v2',policy=p.state_dict(),optimizer=o.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=None,numpy_rng=np.random.get_state(),python_rng=None)
 cp=tmp_path/'parent.pt';torch.save(saved,cp);ap=tmp_path/'actor.pkl'
 with ap.open('wb') as f:pickle.dump(dict(params=dict(Dense_0=dict(kernel=p.actor[0].weight.detach().numpy().T,bias=p.actor[0].bias.detach().numpy()))),f)
 record=dict(**identity,source_update=200,checkpoint=str(cp),checkpoint_sha256=digest(cp),actor=str(ap),actor_sha256=digest(ap))
 return p,record

def test_weights_only_preserves_fresh_optimizer_and_rng(tmp_path):
 src,r=parent(tmp_path);p=Policy();a=SimpleNamespace(policy=p,optimizer=torch.optim.Adam(p.parameters()),device='cpu');rng=torch.get_rng_state().clone()
 restore_weights_only(a,r)
 assert len(a.optimizer.state)==0 and torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(p.state_dict()[k],v) for k,v in src.state_dict().items())
 r['actor_sha256']='wrong'
 with pytest.raises(ValueError,match='SHA'):restore_weights_only(a,r)

def test_initialized_optimizer_is_not_silently_reused(tmp_path):
 _,r=parent(tmp_path);p=Policy();o=torch.optim.Adam(p.parameters());sum(x.sum() for x in p.parameters()).backward();o.step()
 with pytest.raises(ValueError,match='fresh optimizer'):restore_weights_only(SimpleNamespace(policy=p,optimizer=o,device='cpu'),r)
