"""Scratch initialization, fixed alpha stream and self-checkpoint restoration."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import jax
import jax.numpy as j
from tensordict import TensorDict
from sttw_control import path_command_training as t, path_command_conditioning as c
ROOT=Path(__file__).parents[2]
def scratch():return t.load_plan(ROOT/'learning/configs/path_unified_scratch_run.json')
def test_scratch_preparation_without_upper_parent(tmp_path):
 p,cfg,source=scratch()
 assert all(p['initialization'][k] is None for k in ('parent_alpha','checkpoint','checkpoint_sha256'))
 for key,value in [('parent_alpha',1),('checkpoint','missing.pt'),('checkpoint_sha256','a'*64)]:
  bad=copy.deepcopy(p);bad['initialization'][key]=value
  with pytest.raises(ValueError):c.validate_conditional_plan(bad,cfg)
 report=t.prepare(ROOT/'learning/configs/path_unified_scratch_run.json',tmp_path/'declaration')
 assert report['training_policy_transitions_total']==13_107_200 and report['maximum_validation_lower_ticks']==600_000
 ident=c.conditional_identity(p,cfg,source)
 assert ident['initialization_mode']=='scratch' and ident['parent_checkpoint_sha256'] is None
 assert 'curriculum_key' not in ident and ident['acceptance_key']
 bad=copy.deepcopy(cfg);bad['alpha_conditioning']['sampling_mode']='unknown'
 with pytest.raises(ValueError):c.validate_conditional_plan(p,bad)

def test_fresh_seed_zero_output_random_hidden_and_self_resume(tmp_path,monkeypatch):
 p,cfg,source=scratch();obs=TensorDict({'policy':torch.randn(8,351),'critic':torch.randn(8,352)},batch_size=[8])
 def make():
  torch.manual_seed(p['seed']);return t.make_algorithm(obs,2,cfg,'cpu')
 a=make();b=make();before=copy.deepcopy(a.policy.state_dict())
 monkeypatch.setattr(c,'initialize_conditional_actor',lambda *a,**kw:pytest.fail('scratch must never import parent'))
 receipt=c.initialize_upper_policy(a,p,cfg,source)
 assert receipt['mode']=='scratch' and receipt['restored']==[] and receipt['parent_checkpoint'] is None
 assert not a.optimizer.state_dict()['state']
 for k,v in before.items():torch.testing.assert_close(v,b.policy.state_dict()[k],rtol=0,atol=0)
 first=next(m for m in a.policy.actor.modules() if isinstance(m,torch.nn.Linear))
 assert torch.count_nonzero(first.weight[:,336]) and torch.count_nonzero(first.weight)
 assert torch.count_nonzero(a.policy.act_inference(obs))==0
 np.testing.assert_allclose(a.policy.log_std.detach().exp().numpy(),[.25,.10])
 with torch.no_grad():
  for _ in range(2):
   a.act(obs);a.process_env_step(obs,-torch.arange(8,dtype=torch.float32)/10,torch.zeros(8,dtype=torch.bool),{})
  a.compute_returns(obs)
 a.update()
 # Nonzero FiLM must survive self resume, regardless of a first zero-output gradient.
 with torch.no_grad():a.policy.actor.gain_2.add_(.02)
 identity=c.conditional_identity(p,cfg,source)
 file=t.save_checkpoint(tmp_path,a,1,identity,route_counters=np.arange(8),rejections=2)
 saved=t.restore_checkpoint(file,b,identity)
 assert saved['update']==1 and saved['consecutive_rejections']==2 and a.optimizer.state_dict()['state']
 for k,v in a.policy.state_dict().items():torch.testing.assert_close(v,b.policy.state_dict()[k],rtol=0,atol=0)
 assert saved['route_counters'].tolist()==list(range(8))

def test_fixed_mixture_update_invariance_and_independent_routes():
 from types import SimpleNamespace
 p,cfg,_=scratch();ids=j.arange(10000,dtype=j.int32);eps=j.zeros_like(ids)
 sample=jax.jit(jax.vmap(lambda eid,ep,u:c.sample_episode_alpha(911402,eid,ep,cfg,u),in_axes=(0,0,None)))
 arrays=[np.asarray(sample(ids,eps,j.int32(u))) for u in (1,100,101,200)]
 for a in arrays[1:]:np.testing.assert_array_equal(a,arrays[0])
 a=arrays[0];assert .38<np.mean(a==0)<.42 and .38<np.mean(a==1)<.42 and .18<np.mean((a>0)&(a<1))<.22
 env=SimpleNamespace(physics=SimpleNamespace(helpers=SimpleNamespace(pose=lambda _:j.zeros(3))),reset=lambda s,p,v,a:a)
 mixed=t.PathBatch(env,SimpleNamespace(data=None),cfg,p['seed'],None);fixed=t.PathBatch(env,SimpleNamespace(data=None),cfg,p['seed'],0.)
 for eid in (0,102,306):
  x=mixed.reset(j.int32(eid),j.int32(7));y=fixed.reset(j.int32(eid),j.int32(7))
  for a,b in zip(jax.tree.leaves(x.route),jax.tree.leaves(y.route)):np.testing.assert_array_equal(a,b)
  flags=dict(done=j.bool_(False),engineering_fault=j.bool_(False))
  kept=t.reset_finished_episode(x,flags,lambda eid,ep:mixed.reset(eid,ep,200))
  assert float(kept.state)==float(x.state)


def test_dev_reserve_counts_parent_partial_consumption_before_physics():
 p,cfg,_=scratch()
 assert t.check_conditional_dev_budget(dict(validation_control_ticks_total=120000,parent_consumed_validation_control_ticks=360000),cfg,p['alphas'])==120000
 with pytest.raises(RuntimeError,match='lineage budget'):
  t.check_conditional_dev_budget(dict(validation_control_ticks_total=120000,parent_consumed_validation_control_ticks=360001),cfg,p['alphas'])


def test_reboot_charges_lost_updates_and_uncertain_inflight_batch():
 parent=dict(completed_updates=5,stage='sampling',policy_transitions_total=5*65536)
 budget=t.scratch_resume_budget(parent,dict(update=0),512,128,200)
 assert budget['restored_checkpoint_update']==0 and budget['sampling_budget_cursor']==6
 assert budget['known_policy_transitions']==327680 and budget['uncertain_policy_transition_reserve']==65536
 assert budget['charged_policy_transitions']==393216 and budget['remaining_sampling_updates']==194
 assert budget['unrecoverable_sampling_slots']==6
 # Clean saved boundary recovers real learner progress without charging a phantom batch.
 parent=dict(completed_updates=25,stage='optimized',policy_transitions_total=25*65536)
 clean=t.scratch_resume_budget(parent,dict(update=25),512,128,200)
 assert clean['sampling_budget_cursor']==25 and clean['uncertain_policy_transition_reserve']==0
 assert clean['remaining_sampling_updates']==175 and clean['unrecoverable_sampling_slots']==0
 # Repeated reboot preserves prior charge and loss, without double accounting.
 parent=dict(completed_updates=7,stage='optimized',parent_consumed_policy_transitions=6*65536,policy_transitions_total=65536)
 again=t.scratch_resume_budget(parent,dict(update=7,resume_accounting=budget),512,128,200)
 assert again['sampling_budget_cursor']==7 and again['charged_policy_transitions']==7*65536
 assert again['unrecoverable_sampling_slots']==6
