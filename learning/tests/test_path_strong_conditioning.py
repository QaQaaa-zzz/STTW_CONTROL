"""Strong preference contracts without physical simulation or GPU updates."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from sttw_control import path_command_conditioning as c,path_command_training as t
from sttw_control.preference_best import digest

ROOT=Path(__file__).parents[2]
def strong():return t.load_plan(ROOT/'learning/configs/path_strong_preference_run.json')

def test_strong_plan_has_bounded_budget_and_distinct_identity():
    p,cfg,source=strong();c.validate_conditional_plan(p,cfg)
    assert p['updates']==200 and p['seed']==911401 and cfg['alpha_conditioning']['seed']==911402
    assert p['initialization']['parent_alpha']==1 and cfg['path_reward']['primary_speed_multiplier']==2
    ident=c.conditional_identity(p,cfg,source)
    assert ident['architecture']=='alpha_modulated_v1' and ident['reward_version']=='path_safety_preference_v2'
    bad=copy.deepcopy(cfg);bad['alpha_conditioning']['curriculum'][1]['uniform_probability']=.5
    with pytest.raises(ValueError,match='alpha contract'):c.validate_conditional_plan(p,bad)

def test_jitted_curriculum_dynamic_input_changes_new_resets_only():
    p,cfg,_=strong()
    env=SimpleNamespace(physics=SimpleNamespace(helpers=SimpleNamespace(pose=lambda _:j.zeros(3))),reset=lambda snapshot,path,speed,alpha:alpha)
    batch=t.PathBatch(env,SimpleNamespace(data=None),cfg,p['seed'],None)
    reset=jax.jit(jax.vmap(batch.reset,in_axes=(0,0,None)))
    ids=j.arange(10000,dtype=j.int32);eps=j.zeros_like(ids)
    early=reset(ids,eps,j.int32(100));late=reset(ids,eps,j.int32(101))
    a=np.asarray(early.state);b=np.asarray(late.state)
    assert np.mean((a>0)&(a<1))==0 and .47<np.mean(a==0)<.53
    assert .18<np.mean((b>0)&(b<1))<.22
    # Continuing episode stays untouched; dynamic reset called solely for done.
    flags=dict(done=j.bool_(False),engineering_fault=j.bool_(False))
    ep=jax.tree.map(lambda x:x[0],early)
    kept=jax.jit(lambda state,update:t.reset_finished_episode(state,flags,lambda eid,e:batch.reset(eid,e,update)))(ep,j.int32(101))
    assert float(kept.state)==float(ep.state)
    np.testing.assert_array_equal(a,np.asarray(reset(ids,eps,j.int32(100)).state))
    for x,y in zip(jax.tree.leaves(early.route),jax.tree.leaves(late.route)):np.testing.assert_array_equal(x,y)

def test_folded_alpha1_preserves_actor_and_fresh_critic_adam_rng(tmp_path):
    p,cfg,_=strong();old=copy.deepcopy(cfg);old['network'].pop('architecture')
    obs=TensorDict({'policy':torch.randn(8,351),'critic':torch.randn(8,352)},batch_size=[8])
    parent=t.make_algorithm(obs,16,old,'cpu')
    with torch.no_grad():
        for param in parent.policy.actor.parameters():param.add_(.02)
    file=tmp_path/'parent.pt';torch.save(dict(schema=t.SCHEMA,identity=dict(alpha=1,lower_actor_sha256='frozen'),policy=parent.policy.state_dict()),file)
    child=t.make_algorithm(obs,16,cfg,'cpu');before=copy.deepcopy(child.policy.state_dict());rng=torch.get_rng_state().clone()
    c.initialize_conditional_actor(child,file,digest(file),expected_lower_sha='frozen',config=cfg)
    assert not child.optimizer.state_dict()['state'] and torch.equal(rng,torch.get_rng_state())
    for key,value in child.policy.state_dict().items():
        if not key.startswith('actor.'):torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    fixed=obs.clone();fixed['policy'][:,336]=1
    with torch.no_grad():
        expected=parent.policy.act_inference(fixed)
        for alpha in [0,.25,.5,.75,1]:
            variant=obs.clone();variant['policy'][:,336]=alpha
            torch.testing.assert_close(child.policy.act_inference(variant),expected,rtol=1e-5,atol=1e-6)
    registered={id(param) for group in child.optimizer.param_groups for param in group['params']}
    for net in (child.policy.actor,child.policy.critic):
        for name,param in net.named_parameters():
            if 'gain_' in name or 'shift_' in name:assert id(param) in registered and torch.count_nonzero(param)==0
    assert set(t.export_actor(child.policy)['params'])=={'Dense_0','Dense_1','Dense_2','Dense_3','gain_2','shift_2','gain_3','shift_3'}
    path=t.save_checkpoint(tmp_path/'saved',child,100,dict(architecture='alpha_modulated_v1'),route_counters=np.arange(8))
    other=t.make_algorithm(obs,16,cfg,'cpu');saved=t.restore_checkpoint(path,other,dict(architecture='alpha_modulated_v1'))
    assert saved['update']==100
    for key,value in child.policy.state_dict().items():torch.testing.assert_close(value,other.policy.state_dict()[key],rtol=0,atol=0)
