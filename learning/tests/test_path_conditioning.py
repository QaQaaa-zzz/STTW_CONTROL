import copy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as j
import torch
from tensordict import TensorDict
from sttw_control import path_command_training as training
from sttw_control.preference_best import digest

ROOT=Path(__file__).parents[2]
CFG=json.loads((ROOT/'learning/configs/STTW_Path_Safety_candidate.json').read_text())

def conditioning():
    assert importlib.util.find_spec('sttw_control.path_command_conditioning') is not None, 'conditional module missing'
    return importlib.import_module('sttw_control.path_command_conditioning')

def test_episode_alpha_distribution_is_deterministic_and_has_endpoints():
    c=conditioning()
    draw=jax.jit(jax.vmap(lambda i:c.sample_episode_alpha(911402,i,j.int32(3))))
    a=np.asarray(draw(j.arange(20000,dtype=j.int32)))
    np.testing.assert_array_equal(a,np.asarray(draw(j.arange(20000,dtype=j.int32))))
    assert a.dtype==np.float32 and np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    assert .23<np.mean(a==0)<.27 and .23<np.mean(a==1)<.27
    assert .47<np.mean((a>0)&(a<1))<.53
    assert float(c.sample_episode_alpha(911402,2,3))!=float(c.sample_episode_alpha(911403,2,3))

def test_actor_only_warmstart_preserves_fresh_state_and_zeroes_alpha_column(tmp_path):
    c=conditioning();torch.manual_seed(4)
    obs=TensorDict({'policy':torch.randn(8,351),'critic':torch.randn(8,352)},batch_size=[8])
    parent=training.make_algorithm(obs,16,CFG,'cpu')
    with torch.no_grad():
        for p in parent.policy.actor.parameters():p.add_(.02)
        parent.policy.log_std.fill_(-4)
    path=tmp_path/'parent.pt';torch.save({'schema':training.SCHEMA,'identity':{'alpha':0,'lower_actor_sha256':'lower-sha'},'policy':parent.policy.state_dict()},path)
    child=training.make_algorithm(obs,16,CFG,'cpu')
    before=copy.deepcopy(child.policy.state_dict());rng=torch.get_rng_state().clone()
    c.initialize_conditional_actor(child,path,digest(path),expected_lower_sha='lower-sha')
    assert torch.equal(rng,torch.get_rng_state()) and not child.optimizer.state_dict()['state']
    for key,value in child.policy.state_dict().items():
        if not key.startswith('actor.'):assert torch.equal(value,before[key])
    first=next(m for m in child.policy.actor.modules() if isinstance(m,torch.nn.Linear))
    assert torch.count_nonzero(first.weight[:,336])==0
    base=obs.clone();base['policy'][:,336]=0
    with torch.no_grad():
        expected=parent.policy.act_inference(base)
        for alpha in [0,.25,.5,.75,1]:
            variant=base.clone();variant['policy'][:,336]=alpha
            torch.testing.assert_close(child.policy.act_inference(variant),expected,rtol=0,atol=0)
    with pytest.raises(ValueError,match='SHA'):c.initialize_conditional_actor(child,path,'wrong')
    with pytest.raises(ValueError,match='lower identity'):c.initialize_conditional_actor(child,path,digest(path),expected_lower_sha='other-lower')

def test_shared_identity_binds_entire_prepared_source():
    c=conditioning();p,cfg,source=training.load_plan(ROOT/'learning/configs/path_unified_alpha_run.json')
    original=c.conditional_identity(p,cfg,source)
    altered=copy.deepcopy(source);altered['prepared_bank_revision']='different'
    assert c.conditional_identity(p,cfg,altered)['source_sha256']!=original['source_sha256']

def test_conditional_plan_budget_and_legacy_mode_are_distinct():
    conditioning()
    p,c,_=training.load_plan(ROOT/'learning/configs/path_unified_alpha_run.json')
    assert p['policy_mode']=='shared_conditional_v1' and p['updates']==200
    assert c['training_future_phase_C']['validation_updates']==[0,50,100,150,200]
    assert c['path_reward']==CFG['path_reward']
    p2,c2,_=training.load_plan(ROOT/'learning/configs/path_unified_alpha_engineering.json')
    assert p2['engineering_check'] is True and p2['updates']==1
    assert c2==c
    old,_,_=training.load_plan(ROOT/'learning/configs/path_safety_candidate_run.json')
    assert 'policy_mode' not in old and old['alphas']==[0,1]

def test_conditional_seed_is_configurable_but_independent_and_network_fixed():
    module=conditioning();p,c,_=training.load_plan(ROOT/'learning/configs/path_unified_alpha_run.json')
    p['seed']=1729;c['alpha_conditioning']['seed']=1730
    module.validate_conditional_plan(p,c)
    c['alpha_conditioning']['seed']=c['training_sampler']['route_seed']
    with pytest.raises(ValueError,match='independent'):module.validate_conditional_plan(p,c)
    c['alpha_conditioning']['seed']=1730;c['network']['actor_dim']=345
    with pytest.raises(ValueError,match='network'):module.validate_conditional_plan(p,c)
