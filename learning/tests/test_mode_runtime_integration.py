"""Deploy-time gates on the original installed Flax/MuJoCo/RSL environment.

Not a GPU benchmark or a learned-policy outcome. This module intentionally skips
locally when simulator dependencies are absent; deploy.py requires it to run with
ZERO skips before committing/pushing from the user's authenticated workstation.
"""
from dataclasses import asdict, replace
from pathlib import Path
import copy
import json
import hashlib
import numpy as np
import pytest
pytest.importorskip('flax')
pytest.importorskip('mujoco')
pytest.importorskip('optax')
pytest.importorskip('rsl_rl')
pytest.importorskip('tensordict')
import jax
import jax.numpy as jp
import torch
from tensordict import TensorDict
from sttw_control.env import RecoveryEnv, load_config
from sttw_control.training import normalization
from sttw_control.network import (make_policy_identity, save_policy, load_policy,
    verified_actor_initialization, load_independent_mode_bundle)
from sttw_control.rsl_training import make_algorithm, export_actor, initialize_dense_actor, guarded_update


def cfg():
    return load_config('learning/configs/soft_budget_ecbc1.json')


def create_source(tmp_path):
    c=cfg();env=RecoveryEnv(c);mean,std=normalization(c)
    ident=make_policy_identity(env.bundle.identity,asdict(c),c.observation.history_steps)
    n=len(mean);obs=TensorDict({'policy':torch.zeros(4,n)},batch_size=[4])
    algo=make_algorithm(obs,4,1,1,'cpu',activation='elu',hidden_sizes=(128,128,128),schedule='fixed')
    # Nonzero Actor tests copying more than a trivial zero action.
    with torch.no_grad():
        layers=[m for m in algo.policy.actor.modules() if isinstance(m,torch.nn.Linear)]
        layers[-1].weight.fill_(.001);layers[-1].bias.copy_(torch.tensor([.02,-.03]))
    source=tmp_path/'source';cp=source/'checkpoints/update_0244';source.mkdir()
    save_policy(cp,export_actor(algo.policy),mean,std,ident,hidden_sizes=(128,128,128),activation='elu')
    (source/'status.json').write_text(json.dumps({'complete':True}))
    (source/'declaration.json').write_text(json.dumps({'task':asdict(c),'policy_identity':ident}))
    return c,env,algo,source,cp,mean,std


def test_actual_actor_transfer_prediction_identity_and_fresh_critic(tmp_path):
    c,env,source_algo,source,cp,mean,std=create_source(tmp_path)
    for alpha in (0.,.5,1.):
        target=replace(c,priority=replace(c.priority,randomize_alpha=False,fixed_alpha=alpha,
                         training_alphas=(alpha,),validation_alphas=(0.,.5,1.)))
        obs=TensorDict({'policy':torch.randn(4,len(mean))},batch_size=[4])
        algo=make_algorithm(obs,4,1,1,'cpu',activation='elu',hidden_sizes=(128,128,128),schedule='fixed')
        critic=copy.deepcopy(algo.policy.critic.state_dict());std0=algo.policy.log_std.detach().clone()
        params,record=verified_actor_initialization(cp,source,asdict(target),env.bundle.identity,mean,std,(128,128,128),'elu')
        initialize_dense_actor(algo.policy,params)
        torch.testing.assert_close(algo.policy.act_inference(obs),source_algo.policy.act_inference(obs),rtol=0,atol=0)
        for k,v in critic.items():torch.testing.assert_close(algo.policy.critic.state_dict()[k],v,rtol=0,atol=0)
        torch.testing.assert_close(algo.policy.log_std,std0,rtol=0,atol=0);assert not algo.optimizer.state
        assert record['fixed_alpha']==alpha
        original=load_policy(cp,expected=record_identity(source))
        raw=np.asarray(obs['policy'])*std+mean
        np.testing.assert_allclose(np.asarray(jax.vmap(original)(jp.asarray(raw))),
            np.asarray(torch.tanh(algo.policy.act_inference(obs)).detach()),atol=2e-6,rtol=2e-5)
        # Actual RSL update still consumes only its own on-policy samples.
        with torch.no_grad():
            for _ in range(4):
                z=algo.act(obs);algo.process_env_step(obs,-z.square().sum(-1),torch.zeros(4,dtype=torch.bool),{})
            algo.compute_returns(obs)
        _,audit=guarded_update(algo,.02)
        assert audit['final_exact_kl']<=.02


def record_identity(source):
    return json.loads((source/'declaration.json').read_text())['policy_identity']


def test_complete_reset_reuse_keeps_physics_and_zero_residual_baseline(tmp_path):
    from sttw_control.evaluation import evaluate
    c=cfg();env=RecoveryEnv(c)
    prepared=env.reset(12)  # genuine 3.5s closed-loop preparation, not synthetic qpos clone
    before_q=np.asarray(prepared.data.qpos).copy();before_v=np.asarray(prepared.data.qvel).copy()
    branches=[env.set_priority(prepared,a) for a in (0.,.5,1.)]
    for _ in range(3):
        branches=[env.step(s,np.zeros(2)) for s in branches]
        for s in branches:
            np.testing.assert_array_equal(s.data.qpos,branches[0].data.qpos)
            np.testing.assert_array_equal(s.actuator.previous,branches[0].actuator.previous)
    np.testing.assert_array_equal(prepared.data.qpos,before_q);np.testing.assert_array_equal(prepared.data.qvel,before_v)
    # Bound the engineering evaluator loop only; no formal experiment uses this override.
    env.horizon=3
    result=evaluate(env,tmp_path/'reuse',seed=12,priority_alpha=0.,initial_state=prepared)
    assert result['transitions']==3
    decl=json.loads((tmp_path/'reuse/declaration.json').read_text())
    assert decl['reset_reuse'].startswith('complete live tick-zero')
    np.testing.assert_array_equal(prepared.data.qpos,before_q)
    with pytest.raises(ValueError):evaluate(env,tmp_path/'bad',initial_state=branches[0])


def test_three_mode_bundle_requires_same_source_and_whole_checkpoints(tmp_path):
    c,env,source_algo,source,cp,mean,std=create_source(tmp_path)
    source_hash=hashlib.sha256((cp/'actor.msgpack').read_bytes()).hexdigest()
    bundle={'schema':'sttw_independent_modes_v1','complete':True,'source_actor_sha256':source_hash,'modes':{}}
    for alpha in (0.,.5,1.):
        mode=replace(c,priority=replace(c.priority,randomize_alpha=False,fixed_alpha=alpha,
                     training_alphas=(alpha,),validation_alphas=(0.,.5,1.)))
        train=tmp_path/f'mode_{alpha}';out=train/'checkpoints/update_0001';train.mkdir()
        identity=make_policy_identity(env.bundle.identity,asdict(mode),mode.observation.history_steps)
        save_policy(out,export_actor(source_algo.policy),mean,std,identity,hidden_sizes=(128,128,128),activation='elu')
        d={'task':asdict(mode),'policy_identity':identity};(train/'declaration.json').write_text(json.dumps(d))
        (train/'status.json').write_text(json.dumps({'complete':True}))
        (train/'actor_initialization.json').write_text(json.dumps({'actor_payload_sha256':source_hash}))
        bundle['modes'][str(alpha)]={'training':str(train),'checkpoint':str(out),
            'actor_payload_sha256':hashlib.sha256((out/'actor.msgpack').read_bytes()).hexdigest(),
            'declaration_sha256':hashlib.sha256((train/'declaration.json').read_bytes()).hexdigest()}
    path=tmp_path/'modes.json';path.write_text(json.dumps(bundle))
    policies,meta=load_independent_mode_bundle(path,asdict(c),env.bundle.identity)
    assert set(policies)=={0.,.5,1.} and len(meta)==3
    bundle['modes']['1.0']['actor_payload_sha256']='0'*64;path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):load_independent_mode_bundle(path,asdict(c),env.bundle.identity)
