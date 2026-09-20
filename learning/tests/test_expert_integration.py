"""Real checked-Actor, RSL and short CPU-physics integration (no formal training)."""
from dataclasses import asdict,replace
from pathlib import Path
import json,hashlib,copy,subprocess
import numpy as np
import pytest
pytest.importorskip('flax');pytest.importorskip('mujoco');pytest.importorskip('torch');pytest.importorskip('rsl_rl')
import jax
import jax.numpy as jp
import torch
from tensordict import TensorDict
from sttw_control.env import RecoveryEnv,load_config
from sttw_control.network import (ResidualActor,save_policy,make_policy_identity,actor_transfer_data,
                                  load_expert_bundle,load_policy)
from sttw_control.training import TrainingConfig,normalization,plan_experts,train_experts
from sttw_control.rsl_training import make_algorithm,initialize_actor_weights,export_actor,guarded_update
from sttw_control.evaluation import search_candidates


def setup(tmp_path):
    c=load_config('learning/configs/soft_budget_ecbc1.json')
    c=replace(c,horizon_seconds=.025,preparation_seconds=0.,initial_roll_range=0.,
              timed_reference=replace(c.timed_reference,training_mix=False,fixed_scenario=None,fixed=((0.,2.3,0.),)))
    env=RecoveryEnv(c);mean,std=normalization(c)
    actor=ResidualActor((128,128,128),activation='elu')
    params=actor.init(jax.random.PRNGKey(31),jp.zeros(env.observation_size))
    params['params']['Dense_3']['kernel']=jp.zeros_like(params['params']['Dense_3']['kernel'])
    params['params']['Dense_3']['bias']=jp.array([.003,-.002])
    root=tmp_path/'source';ck=root/'checkpoints/update_0244'
    save_policy(ck,params,mean,std,make_policy_identity(env.bundle.identity,asdict(c),10),hidden_sizes=(128,128,128),activation='elu')
    (root/'declaration.json').write_text(json.dumps({'task':asdict(c)}))
    (root/'status.json').write_text('{"complete":true}')
    (root/'best_model.json').write_text(json.dumps({'checkpoint':str(ck)}))
    tc=TrainingConfig(trainer='rsl',initialize_actor=str(ck),num_envs=4,rollout_steps=4,updates=2,
          epochs=1,minibatch_size=16,activation='elu',hidden_sizes=(128,128,128),training_reward_selection=True,
          warmup_pool_size=0,warmup_steps=0,rsl_schedule='fixed',rsl_kl_limit=.02)
    return c,env,params,mean,std,ck,tc


def test_real_actor_transfer_leaves_value_and_optimizer_fresh(tmp_path):
    c,env,params,mean,std,ck,tc=setup(tmp_path)
    target=plan_experts(c,tc)[2][1]
    copied,info=actor_transfer_data(ck,target,env.bundle.identity,mean,std,tc.hidden_sizes,tc.activation)
    obs=TensorDict({'policy':torch.randn(4,env.observation_size)},batch_size=[4])
    algo=make_algorithm(obs,4,1,1,'cpu',activation='elu',hidden_sizes=tc.hidden_sizes,schedule='fixed')
    old_critic=copy.deepcopy(algo.policy.critic.state_dict());old_std=algo.policy.log_std.detach().clone()
    initialize_actor_weights(algo.policy,copied)
    assert not algo.optimizer.state
    for k,v in old_critic.items():torch.testing.assert_close(algo.policy.critic.state_dict()[k],v,rtol=0,atol=0)
    torch.testing.assert_close(algo.policy.log_std,old_std,rtol=0,atol=0)
    expected=ResidualActor(tc.hidden_sizes,activation='elu').apply(params,obs['policy'].numpy())
    np.testing.assert_allclose(torch.tanh(algo.policy.act_inference(obs)).detach().numpy(),expected,atol=2e-6)
    for _ in range(4):
        with torch.no_grad():
            z=algo.act(obs);algo.process_env_step(obs,-z.square().sum(-1),torch.zeros(4,dtype=torch.bool),{})
    with torch.no_grad():algo.compute_returns(obs)
    metrics,audit=guarded_update(algo,.02)
    assert np.isfinite(list(metrics.values())).all() and audit['final_exact_kl']<=.02
    with pytest.raises(ValueError):actor_transfer_data(ck,replace(target,failure_penalty=1.),env.bundle.identity,mean,std,tc.hidden_sizes,tc.activation)


def test_bundle_routes_and_short_cpu_search_reconstructs_rewards(tmp_path):
    c,env,params,mean,std,ck,tc=setup(tmp_path)
    root=tmp_path/'bundle';root.mkdir();members=[]
    for alpha,target,_ in plan_experts(c,tc):
        run=root/f'alpha_{alpha:g}'/'training';path=run/'checkpoints/update_0000'
        altered=jax.tree.map(lambda x:x.copy(),params)
        altered['params']['Dense_3']['bias']=jp.array([.003+alpha*.002,-.002])
        save_policy(path,altered,mean,std,make_policy_identity(env.bundle.identity,asdict(target),10),hidden_sizes=tc.hidden_sizes,activation='elu')
        (run/'declaration.json').write_text(json.dumps({'task':asdict(target)}));(run/'status.json').write_text('{"complete":true}')
        members.append({'alpha':alpha,'training':str(run),'sampled_best':str(path),'last_checkpoint':str(path)})
    (root/'experts.json').write_text(json.dumps({'schema':'sttw_expert_bundle_v1','complete':True,'task':asdict(c),'members':members}))
    (root/'status.json').write_text('{"complete":true}')
    _,policies,ids=load_expert_bundle(root,env.bundle.identity)
    s=env.reset(9);probe=s.obs
    assert float(policies[0.](probe)[0])<float(policies[.5](probe)[0])<float(policies[1.](probe)[0])
    # Identical source and Actor-only copy are indistinguishable in actual short physics.
    source=load_policy(ck,expected=make_policy_identity(env.bundle.identity,asdict(c),10))
    a=b=s
    for _ in range(3):
        a=env.step(a,source(a.obs));b=env.step(b,policies[0.](b.obs))
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    rows=search_candidates(env,tmp_path/'search',seed=9,policy=policies[0.],policy_identity=ids[0.],alpha=0.,start_seconds=.005,duration=.01,offsets=(0.,))
    assert len(rows)==1 and rows[0]['steps']==4 and rows[0]['finite']
    assert rows[0]['reward_reconstruction_max_abs']<3e-5
    tr=np.load(tmp_path/'search/alpha_0_candidate_00.npz')
    assert tr['time'].shape==(4,) and set(tr['scores'].shape)=={3}
    # Corrupted explicit checkpoint is rejected, never substituted.
    p=Path(members[0]['sampled_best'])/'actor.msgpack';p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError):load_expert_bundle(root,env.bundle.identity)


def test_orchestration_is_three_processes_and_refuses_silent_partial_restart(tmp_path,monkeypatch):
    c,env,params,mean,std,ck,tc=setup(tmp_path)
    task_path=tmp_path/'task.json';task_path.write_text(json.dumps(asdict(c)))
    calls=[]
    # Process isolation contract tested with a stub, not claimed as real training.
    def child(command,check):
        assert check;calls.append(command)
        cfg=json.loads(Path(command[command.index('--task')+1]).read_text())
        train=json.loads(Path(command[command.index('--config')+1]).read_text())
        out=Path(command[command.index('--output')+1]);out.mkdir()
        path=out/'checkpoints/update_0002';path.mkdir(parents=True)
        (path/'actor.msgpack').write_bytes(b'explicit-test-stub');(path/'identity.json').write_text('{}')
        (out/'declaration.json').write_text(json.dumps({'task':cfg,'training':train}))
        (out/'status.json').write_text(json.dumps({'complete':True,'last_checkpoint':str(path)}))
        (out/'best_model.json').write_text(json.dumps({'checkpoint':str(path)}))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(subprocess,'run',child)
    out=tmp_path/'experts';result=train_experts(task_path,out,tc)
    assert len(calls)==3 and result['training_transition_budget']==96
    assert [m['alpha'] for m in result['members']]==[0.,.5,1.]
    train_experts(task_path,out,tc,skip_completed=True);assert len(calls)==3
    with pytest.raises(ValueError):train_experts(task_path,out,tc)
    (out/'alpha_1/training/status.json').write_text('{"complete":false}')
    with pytest.raises(ValueError):train_experts(task_path,out,tc,skip_completed=True)
    assert len(calls)==3


@pytest.mark.skipif(__import__('os').environ.get('STTW_TEST_MJX_CPU')!='1',reason='explicit tiny real MJX/RSL loop')
def test_real_cli_actor_initialized_training_saves_owned_checkpoints(tmp_path):
    import sys,os
    c,env,params,mean,std,ck,tc=setup(tmp_path)
    target=plan_experts(c,tc)[1][1]
    tc=replace(tc,plot_interval=0,phase_spread_initialization=False)
    task=tmp_path/'task.json';training=tmp_path/'training_config.json'
    task.write_text(json.dumps(asdict(target)));training.write_text(json.dumps(asdict(tc)))
    out=tmp_path/'tiny_run'
    command=[sys.executable,'learning/cli/train.py','--task',str(task),'--config',str(training),'--output',str(out)]
    run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300,env=dict(os.environ))
    (tmp_path/'tiny_run.log').write_text(run.stdout)
    assert run.returncode==0,run.stdout[-12000:]
    status=json.loads((out/'status.json').read_text());assert status['complete']
    declaration=json.loads((out/'declaration.json').read_text())
    assert declaration['actor_initialization']['actor_sha256']==hashlib.sha256((ck/'actor.msgpack').read_bytes()).hexdigest()
    identity=make_policy_identity(env.bundle.identity,asdict(target),10)
    original=load_policy(ck,expected=make_policy_identity(env.bundle.identity,asdict(c),10))
    initialized=load_policy(out/'checkpoints/update_0000',expected=identity)
    obs=env.set_priority(env.reset(8),.5).obs
    np.testing.assert_allclose(original(obs),initialized(obs),atol=2e-6)
    records=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    assert len(records)==2 and records[-1]['control_transitions']==32
    assert int(json.loads((out/'best_model.json').read_text())['update'])<2
