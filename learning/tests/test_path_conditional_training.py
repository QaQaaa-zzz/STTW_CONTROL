import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import jax.numpy as j
import torch
from tensordict import TensorDict
from sttw_control import path_command_training as training
from sttw_control.path_command_conditioning import sample_episode_alpha,conditional_identity
from sttw_control.preference_best import atomic_json,digest

ROOT=Path(__file__).parents[2]

from flax import struct
@struct.dataclass
class RunnerState:
    tick:object
    alpha:object
    fault:object
@struct.dataclass
class RunnerRoute:
    path:object
    speed:object
    family:object
    sample_group:object


@pytest.mark.parametrize('prefix',['path_unified_alpha','path_strong_preference','path_unified_scratch'])
def test_preparation_counts_one_learner_and_engineering_has_no_dev(tmp_path,prefix):
    formal=training.prepare(ROOT/'learning/configs'/f'{prefix}_run.json',tmp_path/'formal')
    smoke=training.prepare(ROOT/'learning/configs'/f'{prefix}_engineering.json',tmp_path/'smoke')
    assert formal['training_policy_transitions_total']==13_107_200
    assert formal['training_lower_ticks_total']==52_428_800 and formal['maximum_validation_lower_ticks']==600_000
    assert formal['learner_count']==1 and formal['validation_updates']==[0,50,100,150,200]
    assert smoke['training_policy_transitions_total']==128 and smoke['training_lower_ticks_total']==512
    assert smoke['maximum_validation_lower_ticks']==0 and smoke['discard_engineering_update']

def test_path_batch_samples_alpha_at_reset_with_route_rng_independent():
    plan,cfg,_=training.load_plan(ROOT/'learning/configs/path_unified_alpha_run.json')
    # A minimal reset-only environment avoids physical simulation while exercising
    # the production route sampler and per-episode PathBatch wiring.
    env=SimpleNamespace(physics=SimpleNamespace(helpers=SimpleNamespace(pose=lambda _:j.array([0.,0.,0.]))),
        reset=lambda snapshot,path,speed,alpha:alpha)
    batch=training.PathBatch(env,SimpleNamespace(data=None),cfg,plan['seed'],None)
    endpoint=training.PathBatch(env,SimpleNamespace(data=None),cfg,plan['seed'],0.)
    for eid in [0,17,102,103]:
        mixed=batch.reset(j.int32(eid),j.int32(7));fixed=endpoint.reset(j.int32(eid),j.int32(7))
        assert float(mixed.state)==float(sample_episode_alpha(911402,eid,7))
        for a,b in zip(jax.tree.leaves(mixed.route),jax.tree.leaves(fixed.route)):np.testing.assert_array_equal(a,b)

def test_shared_checkpoint_resume_keeps_identity_rng_and_mixed_alpha_ppo(tmp_path):
    plan,cfg,source=training.load_plan(ROOT/'learning/configs/path_unified_alpha_run.json')
    obs=TensorDict({'policy':torch.zeros(8,351),'critic':torch.zeros(8,352)},batch_size=[8])
    obs['policy'][:,336]=torch.tensor([0,.1,.25,.4,.5,.75,.9,1]);obs['critic'][:,:351]=obs['policy']
    algo=training.make_algorithm(obs,2,cfg,'cpu');identity=conditional_identity(plan,cfg,source)
    with torch.no_grad():
        for _ in range(2):
            algo.act(obs);algo.process_env_step(obs,-obs['policy'][:,336],torch.zeros(8,dtype=torch.bool),{})
        algo.compute_returns(obs)
    result=algo.update()
    assert result['accepted_epochs']>0 and np.isfinite(result['mean_kl']) and set(result['kl_by_alpha'])=={'all'}
    path=training.save_checkpoint(tmp_path/'shared/checkpoints',algo,1,identity,route_counters=np.arange(8))
    atomic_json(tmp_path/'shared/last_completed.json',dict(schema=training.SCHEMA,update=1,checkpoint=str(path),checkpoint_sha256=digest(path)))
    receipt,best=training.resume_source(tmp_path,None,identity)
    assert best is None
    other=training.make_algorithm(obs,2,cfg,'cpu');training.restore_checkpoint(receipt['checkpoint'],other,identity)
    for key,value in algo.policy.state_dict().items():torch.testing.assert_close(value,other.policy.state_dict()[key],rtol=0,atol=0)


@pytest.mark.parametrize('plan_file',['path_unified_alpha_engineering.json','path_strong_preference_engineering.json','path_unified_scratch_engineering.json'])
@pytest.mark.parametrize('engineering_fault',[False,True])
def test_engineering_runner_uses_one_learner_and_never_runs_dev(tmp_path,monkeypatch,plan_file,engineering_fault):
    """Exercise the real orchestration/PPO on a deterministic non-physical batch."""
    from flax import struct
    from sttw_control import path_command_statistics as statistics
    from sttw_control import path_command_evaluation as evaluation
    from sttw_control import direct_command_training
    State=RunnerState;Route=RunnerRoute
    class Batch:
        def __init__(self,env,snapshot,config,seed,alpha):
            assert alpha is None
            self.config=config
        def reset(self,eid,episode,absolute_update=1):
            state=State(j.int32(0),sample_episode_alpha(self.config['alpha_conditioning']['seed'],eid,episode,self.config,absolute_update),j.bool_(False))
            route=Route(j.float32(0),j.float32(2),j.int32(0),j.int32(0))
            return training.Episode(state,route,eid,episode,j.zeros(6),j.zeros(10))
        def observation(self,ep):
            obs=j.zeros(351).at[336].set(ep.state.alpha)
            return obs,j.concatenate([obs,j.ones(1)]),ep.state.fault
        def advance_one(self,ep,z,absolute_update=1,*,auto_reset=True):
            nxt=ep.replace(state=ep.state.replace(tick=ep.state.tick+4));obs=self.observation(nxt)
            flags={k:j.bool_(False) for k in ('engineering_fault','done','physical_failure','tracking_domain_failure')}
            flags['engineering_fault']=j.asarray(engineering_fault)
            stats=dict(active_ticks=j.int32(4),rollout=j.array([4.,0.,0.,0.,-.01,0.]),completed_totals=j.zeros(6),
                completed_component_costs=j.zeros(10),diagnostics={'count':j.float32(4)},component_caps={k:j.int32(0) for k in statistics.COST_KEYS})
            return nxt,obs,j.float32(-.01),flags,obs,stats
    env=SimpleNamespace(set_log_template=lambda *args:None,terminal_contract='terminal_contract_v2')
    monkeypatch.setattr(training,'PathBatch',Batch)
    monkeypatch.setattr(training,'load_environment',lambda *args:(env,None))
    original=training.make_algorithm;calls=[]
    def cpu_algorithm(obs,steps,config,device):
        calls.append((len(obs),steps));algo=original(obs,steps,config,'cpu')
        if engineering_fault:
            algo.compute_returns=lambda *a:pytest.fail('faulty rollout must not enter GAE')
            algo.update=lambda *a:pytest.fail('faulty rollout must not optimize')
        return algo
    monkeypatch.setattr(training,'make_algorithm',cpu_algorithm)
    monkeypatch.setattr(training,'_start_tensorboard',lambda root:{'url':'test-only'})
    monkeypatch.setattr(training,'_verify_reward_http',lambda *args:True)
    monkeypatch.setattr(direct_command_training,'notify',lambda *args:None)
    monkeypatch.setattr(statistics,'summarize_statistics',lambda s,c:{'active_lower_samples':float(s['count'])})
    monkeypatch.setattr(evaluation.BatchedPanelEvaluator,'evaluate',lambda *args,**kw:pytest.fail('engineering must not evaluate'))
    out=tmp_path/'engineering'
    if plan_file=='path_unified_scratch_engineering.json':
        from sttw_control import path_command_conditioning as c
        monkeypatch.setattr(c,'initialize_conditional_actor',lambda *a,**kw:pytest.fail('scratch importing parent'))
    if engineering_fault:
        with pytest.raises(RuntimeError,match='discard rollout'):
            training.train(ROOT/'learning/configs'/plan_file,out,execute=True)
        status=json.loads((out/'status.json').read_text())
        assert status['state']=='error' and status['policy_transitions_total']==128
        assert not (out/'shared/checkpoints/update_0001.pt').exists()
        assert (out/'shared/engineering_fault_states.pkl').exists()
        return
    training.train(ROOT/'learning/configs'/plan_file,out,execute=True)
    status=json.loads((out/'status.json').read_text());receipt=json.loads((out/'shared/engineering_receipt.json').read_text())
    assert calls==[(8,16)] and status['state']=='completed' and status['shared_updates']==1
    assert status['policy_transitions_total']==128 and status['training_control_ticks_total']==512
    assert status['validation_control_ticks_total']==0 and receipt['discarded_update'] and not receipt['eligible_as_parent']
    assert not (out/'shared/best_conditional_model.json').exists()


@pytest.mark.parametrize('update',[0,50])
@pytest.mark.parametrize('completeness',['complete','missing','partial'])
def test_resume_completes_only_saved_boundary_validation(tmp_path,monkeypatch,update,completeness):
    from sttw_control import path_command_conditioning as conditioning
    from sttw_control import path_command_evaluation as evaluation
    from sttw_control import path_command_selection as selection
    plan,cfg,source=training.load_plan(ROOT/'learning/configs/path_unified_alpha_run.json')
    identity=conditional_identity(plan,cfg,source);params={'a':np.zeros(2)}
    parent=tmp_path/'parent';out=tmp_path/'new/shared';out.mkdir(parents=True)
    boundary=parent/'shared/validation'/f'update_{update:04d}'
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'model');actor=tmp_path/'actor.pkl';actor.write_bytes(b'actor')
    calls=[];counts={'ticks':0};persisted=[]
    monkeypatch.setattr(selection,'cases',lambda cfg:[('one',0,0)])
    monkeypatch.setattr(selection,'summarize_conditional',lambda panels,cfg:{'score_tuple':[0]})
    monkeypatch.setattr(selection,'persist_conditional_best',lambda *args:persisted.append(args))
    if completeness!='missing':
        boundary.mkdir(parents=True)
        atomic_json(boundary/'panel_identity.json',dict(schema='path_conditional_panel_v1',mode=conditioning.MODE,
            completed=completeness=='complete',alphas={selection.alpha_key(a):a for a in plan['alphas']},
            config_key=training.config_key(cfg),actor_parameters_sha256=None))
        atomic_json(boundary/'source.json',dict(identity,source_update=update,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),actor=str(actor),actor_sha256=digest(actor)))
        for alpha in plan['alphas']:
            folder=boundary/selection.alpha_key(alpha);folder.mkdir()
            np.savez(folder/'one.npz',time=np.arange(4))
            np.savez(folder/'one_path.npz',x=np.arange(4))
    def evaluate(saved_update,saved_checkpoint):
        calls.append((saved_update,saved_checkpoint));counts['ticks']+=120000
    receipt=conditioning.complete_resume_validation(parent,out,update,checkpoint,identity,cfg,params,evaluate,lambda:counts['ticks'])
    assert calls==([] if completeness=='complete' else [(update,checkpoint)])
    assert len(persisted)==int(completeness=='complete')
    assert receipt['new_validation_lower_ticks']==(0 if completeness=='complete' else 120000)
    assert receipt['parent_persisted_boundary_lower_ticks']==(0 if completeness=='missing' else 20)
    assert receipt['repeated_saved_lower_ticks']==(20 if completeness=='partial' else 0)
    assert json.loads((out/'resume_validation.json').read_text())==receipt
