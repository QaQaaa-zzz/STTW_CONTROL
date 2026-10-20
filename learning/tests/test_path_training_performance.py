"""Training payload equivalence, fault gates and terminal GAE contracts."""
import copy
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
import pytest
from sttw_control import path_command_env as environment
from sttw_control import path_command_training as training
from test_path_conditional_statistics import conditional
from test_path_terminal_contract import minimal_env,State,Physical,CFG


def test_training_log_preserves_all_statistics_and_drops_debug_arrays():
    from sttw_control.path_command_statistics import step_statistics
    logs,route,cfg=conditional(.5)
    logs.update(lower_observation=j.zeros((4,210)),preview_body=j.zeros((4,2)),
                actuator_force_min=j.zeros((4,4)))
    compact=environment.training_log(logs)
    assert 'lower_observation' not in compact
    assert 'preview_body' not in compact
    assert 'actuator_force_min' not in compact
    expected=step_statistics(logs,route,cfg);actual=step_statistics(compact,route,cfg)
    assert expected.keys()==actual.keys()
    for key in expected:np.testing.assert_array_equal(expected[key],actual[key])


@pytest.mark.parametrize('failure_tick',[0,1,2,3,4])
@pytest.mark.parametrize('latent',[[0.,0.],[float('nan'),0.]])
def test_compact_scan_matches_full_state_rewards_and_failure_tail(failure_tick,latent):
    env=minimal_env();old_tick=env.tick
    env.zero_log['lower_observation']=j.zeros(210)
    def tick(s,z,path,speed,**kwargs):
        end,log=old_tick(s,z,path,speed)
        return end.replace(physical=Physical((end.tick==failure_tick)&(failure_tick>0))),log
    env.tick=tick
    start=State(Physical(j.bool_(False)),j.bool_(False),j.bool_(False),j.zeros(2),j.int32(0),j.bool_(False))
    full,logs=jax.jit(lambda s:env.policy_step(s,j.array(latent),None,2.,20.))(start)
    compact,small=jax.jit(lambda s:env.policy_step(s,j.array(latent),None,2.,20.,training=True))(start)
    for a,b in zip(jax.tree.leaves(full),jax.tree.leaves(compact)):np.testing.assert_array_equal(a,b)
    assert 'lower_observation' not in small
    for key in small:
        for a,b in zip(jax.tree.leaves(small[key]),jax.tree.leaves(logs[key])):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('diagnostics',[None,False,True])
def test_lower_force_diagnostics_default_off_and_opt_in(monkeypatch,diagnostics):
    seen={}
    class Constructed(Exception):pass
    def physics(*,config,backend):
        seen.update(config);raise Constructed
    monkeypatch.setattr(environment,'TeleopEnv',physics)
    cfg=copy.deepcopy(CFG)
    if diagnostics is not None:cfg['lower_internal_diagnostics']=diagnostics
    with pytest.raises(Constructed):environment.PathCommandEnv(cfg,{},None)
    assert seen['lower_internal_diagnostics'] is (diagnostics is True)


def test_host_payload_reduces_statistics_without_losing_faults_or_counts():
    stats=dict(rollout=j.arange(18,dtype=j.float32).reshape(3,6),
        completed_totals=j.arange(18,dtype=j.float32).reshape(3,6),
        completed_component_costs=j.ones((3,10)),active_ticks=j.array([4,1,0]),
        component_caps={'speed':j.array([0,2,1])},diagnostics={'count':j.array(5.)})
    flags=dict(done=j.array([0,1,1],bool),physical_failure=j.array([0,1,0],bool),
        tracking_domain_failure=j.array([0,0,1],bool),engineering_fault=j.array([0,0,0],bool))
    payload=jax.jit(training.training_step_payload)(j.array([1.,2.,3.]),flags,j.array([0,0,1],bool),stats,j.array([0.,.5,1.]))
    assert int(payload['engineering_fault_count'])==1
    assert int(payload['active_ticks'])==5
    assert int(payload['episode_ends'])==2
    assert int(payload['physical_failures'])==int(payload['domain_failures'])==1
    np.testing.assert_array_equal(payload['alpha_counts'],[1,1,1])
    np.testing.assert_array_equal(payload['rollout'],np.asarray(stats['rollout']).sum(0))
    assert int(payload['component_caps']['speed'])==3
    np.testing.assert_array_equal(payload['completed_component_costs'],[3.]*10)


def test_actual_rsl_gae_cuts_terminal_value_without_timeout_bootstrap():
    import torch
    from tensordict import TensorDict
    obs=TensorDict({'policy':torch.zeros(2,351),'critic':torch.zeros(2,352)},batch_size=[2])
    cfg=copy.deepcopy(CFG);cfg['training_future_phase_C']['minibatches']=2
    algo=training.make_algorithm(obs,1,cfg,'cpu')
    with torch.no_grad():
        # Deliberately large negative successor V would reveal a bootstrap bug.
        for p in algo.policy.critic.parameters():p.zero_()
        list(algo.policy.critic.modules())[-1].bias.fill_(-123.)
        algo.act(obs)
        flags=training.termination_flags(j.array([4,4]),j.array([True,False]),False,j.array([False,True]),20.,CFG)
        reward=torch.tensor([-71.,-91.])
        algo.process_env_step(obs,reward,torch.tensor(np.array(flags['done'])),{})
        algo.compute_returns(obs)
    np.testing.assert_allclose(algo.storage.returns[:, :, 0].numpy(),[[-71.,-91.]],atol=1e-5)


@pytest.mark.skipif(__import__('os').environ.get('STTW_PATH_GPU_CHECK')!='1',reason='explicit bounded GPU/asset check only')
def test_gpu_full_diagnostics_and_compact_training_twin_state(tmp_path):
    """24 lower ticks total, zero PPO updates, no historical run writes."""
    import json,time,os
    from pathlib import Path
    from sttw_control.path_command_statistics import step_statistics
    root=Path(__file__).parents[2]
    plan,cfg,source=training.load_plan(root/'learning/configs/path_conflict_exposure_run.json')
    if jax.default_backend()!='gpu':pytest.fail('GPU check requires CUDA')
    envs=[];states=[];routes=[];timings=[]
    for diagnostic,compact in [(True,False),(False,False),(False,True)]:
        variant=copy.deepcopy(cfg);variant['lower_internal_diagnostics']=diagnostic
        env,snapshot=training.load_environment(variant,source,'terminal_contract_v2')
        batch=training.PathBatch(env,snapshot,variant,plan['seed'],None)
        episode=batch.reset(j.int32(102),j.int32(0),j.int32(81))
        env.set_log_template(episode.state,episode.route.path,episode.route.speed)
        envs.append((env,compact));states.append(episode.state);routes.append(episode.route)
    executors=[]
    for (env,compact),state,route in zip(envs,states,routes):
        started=time.perf_counter()
        fn=jax.jit(lambda st,z,v,env=env,compact=compact,route=route:env.policy_step(st,z,route.path,v,20.,training=compact))
        compiled=fn.lower(state,j.zeros(2),route.speed).compile()
        executors.append(compiled);timings.append(dict(compile_seconds=time.perf_counter()-started))
    max_state_error=0.;log_leaf_counts=[]
    for step in range(2):
        outputs=[]
        for i,execute in enumerate(executors):
            started=time.perf_counter()
            states[i],logs=execute(states[i],j.array([.1,-.2]),routes[i].speed+.1*step)
            jax.block_until_ready(states[i]);timings[i][f'step{step}_seconds']=time.perf_counter()-started
            outputs.append(jax.device_get((states[i],logs)))
        baseline,full=outputs[0]
        for state,logs in outputs[1:]:
            for expected,actual in zip(jax.tree.leaves(baseline),jax.tree.leaves(state)):
                if np.asarray(expected).dtype.kind in 'fc':
                    error=float(np.max(np.abs(expected-actual))) if np.size(expected) else 0.
                    max_state_error=max(max_state_error,error)
                np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
            for key in logs:
                for expected,actual in zip(jax.tree.leaves(full[key]),jax.tree.leaves(logs[key])):
                    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
            expected_stats=step_statistics(full,routes[0],cfg);actual_stats=step_statistics(logs,routes[0],cfg)
            for key in expected_stats:np.testing.assert_allclose(actual_stats[key],expected_stats[key],rtol=1e-6,atol=1e-6)
        log_leaf_counts=[len(jax.tree.leaves(out[1])) for out in outputs]
    receipt=dict(lower_ticks=24,ppo_updates=0,max_state_absolute_error=max_state_error,log_leaf_counts=log_leaf_counts,timings=timings,
        variants=['force_diagnostics_full','default_full','default_training'],benchmark_claim=False)
    destination=Path(os.environ.get('STTW_PATH_GPU_RECEIPT',str(tmp_path/'gpu_receipt.json')))
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


def test_batch_reset_guard_skips_all_route_work_on_nonterminal_steps():
    from flax import struct
    @struct.dataclass
    class Episode:
        env_id:object
        episode:object
        value:object
    called=[]
    def reset(eid,episode):
        jax.debug.callback(lambda eid:called.append(int(eid)),eid)
        return Episode(eid,episode,j.float32(99.))
    ep=Episode(j.arange(3),j.array([1,4,2]),j.array([3.,5.,7.]))
    run=jax.jit(lambda done,fault:training.reset_finished_batch(ep,dict(done=done,engineering_fault=fault),reset))
    kept=run(j.zeros(3,bool),j.zeros(3,bool));jax.block_until_ready(kept)
    assert called==[]
    np.testing.assert_array_equal(kept.value,ep.value)
    actual=run(j.array([False,True,True]),j.array([False,False,True]));jax.block_until_ready(actual)
    np.testing.assert_array_equal(actual.value,[3.,99.,7.])
    np.testing.assert_array_equal(actual.episode,[1,5,2])


@pytest.mark.parametrize('count',[0,1,15,17,40])
def test_sparse_reset_processes_every_terminal_without_full_batch_work(count):
    from flax import struct
    @struct.dataclass
    class E:
        env_id:object
        episode:object
        value:object
    n=40;called=[]
    def reset(eid,episode):
        jax.debug.callback(lambda ids:called.append(np.asarray(ids).copy()),eid)
        return E(eid,episode,eid.astype(j.float32)+100.)
    ep=E(j.arange(n),j.arange(n),j.arange(n,dtype=j.float32))
    done=j.arange(n)<count
    actual=jax.jit(lambda:training.reset_finished_batch(ep,dict(done=done,engineering_fault=j.zeros(n,bool)),reset))()
    jax.block_until_ready(actual)
    np.testing.assert_array_equal(actual.episode,np.arange(n)+(np.arange(n)<count))
    np.testing.assert_array_equal(actual.value,np.arange(n)+100.*(np.arange(n)<count))
    assert sum(np.size(x) for x in called)<=((count+15)//16)*16


def test_evaluation_scan_removes_only_unused_lower_observation_and_preserves_state():
    env=minimal_env();env.zero_log['lower_observation']=j.zeros(210)
    base=State(Physical(j.bool_(False)),j.bool_(False),j.bool_(False),j.zeros(2),j.int32(0),j.bool_(False))
    full,logs=env.policy_step(base,j.zeros(2),None,2.,20.)
    compact,small=env.policy_step(base,j.zeros(2),None,2.,20.,evaluation=True)
    for a,b in zip(jax.tree.leaves(full),jax.tree.leaves(compact)):np.testing.assert_array_equal(a,b)
    assert set(small)==set(logs)-{'lower_observation'}
    for key in small:
        for a,b in zip(jax.tree.leaves(small[key]),jax.tree.leaves(logs[key])):np.testing.assert_array_equal(a,b)


def test_finite_task_end_cuts_value_but_rollout_cut_bootstraps():
    import torch
    from tensordict import TensorDict
    obs=TensorDict({'policy':torch.zeros(2,351),'critic':torch.zeros(2,352)},batch_size=[2])
    cfg=copy.deepcopy(CFG);cfg['training_future_phase_C']['minibatches']=1
    algo=training.make_algorithm(obs,1,cfg,'cpu')
    with torch.no_grad():
        for p in algo.policy.critic.parameters():p.zero_()
        list(algo.policy.critic.modules())[-1].bias.fill_(7.)
        algo.act(obs)
        flags=training.termination_flags(j.array([4000,100]),False,False,False,20.,cfg,'terminal_contract_v2')
        algo.process_env_step(obs,torch.tensor([-1.,-1.]),torch.tensor(np.array(flags['done'])),{})
        algo.compute_returns(obs)
    np.testing.assert_allclose(algo.storage.returns[0,:,0].numpy(),[-1.,-1.+cfg['training_future_phase_C']['gamma']*7.],atol=1e-5)


def test_sparse_reset_runtime_flags_match_reference_on_real_route_sampler():
    from pathlib import Path
    _,cfg,_=training.load_plan(Path(__file__).parents[1]/'configs/path_coverage_shared_run.json')
    env=SimpleNamespace(physics=SimpleNamespace(helpers=SimpleNamespace(pose=lambda _:j.zeros(3))),reset=lambda snapshot,path,speed,alpha:alpha)
    batch=training.PathBatch(env,SimpleNamespace(data=None),cfg,911401,None)
    ep=jax.jit(jax.vmap(batch.reset))(j.arange(100,140),j.arange(40))
    def pair(episodes,done,fault):
        flags=dict(done=done,engineering_fault=fault)
        expected=jax.vmap(lambda e,fl:training.reset_finished_episode(e,fl,batch.reset))(episodes,flags)
        actual=training.reset_finished_batch(episodes,flags,batch.reset)
        return expected,actual
    compare=jax.jit(pair)
    for done in [j.arange(40)==17,j.arange(40)%3==1,j.ones(40,bool)]:
        expected,actual=compare(ep,done,j.arange(40)==9)
        for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
            # Reset batch width changes floating-point prefix-scan reduction order.
            # Use the user's accepted 9e-5 tolerance, never a relative escape.
            if np.asarray(a).dtype.kind in 'fc':np.testing.assert_allclose(a,b,rtol=0,atol=9e-5)
            else:np.testing.assert_array_equal(a,b)
