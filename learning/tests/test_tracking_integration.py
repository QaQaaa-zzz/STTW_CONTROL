"""Real CPU controller/physics and identity regression for opt-in tracking."""
from dataclasses import asdict, replace
import json
import os
import jax
import jax.numpy as jp
import numpy as np
import pytest
from sttw_control.env import RecoveryEnv,load_config,config_from_dict
from sttw_control.tracking_reward import TrackingConfig,return_observation,transition,initial_return
from sttw_control.observation import observation_fields
from sttw_control.training import normalization,TrainingConfig
from sttw_control.network import ResidualActor,make_policy_identity,save_policy,load_policy


def config(**kwargs):
    return replace(load_config('learning/configs/path_priority_recovery.json'),**kwargs)


def test_alpha_input_history_normalization_and_export(tmp_path):
    c=config(horizon_seconds=.025,random_events=None);env=RecoveryEnv(c);s=env.reset(3)
    fields=observation_fields(c.observation)
    assert s.obs.shape==(280,)
    assert fields.index('speed_priority')==18
    mean,scale=normalization(c);assert scale.shape==s.obs.shape
    a,b=env.set_priority(s,0.),env.set_priority(s,1.)
    assert a.history.frames[-1,18]==0 and b.history.frames[-1,18]==1
    # Only the newest alpha field changes; actual histories are not fabricated.
    assert np.count_nonzero(np.asarray(a.obs-b.obs))==1
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
    actor=ResidualActor();params=actor.init(jax.random.PRNGKey(5),s.obs)
    identity=make_policy_identity(env.bundle.identity,asdict(c),10)
    assert identity['observation_fields']==list(fields)
    save_policy(tmp_path/'policy',params,mean,scale,identity)
    restored=load_policy(tmp_path/'policy',expected=identity)
    np.testing.assert_allclose(restored(b.obs),actor.apply(params,b.obs/scale),atol=1e-6)
    assert not np.allclose(restored(a.obs),restored(b.obs),atol=1e-8)


def test_zero_residual_physical_baseline_unchanged_and_alpha_not_authority():
    c=config(horizon_seconds=.025,random_events=None)
    legacy=replace(c,tracking=None,speed_schedule=None,observation=replace(c.observation,include_tracking=False))
    env,old=RecoveryEnv(c),RecoveryEnv(legacy)
    a,b=env.reset(12),old.reset(12)
    for _ in range(5):
        a,b=env.step(a,jp.zeros(2)),old.step(b,jp.zeros(2))
        # Dynamic speed lookup and a constant request can differ by float32
        # rounding (observed <2e-6 rad/s); preserve physical equivalence.
        np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(a.base,b.base,atol=3e-6,rtol=1e-7)
    s=env.reset(12);u=jp.array([.2,-.3]);targets=[]
    for alpha in (0.,.5,1.):
        sa=env.set_priority(s,alpha)
        targets.append(env.prepare_action(sa,u)[1])
    np.testing.assert_array_equal(targets[0],targets[1]);np.testing.assert_array_equal(targets[1],targets[2])
    s=env.step(s,u);assert np.any(np.asarray(s.tracking_state.previous_action)!=0)
    fresh=env.reset(12)
    np.testing.assert_array_equal(return_observation(fresh.tracking_state,c.tracking),np.zeros(8))
    assert np.count_nonzero(np.asarray(fresh.history.mask))==1


def test_config_rejects_hidden_preference_wrong_authority_or_no_geometry():
    c=config()
    with pytest.raises(ValueError):replace(c,observation=replace(c.observation,include_priority=False))
    with pytest.raises(ValueError):replace(c,actuator=replace(c.actuator,composition='full_range'))
    with pytest.raises(ValueError):replace(c,priority=replace(c.priority,risk_gate=True))
    with pytest.raises(ValueError):replace(c,learning_roll_reference=.12)
    with pytest.raises(ValueError):replace(c,bend=None)
    with pytest.raises(ValueError):replace(c,horizon_seconds=11.)
    with pytest.raises(ValueError):replace(c,speed_schedule=((0.,1.),))
    with pytest.raises(ValueError):replace(c,speed_schedule=((0.,2.1),(0.,2.3)))


def test_speed_schedule_visible_and_uses_pre_transition_request():
    c=config(horizon_seconds=.02,random_events=None,speed_schedule=((0.,2.1),(.005,2.5)))
    env=RecoveryEnv(c);s=env.reset(2)
    assert float(env.command(0,s.pose)[1])==pytest.approx(2.1)
    assert float(env.command(1,s.pose)[1])==pytest.approx(2.5)
    out=env._advance(s,s.measurement,s.actuator,jp.zeros(2),False,True,2.1,s.pose)
    # Reward still uses 2.1, the request used for this transition, not upcoming 2.5.
    w=(1+9*float(s.priority_alpha))/11
    assert float(out.tracking_components['speed_tracking'])==pytest.approx(.005*4*w,rel=1e-5)
    assert float(out.history.frames[-1,10])==pytest.approx(2.5)


def test_true_geometric_error_affects_reward_and_failure_replaces_parts():
    c=config(horizon_seconds=.02,random_events=None);env=RecoveryEnv(c);s=env.reset(5)
    clean=env._advance(s,s.measurement,s.actuator,jp.zeros(2),False,True,2.1,s.pose)
    offset=env._advance(s,s.measurement,s.actuator,jp.zeros(2),False,True,2.1,s.pose.at[1].add(.5))
    assert float(offset.tracking_components['path_tracking'])<float(clean.tracking_components['path_tracking'])
    failed=env._advance(s,s.measurement,s.actuator,jp.zeros(2),True,True,2.1,s.pose)
    assert bool(failed.terminated) and float(failed.reward)==-100.
    assert all(float(v)==0 for k,v in failed.tracking_components.items() if k!='failure')
    timed=env._advance(s.replace(tick=jp.int32(env.horizon-1)),s.measurement,s.actuator,jp.zeros(2),False,True,2.1,s.pose)
    assert bool(timed.truncated) and not bool(timed.terminated)


def test_new_optional_fields_do_not_change_legacy_identity():
    c=load_config('learning/configs/command_recovery.json');raw=asdict(c)
    old=json.loads(json.dumps(raw));old.pop('tracking');old.pop('speed_schedule');old['observation'].pop('include_tracking')
    ident=make_policy_identity({'fixture':True},raw,10)
    assert ident==make_policy_identity({'fixture':True},old,10)
    assert normalization(c)[1].shape==(190,)


def test_trace_reconstructs_all_signed_terms_and_final_hold(tmp_path):
    from sttw_control.evaluation import evaluate
    from sttw_control.tracking_diagnostics import audit_trace,write_diagnostics
    c=config(horizon_seconds=.03,random_events=None)
    result=evaluate(RecoveryEnv(c),tmp_path/'baseline',seed=7,priority_alpha=1.)
    assert not result['task_recovery_success'] and not result['terminal_tracking_hold']
    audited=audit_trace(tmp_path/'baseline');assert audited['max_reward_error']<3e-5
    with np.load(tmp_path/'baseline/trace.npz') as t:
        assert 'reward_action_delta' in t and 'return_state' in t
    evaluate(RecoveryEnv(c),tmp_path/'residual',seed=7,priority_alpha=1.,
             policy=lambda obs:jp.zeros(2),policy_identity={'engineering':'zero-residual'})
    out=write_diagnostics(tmp_path/'residual',baseline=tmp_path/'baseline')
    assert out['paired_return_delta']==pytest.approx(0.)
    assert (tmp_path/'residual/analysis/tracking/step_reward.png').exists()
    assert (tmp_path/'residual/analysis/tracking/trajectory.pdf').exists()
    assert (tmp_path/'residual/analysis/tracking/INDEX.md').exists()


def test_training_config_has_short_rollout_large_minibatches_and_no_gate():
    cfg=TrainingConfig(**json.loads(open('learning/configs/ppo_path_priority.json').read()))
    assert cfg.num_envs*cfg.rollout_steps*cfg.updates==4194304
    assert cfg.num_envs*cfg.rollout_steps//cfg.minibatch_size*cfg.epochs==16
    assert cfg.initial_std==.15 and cfg.gamma==.9995
    c=config();assert c.priority.randomize_alpha and not c.priority.risk_gate


@pytest.mark.skipif(os.environ.get('STTW_TEST_MJX_CPU')!='1',reason='opt-in bounded MJX CPU compilation smoke')
def test_mjx_new_environment_shapes_and_one_real_physics_step():
    c=config(horizon_seconds=.005,random_events=None)
    env=RecoveryEnv(c,backend='mjx')
    state=jax.jit(env.reset)(jax.random.PRNGKey(2))
    state=jax.jit(env.step)(state,jp.zeros(2));jax.block_until_ready(state.reward)
    assert state.obs.shape==(280,) and np.isfinite(float(state.reward))
    assert bool(state.truncated)
    assert float(sum(state.tracking_components.values()))==pytest.approx(float(state.reward))
