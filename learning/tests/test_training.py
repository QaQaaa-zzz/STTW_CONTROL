import numpy as np
import jax.numpy as jp
from sttw_control.training import generalized_advantage, gaussian_log_prob


def test_gae_bootstraps_timeout_but_never_crosses_reset():
    rewards=jp.array([[1.],[2.]])
    values=jp.array([[.5],[.6]])
    next_values=jp.array([[.6],[10.]])
    terminated=jp.array([[False],[False]])
    done=jp.array([[False],[True]])
    advantage,returns=generalized_advantage(rewards,values,next_values,terminated,done,.9,1.)
    np.testing.assert_allclose(advantage[:,0],[10.4,10.4],rtol=1e-6)
    terminated=terminated.at[1,0].set(True)
    _,returns=generalized_advantage(rewards,values,next_values,terminated,done,.9,1.)
    np.testing.assert_allclose(returns[:,0],[2.8,2.],rtol=1e-6)


def test_logprob_is_finite_for_large_latent_actions():
    result=gaussian_log_prob(jp.array([[10.,-10.]]),jp.zeros((1,2)),jp.zeros(2))
    np.testing.assert_allclose(result,[-101.837877],rtol=1e-6)


def test_path_observation_checkpoint_and_reward(tmp_path):
    import jax
    from dataclasses import asdict,replace
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.network import ResidualActor,make_policy_identity,save_policy,load_policy
    from sttw_control.training import normalization
    cfg=load_config('learning/configs/circle_learning.json')
    env=RecoveryEnv(cfg)
    state=env.reset(17)
    assert state.obs.shape==(19,)
    np.testing.assert_allclose(state.obs[-4:-1],[0,0,1/3],atol=.03)
    actor=ResidualActor()
    params=actor.init(jax.random.PRNGKey(0),state.obs)
    mean,std=normalization(cfg)
    identity=make_policy_identity(env.bundle.identity,asdict(cfg),1)
    save_policy(tmp_path/'policy',params,mean,std,identity)
    restored=load_policy(tmp_path/'policy',expected=identity)
    np.testing.assert_allclose(restored(state.obs),actor.apply(params,state.obs/std),atol=1e-6)
    plain=RecoveryEnv(replace(cfg,path_error_weight=0.,heading_error_weight=0.))
    pose=state.pose.at[0].add(1.)
    args=(state,state.measurement,state.actuator,jp.zeros(2),False,True,2.,pose)
    penalized=env._advance(*args).reward
    original=plain._advance(*args).reward
    assert float(penalized)<float(original)


def test_history_observations_preserve_baseline_and_reset_mask():
    from dataclasses import replace
    from sttw_control.env import RecoveryEnv,load_config
    from sttw_control.observation import ObservationConfig
    cfg=load_config('learning/configs/disturbance_learning.json')
    plain=RecoveryEnv(cfg)
    history=RecoveryEnv(replace(cfg,observation=ObservationConfig(history_steps=20,include_path=True)))
    a,b=plain.reset(19),history.reset(19)
    assert b.obs.shape==(380,)
    assert np.count_nonzero(np.asarray(b.obs[-20:]))==1
    for _ in range(22):
        a,b=plain.step(a,jp.zeros(2)),history.step(b,jp.zeros(2))
        np.testing.assert_allclose(a.data.qpos,b.data.qpos,atol=1e-10,rtol=0)
    assert np.count_nonzero(np.asarray(b.obs[-20:]))==20
    b=history.reset(19)
    assert np.count_nonzero(np.asarray(b.obs[-20:]))==1
    np.testing.assert_allclose(b.obs[:-20].reshape(20,18)[:-1],0)
