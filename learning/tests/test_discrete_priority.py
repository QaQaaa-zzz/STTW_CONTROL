import jax
import numpy as np
import pytest
from sttw_control.priority import PriorityConfig,sample_alpha
from sttw_control.training_diagnostics import alpha_sample_sums,alpha_sample_summary


def test_discrete_sampling_is_uniform_and_only_declared_values():
    c=PriorityConfig(training_alphas=(0.,.1,1.))
    values=np.asarray(jax.jit(jax.vmap(lambda k:sample_alpha(k,c)))(jax.random.split(jax.random.PRNGKey(6),6000)))
    unique,count=np.unique(values,return_counts=True)
    np.testing.assert_allclose(unique,[0.,.1,1.]);assert np.all(abs(count-2000)<160)


def test_legacy_sampling_is_identical():
    key=jax.random.PRNGKey(9)
    assert float(sample_alpha(key,PriorityConfig()))==float(jax.random.uniform(jax.random.fold_in(key,31)))
    assert float(sample_alpha(key,PriorityConfig(randomize_alpha=False,fixed_alpha=1.)))==1.


@pytest.mark.parametrize('choices',[(),(0.,0.),(-.1,1.),(0.,float('nan'))])
def test_invalid_choices(choices):
    with pytest.raises(ValueError):PriorityConfig(training_alphas=choices)


def test_exact_alpha_logging_separates_zero_and_point_one():
    choices=(0.,.1,1.)
    sums=alpha_sample_sums(np.array([0.,.1,1.,.1]),np.array([1.,2.,3.,4.]),np.zeros(4),{},choices=choices)
    rows=alpha_sample_summary(sums,choices=choices)
    assert [r['samples'] for r in rows]==[1,2,1]
    assert [r['alpha'] for r in rows]==list(choices)
    assert [r['mean_step_reward'] for r in rows]==[1.,3.,3.]


def test_real_cpu_reset_alpha_is_discrete_and_stays_fixed():
    from dataclasses import replace
    from sttw_control.env import RecoveryEnv,load_config
    cfg=replace(load_config('learning/configs/discrete_alpha_ecbc1.json'),preparation_seconds=0.)
    env=RecoveryEnv(cfg,backend='cpu')
    for seed in range(3):
        state=env.reset(jax.random.PRNGKey(seed));alpha=float(state.priority_alpha)
        assert min(abs(alpha-v) for v in (0.,.1,1.))<1e-7
        state=env.step(state,np.zeros(2))
        assert float(state.priority_alpha)==alpha


def test_three_arm_budget_and_only_declared_differences():
    import json
    from pathlib import Path
    tasks=[json.loads(Path(f'learning/configs/discrete_alpha_{name}.json').read_text()) for name in ['ecbc1','ecbc08','direct']]
    for task in tasks:
        assert task['priority']['training_alphas']==[0.,.1,1.]
        task.pop('actuator')
    assert tasks[0]==tasks[1]==tasks[2]
    train=json.loads(Path('learning/configs/ppo_discrete_alpha.json').read_text())
    assert train['updates']==200 and train['num_envs']*train['rollout_steps']*train['updates']==26214400
    assert not train['training_reward_best_enabled']


def test_optional_discrete_sampling_preserves_legacy_identity():
    from sttw_control.network import make_policy_identity
    old={'priority':{'randomize_alpha':True},'observation':{}}
    default={'priority':{'randomize_alpha':True,'training_alphas':None},'observation':{}}
    discrete={'priority':{'randomize_alpha':True,'training_alphas':(0.,.1,1.)},'observation':{}}
    assert make_policy_identity({},old,10)==make_policy_identity({},default,10)
    assert make_policy_identity({},old,10)!=make_policy_identity({},discrete,10)
