import jax
import jax.numpy as j
import numpy as np
from sttw_control import margin_state_sampling as sampler


def example():
    groups=j.array([0,0,4,4])
    risk=j.abs(j.array([-.6,.6,-.6,.6])[:,None]+j.array([1.4,-1.4])[None,:])
    return groups,risk


def test_directional_focus_keeps_alpha_group_and_uniform_support():
    groups,risk=example();p=sampler.state_probabilities(risk,groups,strength=.7)
    assert p.shape==(2,5,4)
    assert p[0,0,0]>p[0,0,1] and p[1,0,1]>p[1,0,0]
    np.testing.assert_allclose(p.sum(-1),1.,atol=1e-6)
    assert np.all(np.asarray(p[:,0,:2])>=.3/2)
    np.testing.assert_array_equal(p[:,0,2:],0)
    np.testing.assert_array_equal(p[:,4,:2],0)


def test_shuffle_preserves_weight_distribution_within_alpha_stratum():
    groups,risk=example();perm=sampler.stratified_permutation(np.asarray(groups),71)
    np.testing.assert_array_equal(np.asarray(groups)[perm],groups)
    p=sampler.state_probabilities(risk,groups,strength=.7)
    q=sampler.state_probabilities(risk[perm],groups,strength=.7)
    np.testing.assert_allclose(q,p[:,:,perm],atol=1e-6)
    np.testing.assert_array_equal(np.sort(perm),np.arange(4))


def test_force_draws_and_preference_strata_are_identical_across_arms():
    groups,risk=example();uniform=sampler.state_probabilities(risk,groups,strength=0.)
    guided=sampler.state_probabilities(risk,groups,strength=.7)
    keys=jax.random.split(jax.random.PRNGKey(983),2048)
    def draws(p):return jax.jit(jax.vmap(lambda key:sampler.draw(key,p,groups)))(keys)
    a,b=draws(uniform),draws(guided)
    np.testing.assert_array_equal(a['force_index'],b['force_index'])
    np.testing.assert_array_equal(a['enabled'],b['enabled'])
    np.testing.assert_array_equal(groups[a['state_index']],groups[b['state_index']])
    assert np.any(np.asarray(a['state_index'])!=np.asarray(b['state_index']))
    np.testing.assert_array_equal(a['state_index'][~a['enabled']],b['state_index'][~b['enabled']])


def test_invalid_risk_falls_back_to_uniform_without_dropping_states():
    groups,_=example();risk=j.full((4,2),j.nan)
    a=sampler.state_probabilities(risk,groups,strength=.7)
    b=sampler.state_probabilities(risk,groups,strength=0.)
    np.testing.assert_allclose(a,b)


def test_continuous_alphas_and_endpoints_have_separate_strata():
    actual=sampler.alpha_strata(j.array([0.,.1,.4,.8,1.]))
    np.testing.assert_array_equal(actual,[0,1,2,3,4])
