"""The endpoint panel must not select winners or manufacture replication."""
import copy
import numpy as np
import pytest

from sttw_control.survival_endpoint_protocol import (
    make_endpoint_cases, validate_endpoint_metadata, choose_evaluation_action,
)


def contexts():
    physical = [dict(route=route, speed=speed, force_N=force,
                     pulse_onset_task_tick=tick, pulse_duration_s=.1,
                     world_direction=[0., 1., 0.])
                for route, speed, tick, forces in [('left', 1.8, 464, [0., 32.]),
                                                   ('right', 1.5, 488, [0., -32.])]
                for force in forces]
    nominal = [dict(route=route, speed=speed, force_N=0.)
               for route in ['left110_R2p6', 'right110_R2p6'] for speed in [2.35, 2.75]]
    return physical, nominal


def test_panel_counts_and_zero_baseline_are_real_unique_episodes():
    cases = make_endpoint_cases(*contexts())
    assert len(cases) == 488
    assert len({c['case_id'] for c in cases}) == 488
    assert sum(c['panel'] == 'physical_deterministic' for c in cases) == 84
    assert sum(c['panel'] == 'physical_stochastic' for c in cases) == 320
    assert sum(c['panel'] == 'nominal_deterministic' for c in cases) == 84
    zero = [c for c in cases if c['role'] == 'zero_middle']
    assert len(zero) == 8
    assert all(c['alpha'] is None and c['inference_mode'] == 'zero' for c in zero)
    assert all(c['speed_reference_floor_m_s'] == (1.5 if c['role'] == 'parent_speed_floor' else None)
               for c in cases)


def test_common_noise_pairs_models_alphas_and_force_but_not_replicates_or_routes():
    rows = [c for c in make_endpoint_cases(*contexts()) if c['panel'] == 'physical_stochastic']
    for route in ['left', 'right']:
        for rep in range(4):
            group = [c for c in rows if c['route'] == route and c['replicate'] == rep]
            assert len(group) == 40
            assert len({c['rng_fold_in'] for c in group}) == 1
    assert len({c['rng_fold_in'] for c in rows}) == 8


def test_bad_contexts_rejected_instead_of_silently_changing_protocol():
    physical, nominal = contexts()
    physical[1]['pulse_duration_s'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        make_endpoint_cases(physical, nominal)
    physical, nominal = contexts()
    physical[1]['speed'] += .1
    with pytest.raises(ValueError, match='paired'):
        make_endpoint_cases(physical, nominal)


def metadata():
    plan = dict(updates=64, critic_only_updates=2, epochs=4, arms={'task_only': 0., 'physical_credit': 1.},
                seed=17, learner_seed=18, model={'checkpoint_sha256': 'parent'},
                effective_config_sha256='serialized-file-bytes', actor_lr=1e-4,
                critic_lr=3e-4, clip=.2, entropy=.001, grad_clip=1., fixed_denominator=24000)
    pair = dict(completed_updates=64, next_noise_update_index=64, base_noise_seed=17,
                source_initialization_sha256='parent', plan_sha256='plan',
                models=[dict(arm=a, next_collection=64) for a in plan['arms']])
    states = {a: dict(schema='sttw_dual_survival_boundary_v1', exact_boundary_resume=True,
                      next_collection=64, actor_steps=248, critic_steps=256,
                      optimizer_steps=[248, 256, 256], config_sha256='config',
                      settings=dict(beta=b, epochs=4, seed=18,
                                    **{k: plan[k] for k in ('actor_lr', 'critic_lr', 'clip',
                                       'entropy', 'grad_clip', 'fixed_denominator')}))
              for a, b in plan['arms'].items()}
    return plan, pair, states


def test_only_complete_declared_endpoint_is_eligible():
    plan, pair, states = metadata()
    assert validate_endpoint_metadata(plan, pair, states, plan_sha256='plan', config_sha256='config')['actor_steps_per_arm'] == 248
    pair['completed_updates'] = 63
    with pytest.raises(ValueError, match='endpoint'):
        validate_endpoint_metadata(plan, pair, states, plan_sha256='plan', config_sha256='config')


@pytest.mark.parametrize('change', ['critic', 'beta', 'duplicate', 'plan', 'config', 'learning_rate', 'seed'])
def test_mismatched_endpoint_identity_and_optimizer_counts_rejected(change):
    plan, pair, states = metadata()
    if change == 'critic': states['physical_credit']['optimizer_steps'][2] -= 1
    if change == 'beta': states['physical_credit']['settings']['beta'] = 0.
    if change == 'duplicate': pair['models'][1] = copy.deepcopy(pair['models'][0])
    if change == 'plan': pair['plan_sha256'] = 'other'
    if change == 'config': states['task_only']['config_sha256'] = 'other'
    if change == 'learning_rate': states['physical_credit']['settings']['actor_lr'] = .01
    if change == 'seed': states['physical_credit']['settings']['seed'] += 1
    with pytest.raises(ValueError):
        validate_endpoint_metadata(plan, pair, states, plan_sha256='plan', config_sha256='config')


def test_explicit_deterministic_and_zero_actions_have_no_gaussian_density_or_rng_use():
    import jax
    import jax.numpy as j
    key = jax.random.PRNGKey(9)
    mean, log_std = j.array([.4, -.2]), j.array([-.5, -.7])
    for mode, expected in [('mean', mean), ('zero', j.zeros(2))]:
        action, logp, density_valid, next_key = jax.jit(
            lambda m, s, k: choose_evaluation_action(m, s, k, mode=mode))(mean, log_std, key)
        np.testing.assert_array_equal(action, expected)
        np.testing.assert_array_equal(next_key, key)
        assert not bool(density_valid) and np.isnan(logp)
    action, logp, density_valid, next_key = choose_evaluation_action(mean, log_std, key, mode='gaussian')
    assert bool(density_valid) and np.isfinite(logp)
    assert not np.array_equal(next_key, key)
    assert not np.array_equal(action, mean)
