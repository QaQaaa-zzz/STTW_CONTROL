"""Math and state tests require NumPy only; JAX equivalence is optional."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

# Load the pure module without importing simulation/Flax or the package __init__.
path = Path(__file__).parents[1] / 'src/sttw_control/tracking_reward.py'
spec = importlib.util.spec_from_file_location('sttw_tracking_reward_test', path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
C = m.TrackingRewardConfig()


def parts(alpha=0.5, **changes):
    kw = dict(roll=0., roll_rate=0., speed_error=0., lateral=0., heading=0.,
              action=np.zeros(2), previous_action=np.zeros(2), alpha=alpha,
              tracking=m.initial_tracking(), config=C, dt=.005)
    kw.update(changes)
    return m.reward_components(**kw)


@pytest.mark.parametrize('alpha', [0., .25, .5, .75, 1.])
def test_constant_weight_sum_and_perfect_reward(alpha):
    wv, wp = m.priority_weights(alpha, C)
    assert wv + wp == pytest.approx(1.)
    assert sum(parts(alpha).values()) == pytest.approx(.025)


def test_preference_endpoints_and_minimum_nonpreferred_weight():
    assert m.priority_weights(0, C)[1] / m.priority_weights(0, C)[0] == pytest.approx(10)
    assert m.priority_weights(1, C)[0] / m.priority_weights(1, C)[1] == pytest.approx(10)
    assert sum(parts(0, lateral=.2).values()) < sum(parts(1, lateral=.2).values())
    assert sum(parts(1, speed_error=.4).values()) < sum(parts(0, speed_error=.4).values())


def test_far_errors_have_cost_and_no_total_positive_clipping():
    assert sum(parts(speed_error=5., lateral=3.).values()) < 0
    assert sum(parts(lateral=.8).values()) > sum(parts(lateral=1.5).values())


def test_failure_is_replacement_and_not_scaled_by_dt():
    p = parts(failed=True, speed_error=1., lateral=1., action=np.ones(2))
    assert p['failure'] == -100
    assert all(x == 0 for k, x in p.items() if k != 'failure')
    assert sum(parts(failed=True, dt=.02).values()) == -100


def test_dt_applied_once_and_slew_uses_residual_not_base_command():
    p = parts(action=np.array([.2, -.3]), previous_action=np.array([.1, -.1]))
    assert p['action_change'] == pytest.approx(-.005*.02*(.1**2+.2**2))
    assert sum(parts(dt=.02).values()) == pytest.approx(4*sum(parts().values()))


def test_working_roll_penalty_is_alpha_independent_and_allows_lean():
    assert parts(roll=.2)['roll'] == 0
    assert parts(0, roll=.4)['roll'] == pytest.approx(parts(1, roll=.4)['roll'])
    assert parts(roll=.4)['roll'] == pytest.approx(-.005)


def tick(state, *, lateral=0., speed=0., failed=False, action=None):
    return m.advance_tracking(state, lateral, 0., speed, 0., 0.,
                              np.zeros(2) if action is None else action, failed, C, .005)


def test_nominal_hold_never_counts_as_recovery():
    s = m.initial_tracking()
    for _ in range(120): s = tick(s)
    assert not s.ever_departed and not s.task_recovered and not s.pending


def test_departure_requires_full_hold_then_returns_to_original_path():
    s = tick(m.initial_tracking(), lateral=.3)
    assert s.pending and s.ever_departed
    for _ in range(99): s = tick(s)
    assert not s.task_recovered and s.pending
    s = tick(s)
    assert s.task_recovered and not s.pending
    assert s.age_ticks == 0


def test_speed_failure_cannot_be_reported_as_joint_task_recovery():
    s = tick(m.initial_tracking(), lateral=.3)
    for _ in range(110): s = tick(s, speed=.5)
    assert not s.pending  # returned geometrically
    assert not s.task_recovered and s.joint_hold_ticks == 0


def test_true_speed_is_not_leaked_into_actor_context():
    s = tick(m.initial_tracking(), lateral=.3)
    a, b = tick(s, speed=0.), tick(s, speed=.8)
    np.testing.assert_array_equal(m.tracking_context(a, C, .005), m.tracking_context(b, C, .005))


def test_deadline_is_observed_departure_not_event_label_and_no_reset_leak():
    s = m.initial_tracking()
    for _ in range(601): s = tick(s, lateral=.3)
    assert s.deadline_missed and s.pending
    assert parts(tracking=s)['return_overdue'] < 0
    fresh = m.initial_tracking()
    assert not fresh.deadline_missed and not fresh.pending
    assert not tick(s, failed=True).task_recovered


@pytest.mark.parametrize('field,value', [('speed_wide', 0), ('priority_ratio', .5),
    ('tracking_rate', float('nan')), ('hold_seconds', 4), ('speed_tolerance_speed_priority', 1),
    ('lateral_tolerance_path_priority', 1)])
def test_invalid_config_rejected(field, value):
    with pytest.raises(ValueError): replace(C, **{field: value})


def test_numpy_jax_jit_vmap_and_scan_equivalence():
    jax = pytest.importorskip('jax')
    jp = pytest.importorskip('jax.numpy')
    def score(alpha):
        return sum(m.reward_components(.32, .15, .25, .12, .07, jp.array([.1, -.2]),
            jp.zeros(2), alpha, m.initial_tracking(xp=jp), C, .005, xp=jp).values())
    alphas = jp.array([0., .5, 1.])
    actual = np.asarray(jax.jit(jax.vmap(score))(alphas))
    expected = [sum(parts(float(a), roll=.32, roll_rate=.15, speed_error=.25,
                        lateral=.12, heading=.07, action=np.array([.1, -.2])).values()) for a in alphas]
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    def scan_tick(s, y):
        new = m.advance_tracking(s, y, 0., 0., 0., 0., jp.zeros(2), False, C, .005, xp=jp)
        return new, m.tracking_context(new, C, .005, xp=jp)
    final, context = jax.jit(lambda ys:jax.lax.scan(scan_tick, m.initial_tracking(xp=jp), ys))(
        jp.concatenate((jp.array([.3]), jp.zeros(100))))
    assert bool(final.task_recovered) and context.shape == (101, 5)
