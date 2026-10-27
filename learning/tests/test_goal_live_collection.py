"""CPU collector integration with a mocked plant advance, not new physics."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import jax
import jax.numpy as j
from flax import struct
from sttw_control.goal_live_collection import (GoalLiveCollectionState,
    make_goal_live_collection_step, original_goal_context, goal_latch_from_block)
from sttw_control.reference_reentry import actor_permission
from sttw_control.causal_reference_baseline import initial_gate


@struct.dataclass
class Physical:
    failed: object
    data: object
    raw: object = struct.field(default_factory=lambda: j.zeros(2))


@struct.dataclass
class Follower:
    invalid: object


@struct.dataclass
class Projection:
    progress: object


@struct.dataclass
class State:
    tick: object
    fault: object
    domain_exit: object
    physical: object
    follower: object
    offsets: object = struct.field(default_factory=lambda: j.zeros(2))
    projection: object = struct.field(default_factory=lambda: Projection(j.float32(0.)))


def block(**changes):
    logs = dict(active_tick=j.ones(4, bool), actual_xy=j.array([[5., 5.], [0., 0.], [5., 5.], [5., 5.]]),
        contact_failure=j.zeros(4, bool), nonfinite=j.zeros(4, bool), lower_fault=j.zeros(4, bool),
        policy_fault=j.zeros(4, bool), horizon_reached=j.zeros(4, bool))
    logs.update(changes)
    return logs


def fake_reference(env, actor, *, horizon_s, allow_reentry):
    assert allow_reentry
    def advance(carry, raw, path, speed, pulse, profile, floor, zero):
        s = carry.state; active = ~(s.physical.failed | s.fault | (s.tick >= 2000))
        legal = actor_permission(s.domain_exit, s.follower.invalid, carry.clean_blocks, zero, ~active)
        logs = block(active_tick=j.repeat(active, 4), actor_action_valid=j.repeat(legal & active, 4))
        return carry.replace(state=s.replace(tick=s.tick + j.where(active, 4, 0))), logs
    return advance


def setup(invalid=False, traceable=False):
    def observation(s, path, speed, horizon):
        x = j.full(351, j.nan if invalid else 2.)
        return x, j.r_[x, j.float32(1.)], j.bool_(invalid)
    env = SimpleNamespace(cc=SimpleNamespace(dt=.005),
        config={'residual_interface': {'speed_reference_bounds_m_s': [1., 3.5]}},
        observation=observation,
        physics=SimpleNamespace(observe=lambda data: (None, data, None)))
    state = State(j.int32(0), j.bool_(False), j.bool_(invalid),
        Physical(j.bool_(False), j.array([0., -2., 0.])), Follower(j.bool_(invalid)))
    carry = GoalLiveCollectionState(state, jax.random.PRNGKey(73), initial_gate(), j.int32(0), j.bool_(False))
    seen = []
    def evaluate(params, obs, critic, goal):
        if not traceable:
            seen.append((np.asarray(obs), np.asarray(goal)))
            assert np.isfinite(obs).all() and np.isfinite(goal).all()
        # Distinct frozen heads; changing the goal really changes raw mean.
        return j.array([goal[0], goal[1]]), j.float32(params.get('g', .7)), j.float32(3.), j.float32(.9)
    with patch('sttw_control.anticipatory_reference.make_reference_step', fake_reference):
        step = make_goal_live_collection_step(env, evaluate, horizon_s=10., radius_m=1., distance_scale_m=10.)
    return carry, step, seen


def test_body_goal_context_is_causal_and_rotation_consistent():
    pose = j.array([1., 2., j.pi / 2])
    context = original_goal_context(pose, j.array([1., 4.]), tick=400, dt=.005,
        horizon_s=10., arrived=False, radius_m=1., distance_scale_m=10.)
    np.testing.assert_allclose(context, [.2, 0., .8, 0., .1], atol=1e-7)


def test_frozen_values_and_goal_context_are_saved_before_actual_action():
    carry, step, seen = setup()
    end, (d, logs) = step(carry, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.array([0., 0.]))
    assert bool(d['valid']) and not bool(d['goal_context'][3])
    assert bool(end.arrived) and int(end.state.tick) == 4
    np.testing.assert_allclose([d['old_goal_value'], d['old_cost_value'], d['physical_value']], [.7, 3., .9])
    _, (later, _) = step(end, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.array([0., 0.]))
    assert later['goal_context'][3] == 1. and bool(later['valid'])  # balance not stopped by goal


def test_invalid_views_are_dummy_and_fallback_goal_is_still_measured():
    carry, step, seen = setup(invalid=True)
    end, (d, logs) = step(carry, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.array([0., 0.]))
    assert not bool(d['valid']) and bool(end.arrived)
    assert logs['fallback_active'].all() and np.isnan(d['old_goal_value'])
    np.testing.assert_array_equal(seen[0][0], np.zeros(351))
    np.testing.assert_array_equal(seen[0][1], np.zeros(5))
    np.testing.assert_array_equal(carry.key, end.key)


def test_same_contact_excludes_goal_later_contact_preserves_goal_and_h_is_inclusive():
    contact = j.array([False, True, False, False])
    assert not bool(goal_latch_from_block(False, block(contact_failure=contact), j.zeros(2), 1.))
    assert bool(goal_latch_from_block(False, block(contact_failure=j.array([False, False, True, False])), j.zeros(2), 1.))
    assert bool(goal_latch_from_block(False, block(horizon_reached=contact), j.zeros(2), 1.))
    assert not bool(goal_latch_from_block(False, block(horizon_reached=j.array([True, False, False, False])), j.zeros(2), 1.))


def test_original_goal_changes_sampling_mean_without_future_pulse_input():
    carry, step, seen = setup()
    args = (carry, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0))
    _, (a, _) = step(*args, j.array([0., 0.]))
    _, (b, _) = step(*args, j.array([2., 0.]))
    assert not np.array_equal(a['mean'], b['mean'])
    np.testing.assert_allclose(b['raw_action'] - a['raw_action'], b['mean'] - a['mean'], atol=1e-6)


def test_malformed_probability_and_nonfinite_original_goal_stop_before_action():
    carry, step, seen = setup()
    end, (d, logs) = step(carry, {'g': 1.1}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.zeros(2))
    assert bool(end.state.fault) and not bool(d['valid']) and not logs['active_tick'].any()
    carry, step, seen = setup(invalid=True)
    end, (d, logs) = step(carry, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.array([j.nan, 0.]))
    assert bool(end.state.fault) and not bool(d['valid']) and not logs['active_tick'].any()


def test_reentry_keeps_measured_latch_and_current_clock():
    carry, step, seen = setup(invalid=True)
    end, _ = step(carry, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.zeros(2))
    _, ready_step, _ = setup()
    ready = end.replace(clean_blocks=j.int32(16),
        state=end.state.replace(tick=j.int32(104), follower=Follower(j.bool_(False))))
    _, (d, _) = ready_step(ready, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.zeros(2))
    assert bool(d['valid']) and int(d['task_tick']) == 104 and d['goal_context'][3] == 1.
    np.testing.assert_allclose(d['goal_context'][2], .948, atol=1e-7)


def test_trace_and_cpu_jit_preserve_shape_and_first_hit_without_plant_integration():
    carry, step, _ = setup(traceable=True)
    args = (carry, {}, j.zeros(2), None, 1.5, j.zeros(5), j.int32(0), j.zeros(2))
    with jax.default_device(jax.devices('cpu')[0]):
        shape = jax.eval_shape(step, *args)
        assert shape[1][0]['goal_context'].shape == (5,)
        end, (d, _) = jax.jit(step)(*args)
        assert bool(end.arrived) and bool(d['valid'])


def test_real_reference_parameter_entry_accepts_wrapper_without_plant_integration():
    from sttw_control.anticipatory_reference import make_reference_step
    from sttw_control.goal_live_collection import _SampledAction
    carry, _, _ = setup()
    env = SimpleNamespace(cc=SimpleNamespace(dt=.005),
        config={'residual_interface': {'speed_reference_bounds_m_s': [1., 3.5]}},
        observation=lambda *args: (j.zeros(351), j.zeros(352), j.bool_(False)),
        physics=SimpleNamespace(observe=lambda data: (None, data, None)))
    def plant_factory(local, *, horizon_s):
        def advance(s, action, path, speed, pulse):
            scalars = ['actual_forward_speed', 'wheel_speed_proxy', 'phi', 'phi_dot',
                'actual_delta', 'peak_roll', 'time', 'path_cross_track', 'path_heading_error',
                'path_progress', 'goal_section_signed_distance', 'tick_reward', 'scored_tick_reward',
                'chi', 'yaw_unwrapped']
            flags = ['contact_failure', 'physical_failure', 'nonfinite', 'lower_fault',
                'policy_fault', 'domain_exit']
            logs = {k: j.zeros(4) for k in scalars}
            logs.update({k: j.zeros(4, bool) for k in flags})
            logs.update(active_tick=j.ones(4, bool), actual_xy=j.zeros((4, 2)),
                physical_tick=j.arange(4), applied_force_impulse=j.zeros((4, 3)),
                governed=j.zeros((4, 2)), nominal=j.zeros((4, 2)), filtered_offset=j.zeros((4, 2)),
                offsets=j.zeros((4, 2)), target_offset=j.tile(action, (4, 1)),
                raw_components={'speed': j.zeros(4)}, effective_components={'speed': j.zeros(4)})
            return s.replace(tick=s.tick + 4), logs
        return advance
    candidate = lambda original, *args: (original, j.float32(0.), original)
    causal = lambda original, obs, offsets, tick, gate, cfg, **kw: (original, gate,
        {'engineering_fault': j.bool_(False), 'detected': j.bool_(False),
         'override_active': j.bool_(False), 'signal': j.float32(0.)})
    with patch('sttw_control.live_path_continuation.make_live_reference_step', plant_factory), \
         patch('sttw_control.anticipatory_reference.candidate_latent', candidate), \
         patch('sttw_control.causal_reference_baseline.causal_reference_action', causal), \
         patch('sttw_control.path_command_reward.configured_cost_keys', lambda cfg: ('speed',)):
        reference = make_reference_step(env, _SampledAction(), horizon_s=10., allow_reentry=True)
        raw = j.array([.3, -.2])
        end, logs = reference(carry, {'latent': raw, 'support_independent_permission': False},
            SimpleNamespace(turn_end=j.float32(4.)), 1.5, j.zeros(5), j.int32(0), j.float32(1.), j.bool_(False))
        goal_step = make_goal_live_collection_step(env,
            lambda params, obs, critic, context: (j.zeros(2), j.float32(.5), j.float32(2.), j.float32(.8)),
            horizon_s=10., radius_m=1., distance_scale_m=10.)
        full_end, (decision, full_logs) = goal_step(carry, {}, j.zeros(2),
            SimpleNamespace(turn_end=j.float32(4.)), 1.5, j.zeros(5), j.int32(0), j.zeros(2))
        assert bool(decision['valid']) and int(full_end.state.tick) == 4
        np.testing.assert_allclose(full_logs['target_offset'], np.tile(decision['raw_action'], (4, 1)))
    assert int(end.state.tick) == 4
    np.testing.assert_allclose(logs['target_offset'], np.tile(raw, (4, 1)))
