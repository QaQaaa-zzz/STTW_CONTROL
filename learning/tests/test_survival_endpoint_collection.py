"""Evaluation controls must keep task identity and physical-tail semantics."""
import copy
from types import SimpleNamespace
import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import struct

from sttw_control.survival_collection import SurvivalCollectionState
from sttw_control.survival_endpoint_collection import (
    evaluation_config, normal_evaluation_interval,
)


def config():
    return dict(timing={'lower_dt_s': .005},
                residual_interface=dict(speed_reference_bounds_m_s=[1., 3.5],
                    steer_reference_bound_rad=.65, filter_tau_s=[.1, .1],
                    correction_speed_down_m_s2=2., correction_speed_up_m_s2=2.,
                    correction_steer_rate_rad_s=1., speed_positive_scale_m_s=.25,
                    speed_negative_scale_m_s=1., steer_scale_rad=.2),
                path_reward={'unchanged': 1}, network={'unchanged': 2})


def test_floor_is_only_a_legal_reference_restriction_not_new_task_reward_or_plant():
    source = config(); saved = copy.deepcopy(source)
    same = evaluation_config(source, speed_floor=None)
    restricted = evaluation_config(source, speed_floor=1.5)
    expected = copy.deepcopy(source)
    expected['residual_interface']['speed_reference_bounds_m_s'][0] = 1.5
    assert restricted == expected and source == saved and same == saved
    same['path_reward']['unchanged'] = 99
    assert source == saved
    for floor in [float('nan'), .9, 3.6]:
        with pytest.raises(ValueError): evaluation_config(source, speed_floor=floor)


def test_floor_keeps_real_filter_back_calculation_and_original_nominal():
    from sttw_control.path_command_policy import correction_tick
    base = config(); cfg = evaluation_config(base, speed_floor=1.5)
    nominal = j.array([1.2, .1]); filtered = j.array([-.2, .03]); offsets = j.array([-.2, .03])
    z = j.array([-1., .2])
    f, out, reference, flags = correction_tick(filtered, offsets, z, nominal, cfg)
    assert float(reference[0]) == 1.5 and bool(flags['reference_clip_channels'][0])
    np.testing.assert_allclose(out, reference-nominal)
    np.testing.assert_allclose(f[0], out[0])
    assert float(nominal[0]) == pytest.approx(1.2)  # requested/nominal task not rewritten


@struct.dataclass
class Physical:
    governor: object


@struct.dataclass
class State:
    tick: object
    fault: object
    domain_exit: object
    physical: object


@struct.dataclass
class Governor:
    current_reference: object


class Actor:
    def apply(self, params, observation):
        return params + observation[:2]


class UnusedActor:
    def apply(self, *_):
        raise AssertionError('zero correction must not evaluate an unused trained Actor')


class Env:
    config = config()
    cc = SimpleNamespace(dt=.005)
    def observation(self, state, path, speed, horizon):
        return j.zeros(351), j.zeros(352), j.bool_(False)

    def policy_step(self, state, action, path, speed, horizon, pulse):
        from sttw_control.path_command_reward import BASE_COST_KEYS
        scalar = j.zeros(4)
        logs = {k: scalar for k in ['actual_forward_speed', 'wheel_speed_proxy', 'phi', 'phi_dot',
            'actual_delta', 'peak_roll', 'path_cross_track', 'path_heading_error', 'path_progress',
            'goal_section_signed_distance', 'tick_reward', 'scored_tick_reward', 'time']}
        logs.update({k: j.zeros(4, bool) for k in ['contact_failure', 'physical_failure',
                     'nonfinite', 'lower_fault', 'policy_fault', 'domain_exit']})
        logs.update(active_tick=j.array([True, True, True, False]),
            actual_xy=j.zeros((4, 2)), reference_xy=j.zeros((4, 2)), governed=j.tile(action, (4, 1)),
            applied_force_impulse=j.tile(pulse[-3:], (4, 1)),
            raw_components={k: scalar for k in BASE_COST_KEYS},
            effective_components={k: scalar for k in BASE_COST_KEYS})
        logs.update(nominal=j.tile(j.array([1.1, .1]), (4, 1)),
                    target_offset=j.tile(action, (4, 1)), filtered_offset=j.tile(action, (4, 1)),
                    offsets=j.tile(action, (4, 1)), reference_clip_channels=j.zeros((4, 2), bool),
                    correction_rate_clipped=j.zeros(4, bool), chi=j.ones(4))
        # This fixture only checks forwarding and log packing; it simulates no physics.
        logs['actual_forward_speed'] = j.repeat(speed, 4)
        return state.replace(tick=state.tick+3, domain_exit=j.bool_(True),
            physical=Physical(Governor(action))), logs


def carry():
    return SurvivalCollectionState(State(j.int32(12), j.bool_(False), j.bool_(False),
            Physical(Governor(j.zeros(2)))), jax.random.PRNGKey(3), j.zeros(2), j.bool_(False))


@pytest.mark.parametrize('mode', ['mean', 'gaussian', 'zero'])
def test_applied_action_matches_recorded_action_and_density_mode(mode):
    state = carry(); params = j.array([.4, -.2]); pulse = j.array([0., .1, 0., 32., 0.])
    actor = UnusedActor() if mode == 'zero' else Actor()
    step = jax.jit(lambda c: normal_evaluation_interval(Env(), actor, c, params,
                   j.array([-.5, -.7]), None, j.float32(1.2), pulse, mode=mode, horizon_s=10.))
    end, (decision, logs) = step(state)
    expected = j.zeros(2) if mode == 'zero' else params
    if mode != 'gaussian':
        np.testing.assert_array_equal(decision['raw_action'], expected)
        np.testing.assert_array_equal(end.key, state.key)
        assert not bool(decision['density_valid']) and np.isnan(decision['log_prob'])
    else:
        assert bool(decision['density_valid']) and np.isfinite(decision['log_prob'])
        assert not np.array_equal(end.key, state.key)
    np.testing.assert_array_equal(logs['reference'], j.tile(decision['raw_action'], (4, 1)))
    np.testing.assert_array_equal(end.held_reference, decision['raw_action'])
    assert bool(end.handed_over)
    np.testing.assert_array_equal(logs['physical_tick'], [12, 13, 14, -1])
    np.testing.assert_array_equal(logs['task_reward_valid'], logs['active_tick'])
    np.testing.assert_allclose(logs['actual_forward_speed'], 1.2)
    np.testing.assert_array_equal(logs['applied_force_impulse'], j.tile(pulse[-3:], (4, 1)))
    assert np.isnan(decision['physical_value']) and np.isnan(decision['old_task_value'])
    np.testing.assert_allclose(logs['nominal_reference'], j.tile(j.array([1.1, .1]), (4, 1)))
    np.testing.assert_array_equal(logs['filtered_offset'], logs['reference'])
    np.testing.assert_array_equal(logs['reference_clip_channels'], j.zeros((4, 2), bool))
