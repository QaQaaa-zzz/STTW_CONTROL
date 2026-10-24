"""Causal carry contract for parent-relative target following."""
import jax
import jax.numpy as j
import numpy as np
import pytest

from sttw_control.path_command_policy import map_latent
from sttw_control.causal_reference_baseline import validate_full_task_origin
from sttw_control.following_causal_reference import (
    initial_following_state, following_causal_reference_action, validate_following_origin,
)


C={'residual_interface': {'speed_negative_scale_m_s':1.,'speed_positive_scale_m_s':.25,'steer_scale_rad':.2}}


def observation(signal=-.03,alpha=0.):
    return (j.zeros(351).at[334:336].set(1.).at[318].set(-signal)
            .at[336].set(alpha).at[340].set(-.9/.6).at[345].set(1.5/3.))


run=jax.jit(following_causal_reference_action)


def test_event_latches_current_parent_before_composition_and_starts_at_locked_action():
    original=j.array([-.8,.4])
    action,state,log=run(original,observation(),j.array([-.4,.02]),j.int32(100),initial_following_state(),C)
    assert log['detected'] and log['override_active']
    np.testing.assert_array_equal(state.parent_anchor,map_latent(original,C))
    np.testing.assert_array_equal(action,log['locked_action'])
    assert state.fired and state.start_tick==100


def test_later_parent_variation_uses_same_arm_and_never_reanchors():
    original=j.array([-.8,.4])
    _,state,_=run(original,observation(),j.array([-.4,.02]),j.int32(100),initial_following_state(),C)
    current=j.array([-.5,.8])
    action,next_state,log=run(current,observation(.04),j.array([-.3,.01]),j.int32(104),state,C)
    np.testing.assert_array_equal(next_state.parent_anchor,state.parent_anchor)
    assert not log['detected'] and log['override_active']
    np.testing.assert_allclose(map_latent(action,C)[0]-map_latent(state.fixed_latent,C)[0],
                               map_latent(current,C)[0]-state.parent_anchor[0],atol=1e-7)
    assert action[1]==current[1]
    returned,after,log=run(current,observation(),j.array([-.3,.01]),j.int32(180),next_state,C)
    np.testing.assert_array_equal(returned,current)
    np.testing.assert_array_equal(after.parent_anchor,state.parent_anchor)
    assert not log['override_active']


def test_first_no_candidate_event_is_still_latched_and_cannot_wait_for_favorable_event():
    first=j.array([-.8,.4]);later=j.array([-.2,.8])
    action,state,log=run(first,observation(.03),j.zeros(2),j.int32(100),initial_following_state(),C)
    assert state.fired and log['detected'] and not log['override_active']
    np.testing.assert_array_equal(action,first)
    action,after,log=run(later,observation(-.03),j.zeros(2),j.int32(104),state,C)
    np.testing.assert_array_equal(action,later)
    np.testing.assert_array_equal(after.parent_anchor,map_latent(first,C))
    assert not log['detected'] and not log['override_active']


def test_initial_origin_validates_extra_carry_and_helper_fault_is_exposed():
    state=initial_following_state()
    validate_full_task_origin(0,700,700,state)
    validate_following_origin(state)
    with pytest.raises(ValueError):validate_following_origin(state._replace(parent_anchor=j.array([.1,0.])))
    bad=state._replace(parent_anchor=j.array([j.nan,0.]))
    action,_,log=run(j.array([-.8,.4]),observation(0.),j.zeros(2),j.int32(4),bad,C)
    assert log['engineering_fault'] and not log['override_active']
    np.testing.assert_array_equal(action,j.array([-.8,.4]))


def test_declared_variant_selects_matching_action_and_initial_carry():
    from sttw_control.reflected_causal_reference import select_reference_rule, initial_reference_state
    assert select_reference_rule('parent_following_v1') is following_causal_reference_action
    validate_following_origin(initial_reference_state('parent_following_v1'))
    assert not hasattr(initial_reference_state('baseline'),'parent_anchor')
    with pytest.raises(ValueError):initial_reference_state('unknown')
