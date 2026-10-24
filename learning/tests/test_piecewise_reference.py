import jax
import jax.numpy as j
import numpy as np
import pytest

from sttw_control.piecewise_reference import (
    piecewise_action, validate_piecewise_schedule, matched_target_families,
)


def test_delay_stage_boundary_and_exact_handback_under_jit():
    parent=j.array([.9,-.8]);targets=j.array([[.1,.2],[.3,.4]])
    run=jax.jit(jax.vmap(lambda tick:piecewise_action(
        parent,targets,j.array([40,80]),tick,8,True,j.array([True,False]))))
    action,active=run(j.array([0,4,8,44,48,84,88,92]))
    np.testing.assert_array_equal(active,[False,False,True,True,True,True,False,False])
    np.testing.assert_array_equal(action[:,0],np.array([.9,.9,.1,.1,.3,.3,.9,.9],np.float32))
    np.testing.assert_array_equal(action[:,1],np.full(8,np.float32(-.8)))


def test_disabled_and_empty_masks_execute_parent():
    parent=j.array([.9,-.8]);targets=j.array([[.1,.2],[.3,.4]])
    for enabled,mask in [(False,[True,True]),(True,[False,False])]:
        action,active=piecewise_action(parent,targets,j.array([40,80]),12,8,enabled,j.array(mask))
        np.testing.assert_array_equal(action,parent)
        assert not active


def test_schedule_validates_positive_aligned_ordered_segments():
    assert validate_piecewise_schedule(.04,[.2,.2],.005)==(8,(40,80))
    for delay,segments in [(-.02,[.2,.2]),(.005,[.2,.2]),(.04,[]),(.04,[0.,.4]),(.04,[.205,.195]),(.04,[float('nan'),.2])]:
        with pytest.raises(ValueError):validate_piecewise_schedule(delay,segments,.005)


def test_equal_budget_and_per_stage_marginals_without_stationary_temporal_pairs():
    static,temporal=matched_target_families(20261024)
    assert static.shape==temporal.shape==(16,2,2)
    np.testing.assert_array_equal(static[:,0],static[:,1])
    assert np.all(np.any(temporal[:,0]!=temporal[:,1],axis=1))
    for stage in [0,1]:
        assert sorted(map(tuple,static[:,stage]))==sorted(map(tuple,temporal[:,stage]))
    np.testing.assert_array_equal(temporal,matched_target_families(20261024)[1])
    assert np.max(np.abs(temporal[:,:,0]))<=.15
    assert np.max(np.abs(temporal[:,:,1]))<=.04
