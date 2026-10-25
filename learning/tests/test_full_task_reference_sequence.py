import numpy as np
import jax.numpy as j
from sttw_control.full_task_reference_sequence import candidate_library, progress_knots, sequence_target, sequence_action
from sttw_control.path_command_policy import map_latent

CONFIG={'path':{'straight_prefix_m':3.5},'residual_interface':dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.2)}


def test_targets_hit_knots_return_zero_outside_and_interpolate_smoothly():
    positions=progress_knots(3.5,5.5)
    np.testing.assert_allclose(positions,[2.,3.5,4.5,5.5,7.])
    targets=j.array([[0.,0.],[-.5,.1],[.2,-.1],[-.2,.03],[0.,0.]])
    for k,p in enumerate(positions):np.testing.assert_allclose(sequence_target(p,positions,targets),targets[k],atol=1e-7)
    for p in [-10.,20.]:np.testing.assert_array_equal(sequence_target(p,positions,targets),[0.,0.])
    np.testing.assert_allclose(sequence_target(4.,positions,targets),[-.15,0.],atol=1e-7)
    left=np.asarray(sequence_target(4.5-1e-4,positions,targets))
    right=np.asarray(sequence_target(4.5+1e-4,positions,targets))
    assert np.linalg.norm(left-right)<1e-6


def test_full_domain_targets_use_real_mapping_and_only_stop_or_zero_disables():
    positions=progress_knots(3.5,5.5);targets=j.array([[0.,0.],[-.5,.1],[.2,-.1],[-.2,.03],[0.,0.]])
    # No Actor-domain flag exists in this API: a legal reference process is not an old Actor.
    action,record=sequence_action(j.float32(4.5),positions,targets,CONFIG,j.bool_(False),j.bool_(False))
    np.testing.assert_allclose(map_latent(action,CONFIG),[.2,-.1],atol=1e-6)
    assert record['reference_valid'] and not record['fault']
    for stop,zero in [(True,False),(False,True)]:
        action,record=sequence_action(j.float32(4.5),positions,targets,CONFIG,j.bool_(stop),j.bool_(zero))
        np.testing.assert_array_equal(action,[0.,0.]);assert not record['reference_valid']
    action,record=sequence_action(j.float32(float('nan')),positions,targets,CONFIG,j.bool_(False),j.bool_(False))
    assert record['fault'];assert np.isfinite(action).all()


def test_frozen_library_contains_zero_and_matched_window_constants():
    first=candidate_library(CONFIG);second=candidate_library(CONFIG)
    np.testing.assert_array_equal(first['targets'],second['targets'])
    a=first['targets'];assert a.shape==(37,5,2) and len(set(first['names']))==37
    np.testing.assert_array_equal(a[:,[0,-1]],0.)
    assert ((a[:,:,0]>=-1)&(a[:,:,0]<=.25)).all() and (abs(a[:,:,1])<=.2).all()
    np.testing.assert_array_equal(a[32],0.)
    for row in a[33:]:np.testing.assert_array_equal(row[1],row[2]);np.testing.assert_array_equal(row[2],row[3])
    # Actual nominal targets remain distinct before filtering; later dose audit cannot discard failures.
    assert len({r.tobytes() for r in a})==37
