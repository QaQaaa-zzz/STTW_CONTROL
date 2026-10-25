import numpy as np
import jax.numpy as j
from sttw_control.geometric_path import Path
from sttw_control.geometric_reacquisition import geometric_target,join_prefix_suffix
from sttw_control.reference_prefix import inverse_offset_target
from sttw_control.path_command_policy import map_latent
import pytest

CONFIG=dict(follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=1.5,lookahead_min_m=.8,lookahead_max_m=1.8,preview_body_x_min_m=.05,wheelbase_m_expected=.408,caster_deg_expected=25.,steer_reference_max_rad=.5),residual_interface=dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.2))


def path():
    s=j.linspace(0.,50.,101)
    return Path(s,j.stack([s,j.zeros_like(s)],axis=1),j.zeros_like(s),j.zeros_like(s),j.float32(0),j.float32(20))


def test_nearest_forward_query_is_chosen_without_touching_original_path():
    p=path();before=np.asarray(p.xy).copy();pose=j.array([0.,2.,j.pi/3])
    target,record=geometric_target(p,j.float32(0),pose,j.float32(1.5),j.zeros(2),j.int32(2),CONFIG)
    assert float(record['lookahead_m'])==4. and bool(record['point_available']) and not bool(record['heading_fallback'])
    assert float(record['body_point'][0])>.05 and target[1]<0
    np.testing.assert_array_equal(p.xy,before)


def test_fixed6_and_adaptive_share_bounded_heading_fallback_when_points_behind():
    pose=j.array([0.,0.,j.pi]);p=path()
    outputs=[]
    for mode in [1,2]:
        target,record=geometric_target(p,j.float32(0),pose,j.float32(1.5),j.array([1.5,.03]),j.int32(mode),CONFIG)
        assert record['heading_fallback'] and not record['point_available']
        assert abs(float(record['desired_steer_rad']))==pytest.approx(.18)
        outputs.append(target)
    np.testing.assert_allclose(*outputs,atol=1e-7)


def test_zero_and_original_asymmetric_mapping_keep_real_authority():
    args=(path(),j.float32(0),j.array([0.,2.,j.pi/3]),j.float32(1.5),j.array([1.5,.5]))
    target,record=geometric_target(*args,j.int32(0),CONFIG)
    np.testing.assert_array_equal(target,[0.,0.]);assert not record['override']
    target,record=geometric_target(*args,j.int32(2),CONFIG)
    latent,mapped,clipped=inverse_offset_target(target,CONFIG)
    np.testing.assert_allclose(map_latent(latent,CONFIG),mapped,atol=1e-6)
    assert abs(float(mapped[1]))<=.2 and bool(clipped[1]) and float(mapped[0])==0.
    _,record=geometric_target(path(),j.float32(0),j.array([j.nan,0.,0.]),j.float32(1.5),j.zeros(2),j.int32(2),CONFIG)
    assert record['fault']


def test_prefix_and_suffix_keep_clock_and_no_fake_diagnostic_prefix():
    p=dict(active_tick=np.ones(2000,bool),physical_tick=np.arange(2000),actual_xy=np.zeros((2000,2)),prefix_only=np.ones(2000))
    s=dict(active_tick=np.array([True,True,False]),physical_tick=np.array([2000,2001,-1]),actual_xy=np.ones((3,2)),new_diagnostic=np.ones(3))
    joined=join_prefix_suffix(p,s)
    np.testing.assert_array_equal(joined['physical_tick'][joined['active_tick']],np.arange(2002))
    assert joined['suffix_active'].sum()==2 and len(joined['active_tick'])==2003
    assert 'prefix_only' not in joined and 'new_diagnostic' not in joined
    np.testing.assert_array_equal(joined['actual_xy'][:2000],p['actual_xy'])
    with pytest.raises(ValueError):join_prefix_suffix(p,dict(s,physical_tick=np.array([0,1,-1])))
