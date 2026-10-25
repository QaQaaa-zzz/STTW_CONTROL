import copy
import jax
import jax.numpy as j
import numpy as np
from sttw_control.anticipatory_reference import progress_envelope,candidate_latent,floor_config
from sttw_control.path_command_policy import map_latent,correction_tick


def config():
    return dict(timing=dict(lower_dt_s=.005),path=dict(straight_prefix_m=3.),
        residual_interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,
            steer_scale_rad=.2,filter_tau_s=[.1,.08],correction_speed_down_m_s2=1.,
            correction_speed_up_m_s2=.5,correction_steer_rate_rad_s=.8,
            speed_reference_bounds_m_s=[1.,3.5],steer_reference_bound_rad=.65))


def test_progress_window_has_smooth_bounded_start_plateau_and_release():
    s=j.array([0.,1.5,2.25,3.,5.,6.,6.75,7.5,10.])
    values=jax.vmap(lambda x:progress_envelope(x,3.,6.))(s)
    np.testing.assert_allclose(values,[0,0,.5,1,1,1,.5,0,0],atol=1e-6)
    derivative=jax.vmap(jax.grad(lambda x:progress_envelope(x,3.,6.)))(j.array([1.5,3.,6.,7.5]))
    np.testing.assert_allclose(derivative,np.zeros(4),atol=1e-6)


def test_candidate_preserves_original_action_when_gate_off_or_profile_is_base():
    c=config();original=j.array([-8.,6.]);nominal=j.array([1.5,-.4])
    for profile in [0,1,2,3]:
        result,gate,_=candidate_latent(original,nominal,j.float32(1.),j.float32(6.),j.int32(profile),c)
        np.testing.assert_array_equal(result,original)
        assert float(gate)==0
    result,_,_=candidate_latent(original,nominal,j.float32(4.),j.float32(6.),j.int32(0),c)
    np.testing.assert_array_equal(result,original)


def test_single_channel_and_joint_targets_use_original_bounded_interface():
    c=config();original=j.array([-.4,.2]);nominal=j.array([1.5,-.4])
    for profile in [1,2,3]:
        action,gate,target=jax.jit(lambda p:candidate_latent(original,nominal,j.float32(4.),j.float32(6.),p,c))(j.int32(profile))
        realized=np.asarray(map_latent(action,c));base=np.asarray(map_latent(original,c))
        np.testing.assert_allclose(realized,[.24 if profile in (1,3) else base[0],
                                          .06 if profile in (2,3) else base[1]],atol=1e-6)
        np.testing.assert_allclose(realized,target,atol=1e-6)
        if profile==1:assert float(action[1])==float(original[1])
        if profile==2:assert float(action[0])==float(original[0])
        assert float(gate)==1 and np.isfinite(np.asarray(action)).all()


def test_per_lane_floor_matches_original_clipping_and_does_not_mutate_config():
    c=config();saved=copy.deepcopy(c)
    def apply(bound):
        cfg=floor_config(c,bound)
        return correction_tick(j.array([-.2,.03]),j.array([-.2,.03]),
                               j.array([-1.,.2]),j.array([1.2,.1]),cfg)
    actual=jax.jit(jax.vmap(apply))(j.array([1.,1.5]))
    reference=apply(1.5)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(reference)):
        np.testing.assert_allclose(np.asarray(a)[1],np.asarray(b),atol=1e-7)
    assert float(actual[2][1,0])==1.5 and bool(actual[3]['reference_clip_channels'][1,0])
    assert c==saved


def test_early_support_releases_at_geometric_midturn_and_preserves_steering():
    from sttw_control.anticipatory_reference import early_support_envelope,speed_support_latent
    c=config();original=j.array([-.5,.7]);nominal=j.array([1.2,.1])
    values=jax.vmap(lambda s:early_support_envelope(s,3.,7.))(j.array([1.5,2.25,3.,4.,5.,6.,7.]))
    np.testing.assert_allclose(values,[0,.5,1,.5,0,0,0],atol=1e-6)
    for s in [1.5,3.,5.]:assert abs(float(jax.grad(lambda p:early_support_envelope(p,3.,7.))(j.float32(s))))<1e-6
    late,gate,_=speed_support_latent(original,nominal,j.float32(5.),j.float32(7.),j.int32(6),c)
    np.testing.assert_array_equal(late,original);assert float(gate)==0
    early,gate,target=speed_support_latent(original,nominal,j.float32(3.),j.float32(7.),j.int32(6),c)
    assert early[1]==original[1] and float(gate)==1
    np.testing.assert_allclose(map_latent(early,c)[0],.25,atol=1e-6)
    dispatched,_,_=candidate_latent(original,nominal,j.float32(3.),j.float32(7.),j.int32(6),c)
    np.testing.assert_allclose(map_latent(dispatched,c)[0],.25,atol=1e-6)
    assert dispatched[1]==original[1]
    for z in [j.array([30.,.7]),j.array([-.5,.7])]:
        result,_,_=speed_support_latent(z,nominal,j.float32(10.),j.float32(7.),j.int32(7),c)
        assert result[1]==z[1] and map_latent(result,c)[0]>=map_latent(z,c)[0]
    _,offset,_,_=correction_tick(j.zeros(2),j.zeros(2),early,nominal,c)
    assert offset[0]<=.5*.005+1e-7
    # Existing profiles must remain bitwise unchanged when not supporting.
    for profile in [0,4,5]:
        result,_,_=candidate_latent(original,nominal,j.float32(3.),j.float32(7.),j.int32(profile),c)
        np.testing.assert_array_equal(result,original)
