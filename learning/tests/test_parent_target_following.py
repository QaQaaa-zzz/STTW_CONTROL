"""Target-composition identities only; no new closed-loop result is implied."""
import jax
import jax.numpy as j
import numpy as np

from sttw_control.path_command_policy import map_latent
from sttw_control.reference_prefix import inverse_offset_target
from sttw_control.parent_target_following import latch_parent_target, following_reference_action


C={'residual_interface': {'speed_negative_scale_m_s':1.,'speed_positive_scale_m_s':.25,'steer_scale_rad':.2}}


def latent(target):
    return inverse_offset_target(j.array(target),C)[0]


def test_same_initial_target_and_constant_parent_exactly_preserve_locked_action():
    original=j.array([-.8,.4]);fixed=latent([-.3,.03]);mask=j.array([True,False])
    anchor=latch_parent_target(j.zeros(2),original,j.bool_(True),C)
    for _ in range(3):
        anchor=latch_parent_target(anchor,j.array([-.1,.5]),j.bool_(False),C)
        action,log=jax.jit(following_reference_action)(original,fixed,mask,anchor,j.bool_(True),C)
        np.testing.assert_array_equal(action,j.where(mask,fixed,original))
        assert not log['engineering_fault']


def test_following_keeps_parent_target_increment_without_modifying_other_channel():
    old=latent([-.3,.02]);current=latent([-.2,.09]);fixed=latent([-.15,.01]);mask=j.array([True,False])
    anchor=map_latent(old,C)
    action,log=following_reference_action(current,fixed,mask,anchor,j.bool_(True),C)
    np.testing.assert_allclose(map_latent(action,C)[0]-map_latent(fixed,C)[0],
                               map_latent(current,C)[0]-anchor[0],atol=1e-7)
    assert action[1]==current[1] and not log['target_clip_channels'].any()


def test_original_asymmetric_speed_and_symmetric_steer_limits_are_retained():
    for parent,anchor,fixed,channel,sign,bound in (
        ([.24,0.],[-.9,0.],[.2,0.],0,1,.25),
        ([-.9,0.],[.24,0.],[-.8,0.],0,-1,1.),
        ([0.,.19],[0.,-.19],[0.,.1],1,1,.2),
    ):
        action,log=following_reference_action(latent(parent),latent(fixed),j.array([channel==0,channel==1]),
                                             j.array(anchor),j.bool_(True),C)
        assert log['target_clip_channels'][channel]
        np.testing.assert_allclose(map_latent(action,C)[channel],sign*bound*(1-1e-6),atol=1e-7)


def test_inactive_or_empty_mask_returns_original_and_fault_never_overrides():
    original=j.array([.2,.4]);fixed=latent([-.3,.03]);anchor=j.zeros(2)
    for active,mask in ((False,[True,False]),(True,[False,False])):
        action,_=following_reference_action(original,fixed,j.array(mask),anchor,j.bool_(active),C)
        np.testing.assert_array_equal(action,original)
    action,log=following_reference_action(original,fixed,j.array([True,False]),j.array([j.nan,0.]),j.bool_(True),C)
    assert log['engineering_fault']
    np.testing.assert_array_equal(action,original)
