"""Unadopted target-composition candidate; caller owns the gate and plant.

Within the SAME active window, retain a fixed bias relative to the parent's
current target rather than freezing the whole selected target. This has no
validated disturbance-tolerance benefit and does not change alpha eligibility.
"""
import jax.numpy as j

from .path_command_policy import map_latent
from .reference_prefix import inverse_offset_target


def latch_parent_target(previous, original, detected, config):
    """Caller supplies the existing single-shot detected flag, even for no-op."""
    return j.where(detected, map_latent(original, config), previous)


def following_reference_action(original, fixed_latent, mask, parent_anchor, active, config):
    """Pure target helper. Anchors remain in physical reference coordinates.

    The caller must consume engineering_fault, maintain the existing 80-tick
    eligibility window, and forbid use after task-domain exit. It must obtain
    original from THIS arm's current Actor/observation, not another trajectory.
    """
    parent = map_latent(original, config)
    locked = map_latent(fixed_latent, config)
    requested = locked + mask * (parent - parent_anchor)
    proposed, realized, clipped = inverse_offset_target(requested, config)
    # Preserve the exact old latent if its selected parent target did not move.
    proposed = j.where(parent == parent_anchor, fixed_latent, proposed)
    fault = (~j.all(j.isfinite(original)) | ~j.all(j.isfinite(fixed_latent))
             | ~j.all(j.isfinite(parent_anchor)))
    enabled = active & mask & ~fault
    action = j.where(enabled, proposed, original)
    return action, dict(engineering_fault=fault, parent_target=parent,
                        requested_follow_target=requested,
                        realized_follow_target=realized,
                        target_clip_channels=enabled & clipped)
