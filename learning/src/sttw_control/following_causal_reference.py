"""Causal carry adapter for the unvalidated parent-relative target candidate."""
from typing import NamedTuple

import jax.numpy as j

from .causal_reference_baseline import initial_gate
from .reflected_causal_reference import reflected_reference_action
from .parent_target_following import latch_parent_target, following_reference_action


class FollowingState(NamedTuple):
    gate: object
    parent_anchor: object

    @property
    def fired(self): return self.gate.fired

    @property
    def start_tick(self): return self.gate.start_tick

    @property
    def target_offset(self): return self.gate.target_offset

    @property
    def fixed_latent(self): return self.gate.fixed_latent

    @property
    def mask(self): return self.gate.mask


def initial_following_state():
    return FollowingState(initial_gate(), j.zeros(2))


def validate_following_origin(state):
    if not bool(j.all(j.isfinite(state.parent_anchor))) or bool(j.any(state.parent_anchor != 0)):
        raise ValueError('parent target anchor must initialize once at complete task origin')


def following_causal_reference_action(original, observation, current_offset, tick, state, config):
    locked, gate, receipt = reflected_reference_action(
        original, observation, current_offset, tick, state.gate, config)
    anchor = latch_parent_target(state.parent_anchor, original, receipt['detected'], config)
    action, detail = following_reference_action(
        original, gate.fixed_latent, gate.mask, anchor, receipt['override_active'], config)
    receipt = dict(receipt, engineering_fault=receipt['engineering_fault'] | detail['engineering_fault'],
                   override_active=receipt['override_active'] & ~detail['engineering_fault'],
                   locked_action=locked, parent_target_anchor=anchor,
                   parent_target_current=detail['parent_target'],
                   following_requested_target=detail['requested_follow_target'],
                   following_realized_target=detail['realized_follow_target'],
                   following_target_clipped=detail['target_clip_channels'])
    return action, FollowingState(gate, anchor), receipt
