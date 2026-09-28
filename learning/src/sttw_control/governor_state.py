"""Prediction boundary: only present closed-loop state, never a schedule."""
from flax import struct

@struct.dataclass
class GovernorState:
    current_reference: object
    last_goal: object
    reserve_ticks: object
    recovery_gain: object
    recovery_hold_ticks: object
    mode: object  # TRACK=0 CONFLICT=1 RECOVER=2 EMERGENCY=3

@struct.dataclass
class Snapshot:
    data: object
    controller: object
    actuator: object
    physical_tick: object
    raw: object
    reference_pose: object
    yaw_wrapped: object
    yaw_unwrapped: object
    governor: GovernorState
    failed: object

@struct.dataclass
class Candidate:
    goal: object
    bypass: object
    valid: object
