"""State conversion and nominal constants for the frozen PCRP warning replay."""
import json
from pathlib import Path

import numpy as np


PHI_C = 0.70
B_NOM = np.deg2rad(6146.05)  # rad/s^2
DT = 0.005
CTRL_CFG = json.loads((Path(__file__).resolve().parents[2] / "tests/fixtures/paper_preference/controller.json").read_text())


def rotate_vec(q, v):
    t = 2.0 * np.cross(q[..., :3], v)
    return v + q[..., 3:] * t + np.cross(q[..., :3], t)


def euler_roll_rate(chassis_body_state):
    """Exact Euler roll rate (rad/s) matching analyze_roll_contacts.roll_state."""
    q = chassis_body_state[..., 3:7]
    x, y, z, w = np.moveaxis(q, -1, 0)
    raw = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    inv = q.copy()
    inv[..., :3] *= -1.0
    omega = rotate_vec(inv, chassis_body_state[..., 10:13])
    return -(omega[..., 0] + np.sin(raw) * np.tan(pitch) * omega[..., 1] + np.cos(raw) * np.tan(pitch) * omega[..., 2])
