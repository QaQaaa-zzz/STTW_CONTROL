"""Regression cases for physically different ways to cross a trench."""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

cli = Path(__file__).parents[1] / 'cli' / 'audit_playground_jump.py'
spec = importlib.util.spec_from_file_location('jump_audit', cli)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('x0,carry_load,dips_in_gap,strict,rear,solid', [
    (-0.30, False, False, True, True, True),
    (-0.08, False, False, False, True, True),
    (-0.08, False, True, False, False, True),
    (-0.08, True, False, False, False, False),
])
def test_measured_launch_and_loaded_landing(tmp_path, x0, carry_load, dips_in_gap, strict, rear, solid):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><freejoint/>
      <geom name="frontwheel_collision" type="ellipsoid" pos=".14 0 0" size=".07 .02 .07"/>
      <geom name="rearwheel_collision" type="ellipsoid" pos="-.14 0 0" size=".07 .02 .07"/>
    </body></worldbody></mujoco>""")
    t = np.arange(100) * .005
    poses = np.tile(model.qpos0, (len(t), 1))
    poses[:, 0] = x0 + 2.5 * t
    poses[:, 2] = .07 + .24 * np.sin(np.pi * np.clip((t - .025) / .4, 0, 1))
    if dips_in_gap:
        over_lip = (t >= .055) & (t <= .07)
        poses[over_lip, 3] = np.cos(.5)
        poses[over_lip, 5] = np.sin(.5)
    forces = np.zeros((len(t), 5, 4))
    forces[:6, :, 1] = 20
    forces[85:, :, 0] = 20
    forces[90:, :, 1] = 20
    if carry_load:
        forces[:, :, 1] = 20
    flags = np.zeros((len(t), 3), bool)
    flags[85:, 2] = True
    path = tmp_path / 'trajectory.npz'
    np.savez(path, qpos=poses, trace=t[:, None], contact_substep=forces,
             wheel_contact_flags=flags, contact_physics_dt_s=.001)
    events, landings = audit.measure(model, path, dict(front_x_m=0., width_m=.45, far_bank_height_m=0.))
    assert any(e['bank_to_bank_flight'] for e in events) == strict
    assert any(e['rear_supported_launch'] for e in events) == rear
    assert any(e['solid_bank_rear_launch'] for e in events) == solid
    assert landings[0] == t[85 if carry_load else 90]
