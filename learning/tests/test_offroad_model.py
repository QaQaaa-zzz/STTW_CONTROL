import math
import numpy as np
import unittest
import tempfile
from pathlib import Path
from sttw_control.model import load_model, export_offroad_model


def check_terrain_preserves_vehicle_and_exposed_height(tmp_path):
    base = load_model().model
    ridge = dict(x_m=5., y_m=0., yaw_rad=math.radians(115),
                 length_m=2., radius_m=.05, height_m=.005)
    path = tmp_path / 'ridge.xml'
    export_offroad_model(path, [ridge])
    terrain = load_model(path).model
    assert terrain.ngeom == base.ngeom + 1
    assert terrain.opt.timestep == base.opt.timestep
    for field in ('body_mass', 'body_inertia', 'jnt_axis', 'actuator_gainprm',
                  'actuator_biasprm', 'actuator_ctrlrange'):
        np.testing.assert_array_equal(getattr(base, field), getattr(terrain, field))
    # New world geom follows the floor, before the vehicle's body geoms.
    for field in ('geom_type', 'geom_size', 'geom_friction', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_array_equal(getattr(base, field),
                                      np.delete(getattr(terrain, field), 1, axis=0))
    assert abs(terrain.geom_pos[1, 2] + terrain.geom_size[1, 0] - .005) < 1e-12
    with unittest.TestCase().assertRaises(FileExistsError):
        export_offroad_model(path, [ridge])


def check_invalid_ridge_does_not_write(tmp_path):
    ridge = dict(x_m=0., y_m=0., yaw_rad=0., length_m=2., radius_m=.05, height_m=.06)
    path = tmp_path / 'invalid.xml'
    with unittest.TestCase().assertRaises(ValueError):
        export_offroad_model(path, [ridge])
    assert not path.exists()


class OffroadModelTests(unittest.TestCase):
    def test_vehicle_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            check_terrain_preserves_vehicle_and_exposed_height(Path(directory))

    def test_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            check_invalid_ridge_does_not_write(Path(directory))


if __name__ == "__main__":
    unittest.main()
