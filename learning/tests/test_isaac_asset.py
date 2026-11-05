import tempfile
import unittest
from pathlib import Path
import mujoco
import numpy as np
from sttw_control.isaac_asset import export_isaac_asset
from sttw_control.model import load_model


class IsaacExportTests(unittest.TestCase):
    def test_export_retains_physical_tree_and_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'asset'
            contract = export_isaac_asset(output)
            source = load_model().model
            exported = mujoco.MjModel.from_xml_path(str(output/'vehicle.xml'))
            for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                          'jnt_axis', 'jnt_range', 'jnt_stiffness', 'dof_damping'):
                np.testing.assert_allclose(getattr(source, field), getattr(exported, field), atol=1e-12)
            a, b = mujoco.MjData(source), mujoco.MjData(exported)
            mujoco.mj_forward(source, a)
            mujoco.mj_forward(exported, b)
            np.testing.assert_allclose(a.xpos, b.xpos, atol=1e-12)
            np.testing.assert_allclose(a.xmat, b.xmat, atol=1e-12)
            self.assertEqual(contract['actor_collision_filter'], 1)
            self.assertEqual(len(contract['vehicle_geoms']), 6)
            self.assertEqual(exported.ngeom, 6)
            self.assertEqual(len(contract['joints']), 4)
            self.assertFalse(contract['actuators'][0]['enabled'])
            with self.assertRaises(FileExistsError):
                export_isaac_asset(output)


if __name__ == '__main__':
    unittest.main()
