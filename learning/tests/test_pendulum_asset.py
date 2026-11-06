import tempfile
from pathlib import Path
import unittest
import mujoco
import numpy as np
from sttw_control.model import load_model
from sttw_control.pendulum_asset import build_pendulum_candidate

class PendulumAssetTest(unittest.TestCase):
    def test_preserves_chassis_and_native_knee_pivot(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'asset'
            report=build_pendulum_candidate('learning/configs/pendulum_common_asset.json',output)
            original=load_model().model;candidate=load_model(output/'vehicle.xml').model
            for name in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat'):
                np.testing.assert_allclose(getattr(candidate,name)[:original.nbody],getattr(original,name),atol=1e-12)
            for name in ('jnt_axis','jnt_pos','jnt_range'):
                np.testing.assert_array_equal(getattr(candidate,name)[:original.njnt],getattr(original,name))
            for name in ('actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange'):
                np.testing.assert_array_equal(getattr(candidate,name)[:original.nu],getattr(original,name))
            knee=mujoco.mj_name2id(candidate,mujoco.mjtObj.mjOBJ_JOINT,'knee_joint')
            np.testing.assert_array_equal(candidate.jnt_pos[knee],[0,0,.11])
            self.assertAlmostEqual(report['candidate_mass_kg']-report['base_mass_kg'],3.49)
