import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
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

    def test_urdf_preserves_articulated_com_at_multiple_poses(self):
        from sttw_control.isaac_urdf import export_isaac_urdf
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';dest=Path(d)/'urdf'
            build_pendulum_candidate('learning/configs/pendulum_common_asset.json',source)
            contract=export_isaac_urdf(dest,source/'vehicle.xml',preserve_self_collisions=True)
            root=ET.parse(dest/'vehicle.urdf').getroot()
            ext=ET.SubElement(root,'mujoco');ET.SubElement(ext,'compiler',fusestatic='false',discardvisual='true')
            imported=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),{n:(dest/n).read_bytes() for n in contract['meshes']})
            native=load_model(source/'vehicle.xml').model
            a,b=mujoco.MjData(native),mujoco.MjData(imported)
            for hip,knee in [(0,0),(-1.2,2.5),(.3,-.7)]:
                for n,value in [('hip_joint',hip),('knee_joint',knee)]:
                    for m,data in [(native,a),(imported,b)]:
                        j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n);data.qpos[m.jnt_qposadr[j]]=value
                mujoco.mj_forward(native,a);mujoco.mj_forward(imported,b)
                rotation=a.xmat[1].reshape(3,3)
                for row in contract['bodies']:
                    i=mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_BODY,row['name']);j=mujoco.mj_name2id(imported,mujoco.mjtObj.mjOBJ_BODY,row['name'])
                    np.testing.assert_allclose(b.xipos[j],rotation.T@(a.xipos[i]-a.xpos[1]),atol=1e-8)
                    self.assertAlmostEqual(native.body_mass[i],imported.body_mass[j],places=10)
            self.assertTrue(contract['requires_self_collision_filtering'])
            self.assertEqual(contract['link_frame_offsets']['uparm'],[0,0,.11])
            self.assertEqual(len(root.findall('link/collision/geometry/cylinder')),1)
