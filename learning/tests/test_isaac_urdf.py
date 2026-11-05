import tempfile
import unittest
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from sttw_control.isaac_urdf import export_isaac_urdf
from sttw_control.model import load_model


class URDFExportTests(unittest.TestCase):
    def test_physical_tree_and_closed_wheel_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'asset'
            contract = export_isaac_urdf(output)
            root = ET.parse(output/'vehicle.urdf').getroot()
            # Independent MuJoCo URDF reader; preserve fixed links for comparison.
            extension = ET.SubElement(root, 'mujoco')
            ET.SubElement(extension, 'compiler', fusestatic='false', discardvisual='true')
            imported = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'),
                {name: (output/name).read_bytes() for name in contract['meshes']})
            source = load_model().model
            a, b = mujoco.MjData(source), mujoco.MjData(imported)
            mujoco.mj_forward(source, a)
            mujoco.mj_forward(imported, b)
            root_rotation = a.xmat[1].reshape(3, 3)
            for row in contract['bodies']:
                si = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, row['name'])
                ti = mujoco.mj_name2id(imported, mujoco.mjtObj.mjOBJ_BODY, row['name'])
                self.assertGreater(ti, 0)
                np.testing.assert_allclose(b.xpos[ti], root_rotation.T@(a.xpos[si]-a.xpos[1]), atol=1e-10)
                self.assertAlmostEqual(source.body_mass[si], imported.body_mass[ti], places=10)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, imported.body_iquat[ti])
                rotation = rotation.reshape(3, 3)
                np.testing.assert_allclose(rotation@np.diag(imported.body_inertia[ti])@rotation.T,
                                           row['inertia_body'], atol=1e-8, rtol=1e-5)  # URDF reader eigen-decomposition tolerance
                tensor = root.find("link[@name='"+row['name']+"']/inertial/inertia")
                for key, i, j in [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]:
                    self.assertEqual(float(tensor.get(key)), row['inertia_body'][i][j])
            for name, metadata in contract['meshes'].items():
                vertices, faces = [], []
                for line in (output/name).read_text().splitlines():
                    kind, *values = line.split()
                    (vertices if kind == 'v' else faces).append([float(v) if kind == 'v' else int(v) for v in values])
                self.assertEqual(len(vertices), 226)
                edges = Counter(tuple(sorted((f[i], f[(i+1)%3]))) for f in faces for i in range(3))
                self.assertTrue(all(count == 2 for count in edges.values()))
                scaled = np.asarray(vertices)/metadata['source_semiaxes_m']
                np.testing.assert_allclose((scaled*scaled).sum(axis=1), 1., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
