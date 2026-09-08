"""Portable loading of the original vehicle without modifying source assets."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_XML=ROOT/'mujoco_ros_ws20260313/model/scalebike_scene_matlab.xml'


@dataclass(frozen=True)
class ModelBundle:
    model: object
    identity: dict
    chassis: int
    steer_qpos: int
    steer_dof: int
    front_dof: int
    rear_dof: int
    imu_gyro: int
    portable_xml: str


def load_model(path=DEFAULT_XML):
    path=Path(path)
    source=path.read_bytes()
    root=ET.fromstring(source)
    # Current wheel collisions are ellipsoids. Only unused plugin assets are
    # stripped; refuse rather than silently replacing any active SDF geometry.
    if root.findall('.//geom[@type="sdf"]') or root.findall('.//geom[@mesh="torus"]'):
        raise ValueError('active SDF geometry requires a separately validated backend')
    for ext in root.findall('extension'):
        root.remove(ext)
    asset=root.find('asset')
    for mesh in list(asset.findall('mesh')):
        if mesh.find('plugin') is not None:
            asset.remove(mesh)
    assets={}
    meshdir=root.find('compiler').get('meshdir','')
    for mesh in asset.findall('mesh'):
        filename=mesh.get('file')
        if filename:
            assets[str(Path(meshdir)/filename)]=(path.parent/meshdir/filename).read_bytes()
    portable=ET.tostring(root,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(portable,assets)
    model.opt.disableactuator=2  # group 1 position startup servo disabled
    def joint(name):
        idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if idx<0: raise ValueError('missing joint '+name)
        return idx
    steering=joint('steering_joint')
    gyro=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,'gyro_local')
    chassis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'chasis')
    if min(gyro,chassis)<0: raise ValueError('missing IMU or chassis')
    identity={'source_xml_sha256':hashlib.sha256(source).hexdigest(),
              'portable_xml_sha256':hashlib.sha256(portable.encode()).hexdigest(),
              'assets':{k:hashlib.sha256(v).hexdigest() for k,v in sorted(assets.items())},
              'transform':'remove_unused_torus_assets_disable_startup_position_servo',
              'mujoco_version':mujoco.__version__}
    return ModelBundle(model,identity,chassis,int(model.jnt_qposadr[steering]),
                       int(model.jnt_dofadr[steering]),int(model.jnt_dofadr[joint('frontwheel_joint')]),
                       int(model.jnt_dofadr[joint('rearwheel_joint')]),int(model.sensor_adr[gyro]),portable)


def export_model(output,source=DEFAULT_XML):
    """Create a relocatable host entry with absolute asset references, no copies.

    Generated XML belongs in ignored build/, and still depends on source meshes.
    """
    bundle=load_model(source)
    root=ET.fromstring(bundle.portable_xml)
    compiler=root.find('compiler')
    compiler.set('meshdir',str((Path(source).resolve().parent/compiler.get('meshdir','')).resolve()))
    root.find('option').set('actuatorgroupdisable','1')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write(ET.tostring(root,encoding='unicode'))
    return bundle.identity
