"""Export an explicit Isaac Gym MJCF input and its MuJoCo source contract.

Run in the MuJoCo environment. Isaac Gym consumes the resulting files in its
own Python environment; neither environment's dependencies are changed.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .model import load_model, DEFAULT_XML


def export_isaac_asset(output, source=DEFAULT_XML):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    bundle = load_model(source)
    model = bundle.model
    root = ET.fromstring(bundle.portable_xml)
    sequence = root.find('compiler').get('eulerseq', 'xyz')
    radians = root.find('compiler').get('angle', 'degree') == 'radian'
    for node in root.iter():
        if 'euler' in node.attrib:
            angles = np.fromstring(node.attrib.pop('euler'), sep=' ')
            quat = np.zeros(4)
            mujoco.mju_euler2Quat(quat, angles if radians else np.deg2rad(angles), sequence)
            node.set('quat', ' '.join(map(str, quat)))
    root.find('compiler').set('eulerseq', 'xyz')
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ('camera', 'light') or (child.tag == 'geom' and child.get('type') == 'mesh'):
                parent.remove(child)
    for tag in ('asset', 'sensor', 'actuator', 'keyframe', 'contact'):
        for node in root.findall(tag):
            root.remove(node)
    world = root.find('worldbody')
    for node in list(world):
        if node.tag == 'geom':
            world.remove(node)
    for node in root.findall('.//geom'):
        node.attrib.pop('material', None)
    active = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
    vehicle = [int(i) for i in active if model.geom_bodyid[i] != 0]
    if any(model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH for i in vehicle):
        raise ValueError('active mesh cannot be discarded by the primitive MJCF exporter')
    if any((model.geom_contype[i] & model.geom_conaffinity[j]) or
           (model.geom_contype[j] & model.geom_conaffinity[i])
           for i in vehicle for j in vehicle):
        raise ValueError('actor filter=1 requires all vehicle self-collision pairs disabled at source')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    def name(kind, i):
        return mujoco.mj_id2name(model, kind, i)
    bodies = [dict(name=name(mujoco.mjtObj.mjOBJ_BODY, i),
                   position=data.xpos[i].tolist(), quat_wxyz=data.xquat[i].tolist(),
                   mass=float(model.body_mass[i]), inertia=model.body_inertia[i].tolist(),
                   inertial_pos=model.body_ipos[i].tolist(), inertial_quat_wxyz=model.body_iquat[i].tolist())
              for i in range(1, model.nbody)]
    for body in bodies:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(body['inertial_quat_wxyz']))
        rotation = rotation.reshape(3, 3)
        body['inertia_body'] = (rotation @ np.diag(body['inertia']) @ rotation.T).tolist()
    joints = [dict(name=name(mujoco.mjtObj.mjOBJ_JOINT, i), axis=model.jnt_axis[i].tolist(),
                   limited=bool(model.jnt_limited[i]), range=model.jnt_range[i].tolist(),
                   spring_reference=float(model.qpos_spring[model.jnt_qposadr[i]]),
                   stiffness=float(model.jnt_stiffness[i]), damping=float(model.dof_damping[model.jnt_dofadr[i]]))
              for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    actuators = [dict(name=name(mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                      joint=name(mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])),
                      enabled=not bool(model.opt.disableactuator & (1 << int(model.actuator_group[i]))),
                      gain=model.actuator_gainprm[i].tolist(), bias=model.actuator_biasprm[i].tolist(),
                      gear=model.actuator_gear[i].tolist(),
                      control_limited=bool(model.actuator_ctrllimited[i]),
                      force_limited=bool(model.actuator_forcelimited[i]),
                      ctrlrange=model.actuator_ctrlrange[i].tolist(),
                      forcerange=model.actuator_forcerange[i].tolist()) for i in range(model.nu)]
    xml = ET.tostring(root, encoding='unicode')
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'gyro_local')
    gyro_site = int(model.sensor_objid[gyro_id])
    contract = dict(source=bundle.identity, exported_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                    imu_site_quat_wxyz=model.site_quat[gyro_site].tolist(),
                    timestep_s=float(model.opt.timestep), bodies=bodies, joints=joints, actuators=actuators,
                    actor_collision_filter=1,
                    root_translation=root.find('worldbody/body').get('pos', '0 0 0'),
                    vehicle_geoms=[dict(body=name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])),
                                        type=int(model.geom_type[i]), size=model.geom_size[i].tolist(),
                                        friction=model.geom_friction[i].tolist()) for i in vehicle],
                    differences=['Ellipsoids are tessellated by Isaac; analytic MuJoCo contact is not preserved.',
                                 'PhysX solver/contact offsets and friction semantics require explicit calibration.',
                                 'Actuators/sensors removed from import; implement their source contracts in runtime.',
                                 'Visual-only meshes omitted; collision primitives and body inertias retained.',
                                 'Restore rigid-body AND DOF states after prepare_sim; it alters states at public time zero.'])
    output.mkdir(parents=True)
    (output/'vehicle.xml').write_text(xml)
    (output/'contract.json').write_text(json.dumps(contract, indent=2))
    return contract
