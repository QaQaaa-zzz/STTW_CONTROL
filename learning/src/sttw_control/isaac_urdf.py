"""URDF conversion of the compiled STTW physical tree for Isaac Gym.

Mass/COM/full inertia and zero joint poses come from MuJoCo, not visual meshes.
Ellipsoids use an explicitly recorded convex OBJ approximation (226 vertices).
"""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .isaac_asset import export_isaac_asset
from .model import load_model, DEFAULT_XML


def _rpy(quat):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    r = rotation.reshape(3, 3)
    return [math.atan2(r[2, 1], r[2, 2]), math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])),
            math.atan2(r[1, 0], r[0, 0])]


def _values(values):
    return ' '.join(format(float(x), '.17g') for x in values)


def export_isaac_urdf(output, source=DEFAULT_XML, wheel_vertices=226, *, preserve_self_collisions=False, wheel_sectors=0):
    if not 6 <= wheel_vertices <= 226:
        raise ValueError('wheel_vertices must be between 6 and 226')
    output = Path(output)
    contract = export_isaac_asset(output, source, preserve_self_collisions=preserve_self_collisions)
    model = load_model(source).model
    root = ET.Element('robot', name='sttw_source_contract')
    bodies = {b['name']: b for b in contract['bodies']}
    meshes = {}
    # URDF child links are based at their hinge. Shift COM/geometries and
    # descendant origins together, preserving the physical rigid-body tree.
    offsets = np.zeros((model.nbody, 3))
    for i in range(1, model.nbody):
        ids = list(range(model.body_jntadr[i], model.body_jntadr[i]+model.body_jntnum[i]))
        if model.body_parentid[i] == 0:
            continue
        if len(ids)>1 or (ids and model.jnt_type[ids[0]] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('converter requires one hinge or fixed joint per body')
        if ids:
            offsets[i] = model.jnt_pos[ids[0]]

    def name(kind, i):
        return mujoco.mj_id2name(model, kind, i)
    for i in range(1, model.nbody):
        body_name = name(mujoco.mjtObj.mjOBJ_BODY, i)
        record = bodies[body_name]
        link = ET.SubElement(root, 'link', name=body_name)
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=_values(np.asarray(record['inertial_pos'])-offsets[i]), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=str(record['mass']))
        inertia = record['inertia_body']
        ET.SubElement(inertial, 'inertia', **{key: str(inertia[a][b]) for key, a, b in
            [('ixx', 0, 0), ('iyy', 1, 1), ('izz', 2, 2), ('ixy', 0, 1), ('ixz', 0, 2), ('iyz', 1, 2)]})
        for g in range(model.body_geomadr[i], model.body_geomadr[i]+model.body_geomnum[i]):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            kind = model.geom_type[g]
            size = model.geom_size[g]
            if kind not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER):
                raise ValueError('source collision geometry is outside this STTW converter')
            mesh_name = 'ellipsoid_'+('_'.join(format(x, '.6g') for x in size))+'.obj'
            if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID and mesh_name not in meshes:
                vertices = [[0., 0., size[2]]]
                for polar in range(1, 8):
                    theta = math.pi*polar/8
                    for azimuth in range(32):
                        phi = 2*math.pi*azimuth/32
                        vertices.append([size[0]*math.sin(theta)*math.cos(phi),
                                         size[1]*math.sin(theta)*math.sin(phi), size[2]*math.cos(theta)])
                vertices.append([0., 0., -size[2]])
                faces = []
                for j in range(32):
                    faces.append([1, 2+j, 2+(j+1)%32])
                    faces.append([226, 194+(j+1)%32, 194+j])
                for ring in range(6):
                    for j in range(32):
                        a, b = 2+ring*32+j, 2+ring*32+(j+1)%32
                        faces.extend([[a, a+32, b], [b, a+32, b+32]])
                if wheel_vertices < len(vertices):
                    from scipy.spatial import ConvexHull
                    original = np.asarray(vertices)
                    normalized = original / size
                    selected = list(dict.fromkeys(int(f(normalized[:, axis]))
                                    for axis in range(3) for f in (np.argmin, np.argmax)))
                    distance = np.full(len(original), np.inf)
                    for index in selected:
                        distance = np.minimum(distance, ((normalized-normalized[index])**2).sum(-1))
                    while len(selected) < wheel_vertices:
                        index = int(distance.argmax())
                        selected.append(index)
                        distance = np.minimum(distance, ((normalized-normalized[index])**2).sum(-1))
                    vertices = original[selected]
                    hull = ConvexHull(vertices)
                    triangles = hull.simplices.copy()
                    for index, face in enumerate(triangles):
                        a, b, c = vertices[face]
                        if np.dot(np.cross(b-a, c-a), hull.equations[index, :3]) < 0:
                            triangles[index] = face[[0, 2, 1]]
                    faces = triangles + 1
                text = '\n'.join(['v '+_values(v) for v in vertices]+['f '+' '.join(map(str, f)) for f in faces])+'\n'
                (output/mesh_name).write_text(text)
                meshes[mesh_name] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(), vertices=len(vertices),
                    faces=len(faces), source_semiaxes_m=size.tolist(), source_polar_segments=8, source_azimuth_segments=32, sampling='original mesh' if wheel_vertices == 226 else 'farthest normalized coverage subset; all six extrema retained')
            sector_meshes = []
            if wheel_sectors and kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                from .collision_geometry import ellipsoid_sectors
                for sector, (vertices, faces) in enumerate(ellipsoid_sectors(size, sectors=wheel_sectors)):
                    filename = mesh_name.removesuffix('.obj')+f'_sector_{sector}.obj'
                    text = '\n'.join(['v '+_values(v) for v in vertices]+['f '+_values(f+1) for f in faces])+'\n'
                    (output/filename).write_text(text)
                    meshes[filename] = dict(sha256=hashlib.sha256(text.encode()).hexdigest(), vertices=len(vertices), faces=len(faces), source_semiaxes_m=size.tolist())
                    sector_meshes.append(filename)
            for tag in ('collision', 'visual'):
                if tag == 'collision' and sector_meshes:
                    for sector, filename in enumerate(sector_meshes):
                        parent = ET.SubElement(link, tag, name=f'geom_{g}_sector_{sector}')
                        ET.SubElement(parent, 'origin', xyz=_values(model.geom_pos[g]-offsets[i]), rpy=_values(_rpy(model.geom_quat[g])))
                        ET.SubElement(ET.SubElement(parent, 'geometry'), 'mesh', filename=filename)
                    continue
                geometry_parent = ET.SubElement(link, tag, name='geom_'+str(g))
                ET.SubElement(geometry_parent, 'origin', xyz=_values(model.geom_pos[g]-offsets[i]), rpy=_values(_rpy(model.geom_quat[g])))
                geometry = ET.SubElement(geometry_parent, 'geometry')
                if kind == mujoco.mjtGeom.mjGEOM_BOX:
                    ET.SubElement(geometry, 'box', size=_values(2*size))
                elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    ET.SubElement(geometry, 'cylinder', radius=str(size[0]), length=str(2*size[1]))
                else:
                    ET.SubElement(geometry, 'mesh', filename=mesh_name)
        if model.body_parentid[i] == 0:
            continue
        joint_ids = list(range(model.body_jntadr[i], model.body_jntadr[i]+model.body_jntnum[i]))
        j = joint_ids[0] if joint_ids else None
        joint = ET.SubElement(root, 'joint', name=name(mujoco.mjtObj.mjOBJ_JOINT, j) if j is not None else body_name+'_fixed',
                              type=('revolute' if model.jnt_limited[j] else 'continuous') if j is not None else 'fixed')
        ET.SubElement(joint, 'parent', link=name(mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[i])))
        ET.SubElement(joint, 'child', link=body_name)
        rotation=np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_quat[i])
        origin=model.body_pos[i]+rotation.reshape(3,3)@offsets[i]-offsets[model.body_parentid[i]]
        ET.SubElement(joint, 'origin', xyz=_values(origin), rpy=_values(_rpy(model.body_quat[i])))
        if j is not None:
            ET.SubElement(joint, 'axis', xyz=_values(model.jnt_axis[j]))
            limit = dict(effort='1000', velocity='1000')
            if model.jnt_limited[j]:
                limit.update(lower=str(model.jnt_range[j, 0]), upper=str(model.jnt_range[j, 1]))
            ET.SubElement(joint, 'limit', **limit)
            ET.SubElement(joint, 'dynamics', damping='0', friction='0')
    xml = ET.tostring(root, encoding='unicode')
    (output/'vehicle.urdf').write_text(xml)
    contract.update(exported_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(), asset_format='urdf',
                    meshes=meshes, link_frame_offsets={name(mujoco.mjtObj.mjOBJ_BODY,i):offsets[i].tolist() for i in range(1,model.nbody)}, urdf_joint_effort_semantics='1000 is importer metadata; runtime source actuator limits and passive torques are authoritative',
                    root_quat_wxyz=contract['bodies'][0]['quat_wxyz'])
    contract['differences'][0] = f'Ellipsoid wheels explicitly sampled to {wheel_vertices}-vertex convex OBJ; analytic contact not preserved.'
    if wheel_sectors:
        contract['differences'][0] = f'Ellipsoid wheels use historical RB {wheel_sectors} convex sectors, polar32/per-sector4; analytic contact not preserved.'
    contract['wheel_sectors'] = wheel_sectors
    contract['wheel_vertices'] = wheel_vertices
    (output/'contract.json').write_text(json.dumps(contract, indent=2))
    return contract
