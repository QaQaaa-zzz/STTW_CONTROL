"""Construct a declared STTW chassis + RB pendulum candidate without editing sources."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .model import load_model


def build_pendulum_candidate(config_path, output):
    cfg=json.loads(Path(config_path).read_text())
    source=Path(cfg['sttw_source']); pendulum=Path(cfg['pendulum_source'])
    base=load_model(source); donor=mujoco.MjModel.from_xml_path(str(pendulum))
    root=ET.fromstring(base.portable_xml)
    root.find('compiler').set('meshdir',str((source.parent/ET.parse(source).getroot().find('compiler').get('meshdir','')).resolve()))
    chassis=root.find(".//body[@name='chasis']")
    fmt=lambda a:' '.join(format(float(x),'.17g') for x in np.asarray(a).reshape(-1))
    rotation=np.empty(9);mount=np.asarray(cfg['mount_rotation_wxyz']);mujoco.mju_quat2Mat(rotation,mount);rotation=rotation.reshape(3,3)
    parents={'base_link':chassis}; added=[]
    for name in ('downarm','uparm'):
        i=mujoco.mj_name2id(donor,mujoco.mjtObj.mjOBJ_BODY,name)
        parent=mujoco.mj_id2name(donor,mujoco.mjtObj.mjOBJ_BODY,int(donor.body_parentid[i]))
        pos=donor.body_pos[i].copy();quat=donor.body_quat[i].copy()
        if parent=='base_link':
            pos=rotation@pos+np.asarray(cfg['mount_translation_m'])
            q=np.empty(4);mujoco.mju_mulQuat(q,mount,quat);quat=q
        body=ET.SubElement(parents[parent],'body',name=name,pos=fmt(pos),quat=fmt(quat));parents[name]=body
        ET.SubElement(body,'inertial',mass=str(donor.body_mass[i]),pos=fmt(donor.body_ipos[i]),quat=fmt(donor.body_iquat[i]),diaginertia=fmt(donor.body_inertia[i]))
        for j in range(donor.body_jntadr[i],donor.body_jntadr[i]+donor.body_jntnum[i]):
            ET.SubElement(body,'joint',name=mujoco.mj_id2name(donor,mujoco.mjtObj.mjOBJ_JOINT,j),type='hinge',pos=fmt(donor.jnt_pos[j]),axis=fmt(donor.jnt_axis[j]),limited='true',range=fmt(donor.jnt_range[j]),damping=str(donor.dof_damping[donor.jnt_dofadr[j]]))
        for g in range(donor.body_geomadr[i],donor.body_geomadr[i]+donor.body_geomnum[i]):
            if not (donor.geom_contype[g] or donor.geom_conaffinity[g]):continue
            kind={int(mujoco.mjtGeom.mjGEOM_BOX):'box',int(mujoco.mjtGeom.mjGEOM_CYLINDER):'cylinder'}[int(donor.geom_type[g])]
            ET.SubElement(body,'geom',name=mujoco.mj_id2name(donor,mujoco.mjtObj.mjOBJ_GEOM,g),type=kind,size=fmt(donor.geom_size[g][:2 if kind=='cylinder' else 3]),pos=fmt(donor.geom_pos[g]),quat=fmt(donor.geom_quat[g]),density='0',contype=str(donor.geom_contype[g]),conaffinity=str(donor.geom_conaffinity[g]),friction=fmt(donor.geom_friction[g]),solref=fmt(donor.geom_solref[g]),solimp=fmt(donor.geom_solimp[g]),rgba='0.2 0.5 0.8 1')
        added.append(dict(name=name,mass=float(donor.body_mass[i])))
    donor_root=ET.parse(pendulum).getroot()
    # The hip motor belongs to the added mechanism, but is attached to the
    # donor chassis rather than either moving arm.
    motor=donor_root.find(".//geom[@name='hip_motor']")
    gi=mujoco.mj_name2id(donor,mujoco.mjtObj.mjOBJ_GEOM,'hip_motor')
    mass=float(motor.get('mass')); radius,half_length=donor.geom_size[gi,:2]
    q=np.empty(4);mujoco.mju_mulQuat(q,mount,donor.geom_quat[gi])
    motor_body=ET.SubElement(chassis,'body',name='pendulum_hip_motor',pos=fmt(rotation@donor.geom_pos[gi]+np.asarray(cfg['mount_translation_m'])),quat=fmt(q))
    radial=mass*(3*radius**2+(2*half_length)**2)/12
    ET.SubElement(motor_body,'inertial',pos='0 0 0',mass=str(mass),diaginertia=fmt([radial,radial,mass*radius**2/2]))
    ET.SubElement(motor_body,'geom',type='cylinder',size=fmt([radius,half_length]),contype='0',conaffinity='0',density='0',rgba='0.2 0.5 0.8 1')
    added.append(dict(name='pendulum_hip_motor',mass=mass))
    for name in ('cmd_hip_f','cmd_knee_f'):
        node=donor_root.find("actuator/*[@name='"+name+"']")
        root.find('actuator').append(ET.fromstring(ET.tostring(node)))
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    ET.SubElement(contact,'exclude',body1='downarm',body2='uparm')
    # Existing ground keyframes retain every original coordinate and command.
    for key in root.findall('keyframe/key'):
        for field,extra in [('qpos',cfg['pendulum_rest_rad']),('qvel',[0,0]),('ctrl',[0,0])]:
            if field in key.attrib:key.set(field,key.get(field)+' '+fmt(extra))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    target=output/'vehicle.xml';target.write_text(ET.tostring(root,encoding='unicode'))
    model=load_model(target).model
    report=dict(config=cfg,sttw_source=base.identity,pendulum_source_sha256=hashlib.sha256(pendulum.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),base_mass_kg=float(base.model.body_mass.sum()),candidate_mass_kg=float(model.body_mass.sum()),added_bodies=added,nq=model.nq,nv=model.nv,nu=model.nu,status='compiled_candidate_not_isaac_imported_or_policy_validated',limits=['Mount transform is an explicit simulation design, not measured hardware geometry','STTW chassis preserved; old RB checkpoint is not validated on this vehicle','Pendulum self contact masks retained; Isaac importer/runtime must support them'])
    (output/'contract.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
