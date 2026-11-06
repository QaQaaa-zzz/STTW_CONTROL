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


def build_rb_platform(config_path, output):
    """Retain the RB vehicle; express its root and wheel coordinates for STTW."""
    cfg=json.loads(Path(config_path).read_text());source=Path(cfg['source'])
    original=mujoco.MjModel.from_xml_path(str(source))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    target=output/'vehicle.xml';mujoco.mj_saveLastXML(str(target),original)
    root=ET.parse(target).getroot();root.find('compiler').set('meshdir',str(source.parent/'meshes'))
    fmt=lambda a:' '.join(format(float(x),'.17g') for x in a)
    mount=np.array([2**-.5,-2**-.5,0.,0.]);inverse=mount*np.array([1,-1,-1,-1])
    rot=np.empty(9);mujoco.mju_quat2Mat(rot,mount);rot=rot.reshape(3,3)
    body=root.find('worldbody/body');old_name=body.get('name');body.set('name','chasis')
    body.set('quat',fmt(inverse))
    # A coordinate change only: all physical root attachments stay at the same world poses.
    for node in body:
        if node.tag not in ('body','geom','site','inertial'):continue
        node.set('pos',fmt(rot@np.fromstring(node.get('pos','0 0 0'),sep=' ')))
        q=np.empty(4);mujoco.mju_mulQuat(q,mount,np.fromstring(node.get('quat','1 0 0 0'),sep=' '));node.set('quat',fmt(q))
    for node in root.iter():
        for attr in ('body1','body2'):
            if node.get(attr)==old_name:node.set(attr,'chasis')
    for name in ('frontwheel_joint','rearwheel_joint'):
        node=root.find(f".//joint[@name='{name}']")
        node.set('axis',fmt(-np.fromstring(node.get('axis','0 0 1'),sep=' ')))
    for key in root.findall('keyframe/key'):
        q=np.fromstring(key.get('qpos'),sep=' ');new=np.empty(4);mujoco.mju_mulQuat(new,q[3:7],inverse);q[3:7]=new
        for name in ('frontwheel_joint','rearwheel_joint'):q[original.jnt_qposadr[original.joint(name).id]]*=-1
        key.set('qpos',fmt(q))
    for geom in root.findall('.//geom'):geom.set('friction',fmt(cfg['friction']))
    actuator=root.find('actuator')
    steering=actuator.find("*[@name='cmd_steering_v']");steering.set('name','cmd_steering_p');steering.set('group','1')
    rear=actuator.find("*[@name='cmd_rearwheel_f']");rear.set('name','cmd_rearwheel_v');rear.set('ctrlrange','-40 5')
    ET.SubElement(actuator,'velocity',name='cmd_steering_v',joint='steering_joint',kv=str(cfg['ground_steering_velocity_gain']),ctrlrange='-3 3',forcerange=steering.get('forcerange'),ctrllimited='true',forcelimited='true')
    for sensor in root.findall('sensor/actuatorfrc'):
        rename={'cmd_steering_v':'cmd_steering_p','cmd_rearwheel_f':'cmd_rearwheel_v'}
        sensor.set('actuator',rename.get(sensor.get('actuator'),sensor.get('actuator')))
    target.write_text(ET.tostring(root,encoding='unicode'))
    model=load_model(target).model
    before=mujoco.MjData(original);after=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(original,before,0);mujoco.mj_resetDataKeyframe(model,after,0)
    mujoco.mj_forward(original,before);mujoco.mj_forward(model,after)
    geometry_error=float(np.max(np.abs(before.geom_xpos-after.geom_xpos)))
    com_error=float(np.max(np.abs(before.xipos-after.xipos)))
    assert geometry_error<1e-6 and com_error<1e-6,(geometry_error,com_error)
    assert np.allclose(original.body_mass,model.body_mass,rtol=0,atol=1e-9)
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),mass_kg=float(model.body_mass.sum()),wheel_radius_m=.07,friction=cfg['friction'],coordinate_changes=['root local frame rotated, world geometry preserved','wheel coordinates reversed; physical rear target sign reversed'],ground_interface='added steering velocity servo, original position servo selected for jump',geometry_world_max_error_m=geometry_error,body_com_world_max_error_m=com_error,physics_steps=0,optimizer_updates=0,status='asset_prepared_not_runtime_validated')
    (output/'identity.json').write_text(json.dumps(report,indent=2));return report
