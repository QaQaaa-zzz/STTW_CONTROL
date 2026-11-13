"""Frozen native RB -> STTW trial using Playground MJX stepping; no learning."""
import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
import mujoco
from mujoco import mjx
# Increase per-pair heightfield contact workspace, retaining the original terrain surface.
import warp as wp
from mujoco.mjx.third_party.mujoco_warp._src import collision_convex
collision_convex.MJ_MAXCONPAIR = 512
collision_convex.vec_maxconpair = wp.types.vector(length=512, dtype=float)
collision_convex.mat_maxconpair = wp.types.matrix(shape=(512, 3), dtype=float)
import torch
from sttw_control.ditch_terrain import build_ditch_terrain
from sttw_control.runtime import configure_compilation_cache
from sttw_control.pendulum_skill import PendulumSkill
from jit_dvgc.model import _build_index


def transplant(bundle, terrain, output, physics_substeps=1, ground_physics=None):
    """Use each original Isaac rectangular patch without resampling its heights."""
    root = ET.parse(bundle.xml_path).getroot()
    root.find('compiler').set('meshdir', str(bundle.xml_path.parent/'meshes'))
    root.find('option').set('timestep', str(.005/physics_substeps))
    root.find('option').set('ccd_iterations', '100')
    world, assets = root.find('worldbody'), root.find('asset')
    for name in ('floor', 'step'):
        geom = world.find(f"geom[@name='{name}']")
        geom.set('contype', '0'); geom.set('conaffinity', '0'); geom.set('rgba', '0 0 0 0')
    world.find("geom[@name='step']").set('pos', f"{terrain['front_x_m']+2} 0 0")
    # Explicit pairs bypass contype/conaffinity; remove the replaced ground contacts.
    contacts = root.find('contact')
    native_wheel_pairs=[dict(pair.attrib) for pair in contacts if pair.get('geom1')=='floor' and pair.get('geom2') in ('frontwheel_collision','rearwheel_collision')]
    for pair in list(contacts):
        if pair.tag == 'pair' and {pair.get('geom1'), pair.get('geom2')} & {'floor', 'step'}:
            contacts.remove(pair)
    vertices, triangles, patches = build_ditch_terrain(terrain, return_patches=True)
    heights = []
    for i, (xs, ys, zz) in enumerate(patches):
        bottom=float(zz.min()); scale=max(float(zz.max()-bottom),1e-6)
        ET.SubElement(assets,'hfield',name=f'bank{i}',nrow=str(len(xs)),ncol=str(len(ys)),size=f'{(ys[-1]-ys[0])/2} {(xs[-1]-xs[0])/2} {scale} .1')
        ET.SubElement(world,'geom',name=f'bank{i}',type='hfield',hfield=f'bank{i}',pos=f'{(xs[-1]+xs[0])/2} {(ys[-1]+ys[0])/2} {bottom}',quat='.7071067811865476 0 0 .7071067811865476',friction='.8 .005 .0001',contype='1',conaffinity='15',rgba='.45 .38 .25 1')
        # Rotate local grid 90 degrees so MuJoCo's cell diagonal matches Isaac.
        heights.append(((zz[::-1,:]-bottom)/scale).astype(np.float32))
    ground_physics = ground_physics or {}
    if ground_physics.get('native_wheel_contacts',False):
        for i in range(len(patches)):
            for pair in native_wheel_pairs:ET.SubElement(contacts,'pair',**dict(pair,geom1=f'bank{i}'))
    if 'integrator' in ground_physics:
        root.find('option').set('integrator',ground_physics['integrator'])
    if ground_physics.get('steering_velocity_gain') is not None:
        actuator=root.find("actuator/position[@joint='steering_joint']")
        actuator.tag='velocity'; actuator.attrib.pop('kp'); actuator.set('kv',str(ground_physics['steering_velocity_gain']))
        actuator.set('ctrlrange','-3 3')
    if 'friction' in ground_physics:
        for geom in root.findall('.//geom'):
            if geom.get('name') in ('frontwheel_collision','rearwheel_collision') or geom.get('hfield'):
                geom.set('friction',f"{ground_physics['friction']} .005 .0001")
    if ground_physics.get('wheel_collision') == 'cylinder':
        for name in ('frontwheel_collision','rearwheel_collision'):
            geom=root.find(f".//geom[@name='{name}']")
            geom.set('type','cylinder'); geom.set('size','.07 .025')
    equality=root.find('equality')
    if equality is None:equality=ET.SubElement(root,'equality')
    for joint in ('hip_joint','knee_joint'):
        ET.SubElement(equality,'joint',name=f'ground_brake_{joint}',joint1=joint,active='false',solref='.002 1',solimp='.9999 .9999 .001 .5 2')
    path=output/'terrain.xml'; ET.ElementTree(root).write(path)
    model=mujoco.MjModel.from_xml_path(str(path))
    for i,z in enumerate(heights):
        adr=model.hfield_adr[i]; model.hfield_data[adr:adr+z.size]=z.ravel()
    mujoco.mj_saveModel(model,str(output/'terrain.mjb'),None)
    np.savez_compressed(output/'terrain_source.npz',vertices=vertices,triangles=triangles,**{f'hfield{i}':z for i,z in enumerate(heights)})
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=Path('learning/configs/playground_rb_skill_sources.json'))
    p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skills',nargs='+',default=['source','rb_run1','rb_run2','rb_run3'])
    p.add_argument('--modes',nargs='+',choices=['h75','zero_middle'],default=['h75','zero_middle'])
    p.add_argument('--model',type=Path,help='Reuse an existing terrain MJB from this task')
    p.add_argument('--approach', choices=['h75'], default='h75')
    p.add_argument('--routes',nargs='+',type=int,default=[60,61,62]); p.add_argument('--seconds',type=float,default=10)
    p.add_argument('--physics-substeps',type=int,default=1,choices=[1,5,25],help='Physics substeps per unchanged 5 ms controller tick')
    p.add_argument('--ground-only',action='store_true',help='Keep ground control throughout; no RB activation')
    a=p.parse_args(); configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache'); a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
    cfg=json.loads(a.config.read_text()); hand=cfg['ground_handoff']; terrain=cfg['ditch_terrain']
    ground_physics=cfg.get('ground_physics',{})
    rb_physics=cfg.get('rb_physics')
    if rb_physics and rb_physics != {'integrator':'RK4','physics_dt_s':.005}:
        raise ValueError('RB native runtime requires RK4 at 5 ms')
    velocity_steering='steering_velocity_gain' in ground_physics
    if ground_physics and a.model:
        raise ValueError('Physics adaptation requires a freshly built model')
    if velocity_steering and not a.ground_only:
        raise ValueError('RB requires its native position steering actuator')
    bundle=Path(hand['bundle']); identity=json.loads((bundle/'identity.json').read_text()); hcfg=json.loads((bundle/'config.json').read_text())
    controller=json.loads(Path(json.loads(Path(hand['settings']).read_text())['controller_config']).read_text())
    if 'nominal_steer_slew_rad_s' in hand:hcfg['follower']['nominal_steer_slew_rad_s']=hand['nominal_steer_slew_rad_s']
    overrides=hand.get('follower_overrides',{})
    if set(overrides)-set(hcfg['follower']):raise ValueError('Unknown follower override')
    hcfg['follower'].update(overrides)
    skills={name:PendulumSkill(a.registry,name) for name in a.skills}
    for name,path in cfg.get('rb_adaptation_bundles',{}).items():skills[name].load_adapted_actor(path)
    original=next(iter(skills.values())).bundle; idx=original.model_index
    model=mujoco.MjModel.from_binary_path(str(a.model)) if a.model else transplant(original,terrain,a.output,a.physics_substeps,ground_physics)
    if not np.isclose(model.opt.timestep,.005/a.physics_substeps):
        raise ValueError('Cached terrain model physics dt differs from requested substeps; rebuild it')
    idx=_build_index(model); mm=mjx.put_model(model,impl='warp')
    brake_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_EQUALITY,f'ground_brake_{joint}') for joint in ('hip_joint','knee_joint')])
    if (brake_ids<0).any():raise ValueError('Cached model lacks STTW pendulum brakes; rebuild it')
    if 'ground_suite' in cfg:
        if not a.ground_only:raise ValueError('Ground suite does not execute RB')
        from sttw_control.playground_ground_evaluation import evaluate_ground_suite
        evaluate_ground_suite(model,mm,idx,cfg,bundle,hcfg,identity,controller,a.output,a.physics_substeps,brake_ids)
        return
    n=len(a.routes)*len(skills)*len(a.modes)*len(hand.get('handoff_variants',[{}]))
    contract=dict(terrain_model=str(a.model) if a.model else str(a.output/'terrain.mjb'),role='bounded native Playground transfer DEV, no training',skills={k:s.record for k,s in skills.items()},terrain=terrain,source_config=str(a.config),routes=a.routes,episode_s=a.seconds,episodes=n,physics_steps_max=round(a.seconds/.005)*n*(a.physics_substeps+(1 if rb_physics else a.physics_substeps)),backend='MJX-Warp vmap; GPU Torch H75/ECBC/ESO; JAX RB; DLPack; final-only host export',heightfield_contact_workspace=512,physics_dt=.005/a.physics_substeps,physics_substeps=a.physics_substeps,lower_dt=.005,upper_dt=.02,ground_model=identity,goal=hand['goal'],road_half_width_m=hand['road_half_width_m'],start_x_m=hand.get('start_x_m',1.5),start_vx_m_s=hand.get('start_vx_m_s',2.),handoff=hand.get('handoff_mechanism','fixed contact delay'),handoff_delay_s=hand.get('handoff_delay_s',0.),approach=a.approach,ground_only=a.ground_only,jump_start_x_m=hand['approach']['jump_start_x_m'],scope='Declared approach controller then native RB jump then H75 or zero-middle STTW. No planning or training.',ground_physics=ground_physics,rb_physics=rb_physics,terrain_conversion='Original Isaac rectangular patches, coordinates and heights retained without interpolation; 90-degree local grid rotation matches original triangle diagonal',pendulum='STTW hip/knee equality brakes at entry posture; released during RB, relocked at measured handoff posture; native finite servos retained',observer='measured ESO initialization, empty H75 history at handoff')
    (a.output/'declaration.json').write_text(json.dumps(contract,indent=2))
    from sttw_control.playground_handoff_evaluation import evaluate_handoff
    evaluate_handoff(model,mm,idx,cfg,bundle,hcfg,identity,controller,skills,a,brake_ids)


if __name__=='__main__':main()
