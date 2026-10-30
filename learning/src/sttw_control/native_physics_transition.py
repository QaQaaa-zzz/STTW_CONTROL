"""Explicit fresh-physics preparation; no old prepared state is transplanted.

Load by file in a fresh process for execute, preserving frozen package imports.
Preparation copies authenticated source/assets only. Execute is a separate,
finite ECBC-only eight-state preparation, with no policy training/evaluation.
"""
import copy
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
import pickle
import shutil
import sys
import xml.etree.ElementTree as ET


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def wheel_assets(xml):
    """Resolve real mesh files without copying unused plugin/cache/run assets."""
    xml=Path(xml).resolve();root=ET.parse(xml).getroot();parent={c:p for p in root.iter() for c in p}
    wheels=root.findall(".//geom[@class='wheel']")
    if len(wheels)!=2 or any(float(g.get('friction','nan').split()[0])!=.8 for g in wheels):raise ValueError('two explicit wheel sliding frictions .8 required')
    meshdir=root.find('compiler').get('meshdir','');assets=[]
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file'):
            relative=Path(meshdir)/mesh.get('file')
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('asset path escapes model directory')
            assets.append(relative)
    if root.findall('include'):raise ValueError('includes need an explicit expanded asset contract')
    return sorted(set(assets)),[parent[g].get('name') for g in wheels]


def prepare_transition(source,root,config):
    source=Path(source).resolve();root=Path(root).resolve();config=Path(config).resolve();cfg=read(config);old=read(source/'plan.json')
    if root.exists():raise ValueError('fresh isolated output directory required')
    if digest(source/'plan.json')!=read(source/'seal.json')['plan_sha256']:raise ValueError('source seal differs')
    randomized=cfg.get('initial_randomization')
    count=8 if randomized is None else randomized['count']
    if (not isinstance(count,int) or count<1 or cfg['maximum_preparation_lower_ticks']!=count*700
        or cfg['preparation_ticks']!=700 or not 0<=cfg['evaluation_bank_index']<count):
        raise ValueError('explicit state count and 700-tick preparation budget required')
    if randomized is None and cfg['evaluation_bank_index']!=6:raise ValueError('original bank6 contract required')
    if randomized is not None:
        import numpy as np
        bounds=np.asarray([randomized['speed_m_s'],randomized['roll_rad']],dtype=float)
        if (bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>=bounds[:,1])
            or bounds[0,0]<1.5 or bounds[0,1]>3.5 or np.max(np.abs(bounds[1]))>.1
            or randomized['data_role'] not in ('TRAIN','DEV','TEST')):
            raise ValueError('bounded explicit random initial distribution and data role required')
        # Stratify speed so a finite bank covers the full declared interval.
        rng=np.random.default_rng(randomized['seed'])
        samples=rng.uniform(size=(count,2))
        samples[:,0]=(rng.permutation(count)+samples[:,0])/count
        conditions=(bounds[:,0]+samples*(bounds[:,1]-bounds[:,0])).tolist()
        excitation=cfg.get('terminal_excitation')
        if excitation is not None:
            force=np.asarray(excitation['world_y_force_N'],float)
            duration=np.asarray(excitation['duration_s'],float)
            limits=excitation['acceptance']
            if (force.shape!=(2,) or duration.shape!=(2,) or not np.isfinite(np.r_[force,duration]).all()
                or force[0]>=force[1] or np.max(np.abs(force))>30 or not 0<duration[0]<=duration[1]<=.5
                or set(limits)!={'end_roll_abs_max_rad','end_roll_rate_abs_max_rad_s','end_speed_error_abs_max_m_s','end_steer_abs_max_rad'}
                or any(not np.isfinite(v) or v<=0 for v in limits.values())):
                raise ValueError('bounded terminal excitation and explicit acceptance required')
            pulse_force=rng.uniform(*force,size=count)
            pulse_duration=rng.uniform(*duration,size=count)
            pulses=np.column_stack([3.5-pulse_duration,np.full(count,3.5),np.zeros(count),pulse_force,np.zeros(count)]).tolist()
    xml=Path(cfg['source_xml']).resolve()
    if digest(xml)!=cfg['expected_source_xml_sha256']:raise ValueError('new XML identity differs')
    assets,wheel_bodies=wheel_assets(xml);native=Path(old['native_runtime_root']).resolve()
    specpath=Path(old['initial_source']['upper_run'])/'alpha0/frozen_config_stage52.json';spec=read(specpath)
    if digest(specpath)!=cfg['expected_plant_spec_sha256']:raise ValueError('original preparation specification identity differs')
    init=spec['initialization']
    if len(init['initial_conditions_speed_roll'])!=8 or init['preparation_seconds']!=3.5 or not init['preparation_zero_correction']:raise ValueError('original preparation contract differs')
    for model in [old['models']['H75'],old['initial_source']['candidate']]:
        for key in ('actor','checkpoint'):
            if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen checkpoint identity differs')
    if randomized is not None:
        init=copy.deepcopy(init)
        init['initial_conditions_speed_roll']=conditions
        init['randomization']=randomized
        if excitation is not None:
            init.update(excitation['acceptance'])
            init['terminal_excitation']=excitation
            init['terminal_force_pulses']=pulses
        spec=copy.deepcopy(spec);spec['initialization']=init
    root.mkdir();runtime=root/'native_runtime';sources={}
    for name,expected in old['native_expected_sources'].items():
        path=Path(name).resolve()
        if not path.is_relative_to(native):continue  # Old receipt also lists CLI/tests, unused by this preparation.
        if digest(path)!=expected:raise ValueError('native source identity differs')
        dest=runtime/'learning/src'/path.relative_to(native);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);sources[str(dest)]=digest(dest)
    for name in ('teleop_pref_governor_v1.json','fixed_endpoint_directional_alpha0.json'):
        src=native.parent/'configs'/name;dest=runtime/'learning/configs'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);sources[str(dest)]=digest(dest)
    modeldir=runtime/'mujoco_ros_ws20260313/model';modeldir.mkdir(parents=True)
    for relative in [Path('scalebike_scene_matlab.xml'),*assets]:
        src=xml if relative.name=='scalebike_scene_matlab.xml' else xml.parent/relative;dest=modeldir/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);sources[str(dest)]=digest(dest)
    shutil.copyfile(config,root/'resolved_config.json');write(root/'plant_spec.json',spec)
    shutil.copyfile(__file__,root/'preparation_utility.py')
    for artifact in ('resolved_config.json','plant_spec.json','preparation_utility.py'):sources[str(root/artifact)]=digest(root/artifact)
    initial=copy.deepcopy(old['initial_source']);initial.pop('prepared_bank');initial.pop('prepared_bank_sha256');initial.pop('model');initial['bank_index']=cfg['evaluation_bank_index']
    plan=dict(mode='native_random_initial_preparation' if randomized else 'native_physics_transition',source=str(source),source_plan_sha256=digest(source/'plan.json'),old_model_identity=old['initial_source']['model'],old_bank_reuse=False,initial_source_pending=initial,
        models={'H75':old['models']['H75']},config=old['config'],contexts=old['contexts'],native_runtime_root=str(runtime/'learning/src'),copied_files_sha256=sources,wheel_bodies=wheel_bodies,
        preparation_constructor='TeleopEnv with sealed plant specification; fresh physical initial conditions followed by 700 ECBC-only ticks per state. No learned lower policy during preparation.',initialization=init,preparation_ticks=700,maximum_preparation_lower_ticks=count*700,optimizer_updates=0,policy_evaluation_lower_ticks=0,
        utility_sha256=digest(__file__),shared_jax_cache=cfg['shared_jax_cache'],inputs={str(config):digest(config),str(specpath):digest(specpath),str(xml):digest(xml)},limits='Physics transition: old H75 checkpoint unchanged; old q labels/calibration belong to old physics. Fresh ECBC preparation only, not five-alpha task evaluation.')
    write(root/'plan.json',plan);write(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')));write(root/'status.json',dict(phase='prepared',actual_preparation_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    return plan


def execute_preparation(root):
    """Explicit caller authorization required; not called by prepare or tests."""
    root=Path(root).resolve();p=read(root/'plan.json')
    if digest(root/'plan.json')!=read(root/'seal.json')['plan_sha256'] or read(root/'status.json')['phase']!='prepared':raise ValueError('fresh prepared/sealed transition required')
    if digest(__file__)!=p['utility_sha256']:raise ValueError('preparation utility implementation drift')
    for path,expected in {**p['inputs'],**p['copied_files_sha256']}.items():
        if digest(path)!=expected:raise ValueError('transition input/source drift')
    if any(name=='sttw_control' or name.startswith('sttw_control.') for name in sys.modules):raise ValueError('execute in a fresh process, before modern package import')
    sys.dont_write_bytecode=True;sys.path.insert(0,p['native_runtime_root'])
    import jax
    import jax.numpy as j
    import numpy as np
    from dataclasses import asdict
    from sttw_control.teleop_env import TeleopEnv
    from sttw_control.actuator import initial_actuator
    from sttw_control.runtime import configure_compilation_cache
    configure_compilation_cache(p['shared_jax_cache'])
    started=time.monotonic();status=dict(phase='initializing',pid=os.getpid(),actual_preparation_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0)
    write(root/'status.json',status)
    try:
        physics=TeleopEnv(config=read(root/'plant_spec.json'),backend='mjx');base=physics.initial();conditions=j.asarray(p['initialization']['initial_conditions_speed_roll']);jax.block_until_ready(base)
        if asdict(physics.cc)!=p['initial_source_pending']['controller'] or asdict(physics.ac)!=p['initial_source_pending']['actuator']:raise ValueError('controller/actuator contract differs')
        def initial(condition):
            speed,angle=condition;q=base.data.qpos;w,x,y,z=q[3:7];ca,sa=j.cos(-angle/2),j.sin(-angle/2)
            q=q.at[3:7].set(j.array([ca*w-sa*x,ca*x+sa*w,ca*y-sa*z,ca*z+sa*y]));b=physics.bundle
            vel=base.data.qvel.at[0].set(speed).at[b.rear_dof].set(-speed/.1).at[b.front_dof].set(-speed/.1)
            data=base.data.replace(qpos=q,qvel=vel,ctrl=base.data.ctrl.at[1].set(-speed/.1));data=physics.helpers._mjx.forward(physics.helpers.mjx_model,data)
            pose=physics.helpers.pose(data);raw=j.array([speed,0.])
            return base.replace(data=data,raw=raw,actuator=initial_actuator(physics.ac,speed/.1),reference_pose=pose,yaw_wrapped=pose[2],yaw_unwrapped=pose[2],governor=base.governor.replace(current_reference=raw,last_goal=raw))
        bank=jax.jit(jax.vmap(initial))(conditions)
        count=len(conditions)
        actual_log_blocks=[]
        # Same ECBC-only _step loop, batched states; preserve every tick/failure.
        if 'terminal_force_pulses' in p['initialization']:
            pulses=j.asarray(p['initialization']['terminal_force_pulses'])
            def chunk(bank):return jax.lax.scan(lambda b,_:jax.vmap(lambda state,pulse:physics._step(state,state.raw,True,force_pulse=pulse))(b,pulses),bank,None,length=50)
        else:
            def chunk(bank):return jax.lax.scan(lambda b,_:jax.vmap(lambda state:physics._step(state,state.raw,True))(b),bank,None,length=50)
        chunk=jax.jit(chunk)
        for block in range(14):
            status.update(phase='inflight',allocated_lower_slots=(block+1)*50*count,inflight_cost_unknown=True);write(root/'status.json',status)
            bank,logs=chunk(bank);jax.block_until_ready(bank)
            actual_log_blocks.append(jax.device_get(logs))
            with (root/'last_preparation_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(bank),f)
            np.savez_compressed(root/'preparation_trace.npz',**{k:np.concatenate([np.asarray(x[k]) for x in actual_log_blocks]) for k in ('actual_xy','phi','phi_dot','actual_forward_speed','actual_delta','applied_force_impulse') if k in actual_log_blocks[0]})
            status.update(phase='preparing',actual_preparation_lower_ticks=(block+1)*50*count,inflight_cost_unknown=False,elapsed_s=time.monotonic()-started);write(root/'status.json',status)
            if bool(j.any(bank.failed)):raise RuntimeError('fresh preparation failed; no replacement or retry')
        last=jax.tree.map(lambda x:np.asarray(x[-1]),logs);limits=p['initialization']
        passed=(~np.asarray(bank.failed))&(abs(last['phi'])<=limits['end_roll_abs_max_rad'])&(abs(last['phi_dot'])<=limits['end_roll_rate_abs_max_rad_s'])&(abs(last['actual_delta'])<=limits['end_steer_abs_max_rad'])&(abs(last['actual_forward_speed']-np.asarray(conditions[:,0]))<=limits['end_speed_error_abs_max_m_s'])
        write(root/'prepared_metrics.json',dict(passed=passed.tolist(),conditions=np.asarray(conditions).tolist(),roll=last['phi'].tolist(),roll_rate=last['phi_dot'].tolist(),speed=last['actual_forward_speed'].tolist(),steer=last['actual_delta'].tolist()))
        with (root/'prepared_bank.pkl').open('wb') as f:pickle.dump(jax.device_get(bank),f)
        if not passed.all():raise RuntimeError('fresh preparation acceptance failed; raw bank retained, not adopted')
        initial=copy.deepcopy(p['initial_source_pending']);initial.update(model=physics.bundle.identity,prepared_bank=str(root/'prepared_bank.pkl'),prepared_bank_sha256=digest(root/'prepared_bank.pkl'))
        write(root/'initial_source.json',initial);write(root/'native_expected_sources.json',{str(f):digest(f) for f in Path(p['native_runtime_root']).rglob('*.py')})
        status.update(phase='completed',old_prepared_state_reused=False,physical_tick_end=np.asarray(bank.physical_tick).tolist());write(root/'status.json',status)
    except BaseException as exc:
        status.update(phase='error',error=repr(exc));write(root/'status.json',status);raise
