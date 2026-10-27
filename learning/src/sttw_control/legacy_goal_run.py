"""Fresh-process orchestration around byte-frozen legacy control modules.

This module is loaded by file before importing sttw_control. It never imports
modern control code into the archived package. All plant steps use the existing
20s-capable reference-step factory with its declared reentry permission.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def own_files():
    folder=Path(__file__).resolve().parent
    return [Path(__file__).resolve(),folder.parent.parent/'cli/run_legacy_goal_continuation.py',
            folder/'goal_replanning.py',folder/'_vendor/pythonrobotics/dubins.py',
            folder/'_vendor/pythonrobotics/angle.py',folder/'_vendor/pythonrobotics/UPSTREAM.json',
            folder/'_vendor/pythonrobotics/LICENSE']


def prepare_execution(root):
    root=Path(root).resolve()
    if (root/'execution_plan.json').exists():raise FileExistsError('execution contract already exists')
    preparation=json.loads((root/'plan.json').read_text());seal=json.loads((root/'seal.json').read_text())
    for name,key in [('plan.json','plan_sha256'),('inventory.json','inventory_sha256'),('runtime_manifest.json','runtime_manifest_sha256')]:
        if digest(root/name)!=seal[key]:raise ValueError('preparation drift '+name)
    audit=json.loads((root/'input_audit.json').read_text())
    if audit['status']!='completed' or audit['cases_checked']!=90 or not audit['initial_full_carry_equality_excluding_alpha']:
        raise ValueError('actual input audit required')
    for name,sha in preparation['input_sha256'].items():
        if digest(name)!=sha:raise ValueError('preparation input changed '+name)
    # Frozen modules resolve these non-Python inputs relative to their package.
    # Reuse the existing assets; environment loading verifies the full original
    # model/controller/actuator identities against the archived declaration.
    worktree=Path(__file__).resolve().parents[3]
    runtime=Path(preparation['runtime_root']).parents[1]
    for link,target in [(runtime/'learning/configs',worktree/'learning/configs'),
                        (runtime/'mujoco_ros_ws20260313',worktree/'mujoco_ros_ws20260313')]:
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        if link.resolve()!=target.resolve():raise ValueError('runtime resource link changed')
    resource_files=[worktree/'learning/configs'/name for name in
                    ('fixed_endpoint_directional_alpha0.json','teleop_pref_governor_v1.json')]
    resource_files.append(worktree/'mujoco_ros_ws20260313/model/scalebike_scene_matlab.xml')
    files=own_files()
    execution=dict(preparation_plan_sha256=digest(root/'plan.json'),preparation_seal_sha256=digest(root/'seal.json'),
        input_audit_sha256=digest(root/'input_audit.json'),case_count=90,horizon_s=20.,chunks=20,upper_steps_per_chunk=50,
        maximum_active_lower_ticks=360000,maximum_allocated_lower_slots=360000,optimizer_updates=0,
        initialization='Original archived task0/physical700 states, no reset/alpha substitution; actors inference only',
        control='Archived make_reference_step(...horizon_s=20.,allow_reentry=True); original per-lane floor and profile0, no new control terms',
        planning='Existing one-shot1s causal Dubins; original goal frozen. Geometry-only rebind retains clean counts and all other carry.',
        groups='All compatible Actor parameter trees share one batched90 group; incompatible families split without padding.',
        source_sha256={str(p):digest(p) for p in files},prepared=True,execution_ready=False,
        resource_sha256={str(p):digest(p) for p in resource_files},
        stop=preparation['stop'],data_role=preparation['data_role'])
    archive=root/'host_source';archive.mkdir()
    for p in files:(archive/p.name).write_bytes(p.read_bytes())
    save(root/'execution_plan.json',execution)
    save(root/'execution_seal.json',dict(execution_plan_sha256=digest(root/'execution_plan.json')))
    return execution


def verify(root):
    root=Path(root).resolve();p=json.loads((root/'plan.json').read_text());e=json.loads((root/'execution_plan.json').read_text())
    seal=json.loads((root/'seal.json').read_text())
    if digest(root/'plan.json')!=seal['plan_sha256'] or digest(root/'inventory.json')!=seal['inventory_sha256']:
        raise ValueError('preparation drift')
    if digest(root/'execution_plan.json')!=json.loads((root/'execution_seal.json').read_text())['execution_plan_sha256']:
        raise ValueError('execution plan drift')
    for f,sha in {**p['input_sha256'],**p['runtime_expected_sources'],**e['source_sha256'],**e['resource_sha256']}.items():
        if digest(f)!=sha:raise ValueError('sealed source/input drift '+f)
    if e['preparation_plan_sha256']!=digest(root/'plan.json') or e['input_audit_sha256']!=digest(root/'input_audit.json'):
        raise ValueError('execution source binding changed')
    mount=Path('/media/qy/6A7E3DB07E3D75C3')
    if not os.path.ismount(mount) or not root.is_relative_to(mount):raise ValueError('mounted HDD required')
    stat=os.statvfs('/home/qy/STTW_CONTROL')
    if stat.f_bavail*stat.f_frsize<50*1024**3:raise ValueError('system free space below50GiB')
    return p,e


def make_geometry_rebind(env,project,pursuit):
    """One pure geometry substitution, without clearing any dynamic state."""
    import jax.numpy as j
    def one(carry,path,speed):
        _,pose,actual_speed=env.physics.observe(carry.state.physical.data)
        projection=project(path,pose[:2],j.float32(0),j.float32(0),env.config)
        follower=pursuit(path,projection.progress,pose,actual_speed,speed,env.config)
        return carry.replace(state=carry.state.replace(projection=projection,follower=follower))
    return one


def assert_geometry_only(before,after):
    """Check the new operation's invariant, not replay of historical physics."""
    import jax
    restored=after.replace(state=after.state.replace(projection=before.state.projection,follower=before.state.follower))
    if jax.tree.structure(before)!=jax.tree.structure(restored):raise ValueError('rebind changed state schema')
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(restored)):
        if not np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True):
            raise ValueError('rebind changed non-geometric state')


def unexplained_boundary_stop(state_fault,physical_failed,known_contact):
    """Observation faults can latch before any active tick emits a log."""
    return np.asarray(state_fault,bool)|(np.asarray(physical_failed,bool)&~np.asarray(known_contact,bool))


def planner_module():
    from types import ModuleType
    import sttw_control.geometric_path as geometry
    folder=Path(__file__).resolve().parent
    package=ModuleType('_sttw_legacy_planner');package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    sys.modules[package.__name__+'.geometric_path']=geometry
    name=package.__name__+'.goal_replanning'
    spec=importlib.util.spec_from_file_location(name,folder/'goal_replanning.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def load(root):
    root=Path(root).resolve();plan,execution=verify(root)
    if any(n=='sttw_control' or n.startswith('sttw_control.') for n in sys.modules):
        raise ValueError('fresh process required; no modern package mixing')
    native=Path(plan['runtime_root']);sys.path.insert(0,str(native))
    import jax
    import jax.numpy as j
    from sttw_control.contact_only_environment import load_environment_contact_only
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.geometric_path import Path as GeometricPath,project
    from sttw_control.path_reference import pursuit
    from sttw_control.anticipatory_reference import make_reference_step
    from sttw_control.reference_reentry import ReentryEvaluationState
    from sttw_control.causal_reference_baseline import initial_gate
    from sttw_control.anticipatory_run import architecture_groups
    from sttw_control.runtime import configure_compilation_cache
    cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
    cases=json.loads((root/'inventory.json').read_text());cfgs={};declarations={};banks={};path_bank={};models=None
    for route,context in plan['routes'].items():
        cfgs[route]=json.loads(Path(context['config']).read_text())
        declarations[route]=json.loads(Path(context['declaration']).read_text())
        with Path(context['initial']).open('rb') as f:banks[route]=pickle.load(f).state
        with np.load(context['path']) as z:path_bank[route]=GeometricPath(*(j.asarray(z[k]) for k in ('s','xy','heading','curvature','turn_end','goal')))
        these=json.loads(Path(context['models']).read_text())['models']
        if models is None:models=these
        elif {k:models[k]['actor_sha256'] for k in ('parent','task_only','physical_credit')}!={k:these[k]['actor_sha256'] for k in ('parent','task_only','physical_credit')}:
            raise ValueError('model bank differs between routes')
    if cfgs['left']!=cfgs['right'] or declarations['left']['source']!=declarations['right']['source']:
        raise ValueError('one identical frozen environment required')
    cfg=cfgs['left'];env,_=load_environment_contact_only(cfg,declarations['left']['source'],'terminal_contract_v2')
    if env.cc.dt!=.005:raise ValueError('legacy lower clock changed')
    params={}
    for role in {c['actor_role'] for c in cases}:
        with Path(models[role]['actor']).open('rb') as f:params[role]=jax.tree.map(j.asarray,pickle.load(f))
    states=[jax.tree.map(lambda x:j.asarray(x[c['source_batch_slot']]),banks[c['route_group']]) for c in cases]
    for s,c in zip(states,cases):
        if int(s.tick)!=0 or int(s.physical.physical_tick)!=700 or float(s.alpha)!=c['state_alpha'] or bool(s.fault|s.physical.failed|s.domain_exit):
            raise ValueError('invalid complete original state')
    for s,c in zip(states,cases):
        obs,_,bad=env.observation(s,path_bank[c['route_group']],j.float32(c['requested_speed_m_s']),20.)
        if obs.shape!=(351,) or bool(bad) or not np.isfinite(np.asarray(obs)).all():raise ValueError('invalid original observation')
    env.set_log_template(states[0],path_bank['left'],j.float32(cases[0]['requested_speed_m_s']))
    stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
    groups=[]
    for family in architecture_groups(list(range(90)),cases,models,cfg,batch_size=90):
        ids=family['case_indices'];selected=[cases[i] for i in ids];n=len(ids)
        one=make_reference_step(env,make_path_actor(family['actor_config']),horizon_s=20.,allow_reentry=True)
        # Bind this factory's one-step function per group, not a changing loop cell.
        def build_chunk(step):
            @jax.jit
            def chunk(c,p,paths,speeds,pulses,profiles,floors,zero):
                return jax.lax.scan(lambda st,_:jax.vmap(step)(st,p,paths,speeds,pulses,profiles,floors,zero),c,None,length=50)
            return chunk
        carry=ReentryEvaluationState(stack([states[i] for i in ids]),stack([initial_gate() for _ in ids]),j.zeros(n,j.int32))
        paths=[path_bank[c['route_group']] for c in selected]
        goals=np.array([[np.interp(float(path.goal),np.asarray(path.s),np.asarray(path.xy)[:,k]) for k in (0,1)] for path in paths])
        pulses=np.array([[float(states[i].physical.physical_tick)*.005+c['pulse_onset_task_tick']*.005,
            float(states[i].physical.physical_tick)*.005+c['pulse_onset_task_tick']*.005+c['pulse_duration_s'],
            *(c['force_N']*np.asarray(c['world_direction']))] for i,c in zip(ids,selected)],np.float32)
        args=(stack([params[c['actor_role']] for c in selected]),stack(paths),
            j.asarray([c['requested_speed_m_s'] for c in selected],j.float32),j.asarray(pulses),
            j.asarray([c['profile'] for c in selected],j.int32),j.asarray([c['reference_speed_floor_m_s'] for c in selected],j.float32),
            j.asarray([c['zero_mode'] for c in selected]))
        rebind=make_geometry_rebind(env,project,pursuit)
        def build_bind(rebinder,size):
            @jax.jit
            def bind(carry,paths,speeds,mask):
                new=jax.vmap(rebinder)(carry,paths,speeds)
                return jax.tree.map(lambda a,b:j.where(mask.reshape((size,)+(1,)*(a.ndim-1)),a,b),new,carry)
            return bind
        @jax.jit
        def poses(carry):return jax.vmap(lambda data:env.physics.observe(data)[1])(carry.state.physical.data)
        groups.append(dict(ids=ids,cases=selected,carry=carry,args=args,paths=paths,goals=goals,pulses=pulses,
            chunk=build_chunk(one),bind=build_bind(rebind,n),poses=poses,architecture=family['architecture']))
    consumed={}
    for name,module in tuple(sys.modules.items()):
        if name=='sttw_control' or name.startswith('sttw_control.'):
            f=getattr(module,'__file__',None)
            if not f:continue
            f=str(Path(f).resolve())
            if not Path(f).is_relative_to(native) or plan['runtime_expected_sources'].get(f)!=digest(f):raise ValueError('mixed/unsealed legacy import '+f)
            consumed[f]=digest(f)
    return dict(plan=plan,execution=execution,cases=cases,config=cfg,groups=groups,env=env,cache=cache,
        path_bank=path_bank,consumed=consumed,planner=planner_module())


def flatten(logs):
    return {k:np.asarray(v).transpose((0,2,1)+tuple(range(3,v.ndim))).reshape((-1,v.shape[1])+v.shape[3:]) for k,v in logs.items()}


def run(root,*,shape_only=False):
    root=Path(root).resolve();directory=root/('shape' if shape_only else 'evaluation')
    if not shape_only:
        ready=json.loads((root/'prelaunch_receipt.json').read_text())
        if not ready['execution_ready'] or ready['execution_plan_sha256']!=digest(root/'execution_plan.json'):
            raise ValueError('final bound prelaunch receipt required')
    directory.mkdir(exist_ok=False);started=time.monotonic();actual=slots=0
    status=dict(phase='initializing',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,completed_cases=0)
    def update(**values):
        status.update(values,elapsed_s=time.monotonic()-started)
        save(directory/'status.json',status)
        if not shape_only:save(root/'status.json',status)
    update()
    try:
        import jax
        import jax.numpy as j
        runtime=load(root);shape_rows=[]
        for group in runtime['groups']:
            carry=group['carry'];args=group['args'];n=len(group['cases'])
            shaped=jax.eval_shape(group['chunk'],carry,*args)
            jax.eval_shape(group['bind'],carry,args[1],args[2],j.zeros(n,bool))
            # A real batch rebind without integration: deliberately mark one
            # lane, confirm all histories/control states survive the operation.
            rebound=group['bind'](carry,args[1],args[2],j.arange(n)==0)
            jax.block_until_ready(rebound);assert_geometry_only(carry,rebound)
            # Exercise the actual upstream planner on one actual source pose;
            # the prospective goal and array shape are checked without stepping.
            poses=np.asarray(jax.device_get(group['poses'](carry)))
            count=len(group['paths'][0].s);length=(count-1)*.02
            candidate,meta=runtime['planner'].dubins_goal_path(poses[0],group['goals'][0],radius_m=1.8,length_m=length,spacing_m=.02)
            if len(candidate.s)!=count:raise ValueError('planner path shape differs from original')
            goal=np.array([np.interp(float(candidate.goal),np.asarray(candidate.s),np.asarray(candidate.xy)[:,k]) for k in (0,1)])
            np.testing.assert_allclose(goal,group['goals'][0],atol=2e-5,rtol=0)
            shape_rows.append(dict(case_ids=[c['case_id'] for c in group['cases']],real_lanes=n,
                active_tick_shape=list(shaped[1]['active_tick'].shape),original_path_points=count,
                planner_checked_on_actual_pose=True,rebind_non_geometric_leaves_preserved=True,
                pulse_first=group['pulses'][0].tolist(),architecture=group['architecture']))
        receipt=dict(groups=shape_rows,consumed_archived_sources=runtime['consumed'],physics_steps=0,
            optimizer_updates=0,original_cases=len(runtime['cases']),cache_directory=runtime['cache'],
            execution_plan_sha256=digest(root/'execution_plan.json'),source=runtime['execution']['source_sha256'],
            devices=[str(d) for d in jax.devices()],scope='Actual environment/351 observation/Actor shapes and pure geometry operations; zero plant integration, no control outcome evidence.')
        save(directory/'shape_receipt.json',receipt)
        if shape_only:
            update(phase='completed',physics_steps=0);return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual evaluation requires GPU')
        (directory/'chunks').mkdir();(directory/'cases').mkdir()
        for route,path in runtime['path_bank'].items():
            np.savez_compressed(directory/(route+'_original_path.npz'),**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
        save(directory/'manifest.json',dict(plan_sha256=digest(root/'plan.json'),execution_plan_sha256=digest(root/'execution_plan.json'),
            cases=runtime['cases'],config=runtime['config'],consumed_archived_sources=runtime['consumed'],
            physics_identity=runtime['env'].physics.bundle.identity,lower_dt_s=.005,upper_dt_s=.02,horizon_s=20.,
            new_optimizer_updates=0,original_world_goals_by_case={c['case_id']:group['goals'][i].tolist()
                for group in runtime['groups'] for i,c in enumerate(group['cases'])},
            clock='physical_tick/time are task pre-step; actual_xy is post-step; force pulse is absolute physical time',
            reward_role='Current-route V3 diagnostic only after replan, not canonical original-alpha objective'))
        decisions=[];index=[];initial_audit=[]
        for g,group in enumerate(runtime['groups']):
            n=len(group['cases']);carry=group['carry'];parameters,paths,speeds,pulses,profiles,floors,zero=group['args']
            path_list=list(group['paths']);changed=np.zeros(n,bool);arrived=np.zeros(n,bool);known_contact=np.zeros(n,bool);chunks=[]
            with (directory/f'initial_carry_group_{g:02d}.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            initial_audit.extend(dict(case_id=c['case_id'],task_tick=int(np.asarray(carry.state.tick)[i]),
                physical_tick=int(np.asarray(carry.state.physical.physical_tick)[i]),state_alpha=float(np.asarray(carry.state.alpha)[i]),
                pulse=group['pulses'][i].tolist()) for i,c in enumerate(group['cases']))
            for block in range(20):
                if block:
                    poses=np.asarray(jax.device_get(group['poses'](carry)))
                    cross,invalid,failed,fault,ticks,clean=jax.device_get((carry.state.projection.cross_track,
                        carry.state.follower.invalid,carry.state.physical.failed,carry.state.fault,carry.state.tick,carry.clean_blocks))
                    mask=np.zeros(n,bool)
                    for i,c in enumerate(group['cases']):
                        if runtime['planner'].replan_requested(xy=poses[i,:2],goal_xy=group['goals'][i],
                            cross_track_m=float(cross[i]),follower_invalid=bool(invalid[i]),already_replanned=bool(changed[i]),
                            contact=bool(failed[i]),fault=bool(fault[i]),ever_arrived=bool(arrived[i])):
                            count=len(path_list[i].s)
                            path,meta=runtime['planner'].dubins_goal_path(poses[i],group['goals'][i],radius_m=1.8,length_m=(count-1)*.02,spacing_m=.02)
                            if len(path.s)!=count:raise ValueError('numerical path shape mismatch')
                            path_list[i]=path;mask[i]=True;changed[i]=True
                            file=directory/f'replanned_path_group{g:02d}_lane{i:02d}.npz'
                            np.savez_compressed(file,**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
                            decisions.append(dict(case_id=c['case_id'],group=g,lane=i,task_tick=int(ticks[i]),
                                clean_blocks_before=int(clean[i]),physical_pose=poses[i].tolist(),
                                original_goal_xy=group['goals'][i].tolist(),path_file=str(file),**meta))
                    if mask.any():
                        paths=jax.tree.map(lambda *x:j.stack(x),*path_list)
                        before=carry;carry=group['bind'](carry,paths,speeds,j.asarray(mask))
                        jax.block_until_ready(carry);assert_geometry_only(before,carry)
                    save(directory/'replanning_decisions.json',decisions)
                    if bool(j.any(carry.state.follower.numerical_fault)):
                        raise RuntimeError('geometry rebinding numerical fault; prior chunks retained')
                update(phase='running',stage='collecting',group=g,block=block)
                carry,logs=group['chunk'](carry,parameters,paths,speeds,pulses,profiles,floors,zero)
                jax.block_until_ready(carry);logs=jax.device_get(logs);chunks.append(logs)
                # Persist actual output/cost BEFORE inspecting numerical faults.
                flat=flatten(logs);file=directory/'chunks'/f'group{g:02d}_block{block:02d}.npz'
                np.savez_compressed(file,**flat)
                actual+=int(np.asarray(logs['active_tick']).sum());slots+=int(np.asarray(logs['active_tick']).size)
                active=flat['active_tick'];fatal=flat['contact_failure']|flat['nonfinite']|flat['lower_fault']|flat['policy_fault']
                for i in range(n):
                    bad=np.flatnonzero(active[:,i]&fatal[:,i]);before_bad=active[:,i]&(np.arange(len(active))<(bad[0] if len(bad) else len(active)))
                    distance=np.linalg.norm(flat['actual_xy'][:,i]-group['goals'][i],axis=1)
                    arrived[i]|=bool(np.any(before_bad&(distance<=1.)))
                known_contact|=np.any(active&flat['contact_failure'],axis=0)
                state_fault,physical_failed=jax.device_get((carry.state.fault,carry.state.physical.failed))
                boundary_bad=unexplained_boundary_stop(state_fault,physical_failed,known_contact)
                update(active_lower_ticks=actual,allocated_lower_slots=slots)
                if boundary_bad.any():
                    save(directory/'boundary_fault.json',dict(group=g,block=block,case_ids=[group['cases'][i]['case_id'] for i in np.flatnonzero(boundary_bad)],
                        state_fault=np.asarray(state_fault).tolist(),physical_failed=np.asarray(physical_failed).tolist(),known_contact=known_contact.tolist()))
                    raise RuntimeError('latched observation/terminal fault, including inactive-only blocks; raw chunk and cost retained')
                if np.any(active&(flat['nonfinite']|flat['lower_fault']|flat['policy_fault'])):
                    raise RuntimeError('engineering fault: raw chunk and actual cost retained, no negative capability label')
            whole={k:np.concatenate([x[k] for x in chunks],axis=0) for k in chunks[0]};flat=flatten(whole)
            for i,c in enumerate(group['cases']):
                raw={k:x[:,i] for k,x in flat.items()};f=directory/'cases'/f'case_{group["ids"][i]:03d}.npz'
                np.savez_compressed(f,**raw)
                index.append(dict(case_id=c['case_id'],case=c,raw_file=str(f),raw_sha256=digest(f),
                    group=g,lane=i,active_lower_ticks=int(raw['active_tick'].sum()),
                    original_path=str(directory/(c['route_group']+'_original_path.npz')),original_goal_xy=group['goals'][i].tolist()))
            save(directory/'case_index.json',index)
            update(completed_cases=len(index))
        if actual>360000 or slots!=360000 or len(index)!=90:raise ValueError('fixed declared accounting differs')
        save(directory/'initial_audit.json',initial_audit)
        save(directory/'replanning_decisions.json',decisions)
        update(phase='completed',stage='all_raw_saved',completed_cases=90,replans=len(decisions))
    except BaseException as exc:
        update(phase='error',error_type=type(exc).__name__,error=str(exc),active_lower_ticks=actual,allocated_lower_slots=slots)
        (directory/'traceback.txt').write_text(traceback.format_exc());raise
