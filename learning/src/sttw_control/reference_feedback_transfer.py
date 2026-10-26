"""Frozen Actor x feedback factorial on declared new immutable task paths."""
import ast
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .preference_best import atomic_json, digest

TRANSFER_ROLES = ['parent_causal', 'free_GAE', 'parent_feedback_010', 'feedback_010']


def transfer_cases(protocol, contexts):
    if (protocol['schema'] != 'frozen_actor_feedback_geometry_transfer_v1'
            or protocol['roles'] != TRANSFER_ROLES or protocol['bank_indices'] != [2,4]
            or protocol['alphas'] != [0.,.5,1.] or protocol['feedback_gain_rad'] != .1
            or protocol['goal_radius_m'] != .5 or protocol['pulse_duration_s'] != .1
            or protocol['forces_N'] != [0.,-16.,16.,-32.,32.]):
        raise ValueError('fixed four-role transfer protocol required')
    expected = {f"{spec['name']}__bank{bank}" for spec in protocol['contexts'] for bank in [2,4]}
    if set(contexts) != expected or len(expected) != 4:
        raise ValueError('all two geometries and two inherited starts required')
    rng = np.random.default_rng(protocol['seed'])
    onsets = {(spec['name'],force):int(rng.integers(*protocol['onset_upper_tick_range']))*4
              for spec in protocol['contexts'] for force in protocol['forces_N']}
    cases=[]
    for context,c in contexts.items():
        for alpha in protocol['alphas']:
            for force in protocol['forces_N']:
                onset=onsets[c['geometry_name'],force]
                if onset < 0 or onset+20 > 2000:
                    raise ValueError('pulse outside original window')
                event=f'F{force:g}_r0'
                for role in TRANSFER_ROLES:
                    parent=role in ['parent_causal','parent_feedback_010']
                    gain=.1 if role in ['parent_feedback_010','feedback_010'] else 0.
                    cases.append(dict(case_id=f'{context}/a{alpha:g}/{event}/{role}',
                        context=context,route=context,geometry_name=c['geometry_name'],bank_index=c['bank_index'],
                        alpha=alpha,force_N=force,pulse_onset_task_tick=onset,pulse_duration_s=.1,
                        world_direction=c['world_direction'],requested_speed_m_s=c['speed_m_s'],
                        role=role,actor_role='parent_causal' if parent else 'free_GAE',
                        mode_index=1,profile=0,causal_support=True,feedback_gain_rad=gain,
                        event_id=event,replicate=0))
    if len(cases)!=240 or len({c['case_id'] for c in cases})!=240:
        raise ValueError('complete240 unique new episodes required')
    return cases


def build_transfer_path(spec, pose, config, prefix_m):
    from .geometric_path import build_path
    if spec['build_direction'] not in ['left','right']:
        raise ValueError('explicit left/right geometry required')
    geometry=copy.deepcopy(config);pc=geometry['path']
    pc.update(straight_prefix_m=prefix_m,turn_angle_rad=float(np.deg2rad(spec['angle_deg'])),turn_radius_m=spec['radius_m'])
    end=pc['straight_prefix_m']+pc['curvature_transition_length_m']+pc['turn_angle_rad']*pc['turn_radius_m']
    pc['straight_tail_m']=80.-end
    builder='left90_R2' if spec['build_direction']=='left' else 'right90_R2'
    path=build_path(builder,pose,geometry)
    grid=np.linspace(0,80,4001).astype(np.float32)
    arrays=dict(s=grid,xy=np.column_stack([np.interp(grid,np.asarray(path.s),np.asarray(path.xy)[:,i]) for i in [0,1]]).astype(np.float32),
        heading=np.interp(grid,np.asarray(path.s),np.asarray(path.heading)).astype(np.float32),
        curvature=np.interp(grid,np.asarray(path.s),np.asarray(path.curvature)).astype(np.float32),
        turn_end=np.asarray(path.turn_end),goal=np.asarray(path.goal))
    return arrays,geometry['path']


def prepare_transfer(root, *, source, protocol_file):
    from .nominal_initial_confirmation import authenticate_initial_models
    from prepare_short_recovery_campaign import register
    root,source,protocol_file=map(lambda p:Path(p).resolve(),[root,source,protocol_file])
    old=json.loads((source/'plan.json').read_text())
    protocol=json.loads(protocol_file.read_text())
    for name in ['status.json','analysis/status.json']:
        if json.loads((source/name).read_text())['phase']!='completed':
            raise ValueError('completed original real-start reference required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('changed source declaration')
    for file,sha in old['sealed_files'].items():
        if digest(file)!=sha:raise ValueError('changed sealed source')
    module=Path(__file__).parent
    runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
        'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
        'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
        'reference_reentry.py','geometric_path.py']
    for name in runtime:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed frozen runtime: '+name)
    def body(file,symbol):
        return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name==symbol)
    for name,symbol in [('finite_value_evaluation.py','make_evaluation_step'),
                        ('finite_value_evaluation.py','rebind_path_context')]:
        if body(module/name,symbol)!=body(source/'source_at_launch/learning/src/sttw_control'/name,symbol):
            raise ValueError('changed frozen control/initialization kernel')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    authenticate_initial_models(cfg,old['models'])
    models={role:copy.deepcopy(old['models'][role]) for role in ['parent_causal','free_GAE']}
    models['parent_feedback_010']=copy.deepcopy(models['parent_causal'])
    models['feedback_010']=copy.deepcopy(models['free_GAE'])
    for role,model in models.items():
        model['feedback_gain_rad']=.1 if role in ['parent_feedback_010','feedback_010'] else 0.
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 固定Actor反馈新几何准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        shutil.copyfile(protocol_file,root/'protocol.json')
        (root/'paths').mkdir();contexts={}
        for spec in protocol['contexts']:
            origin=old['contexts'][spec['source_base']+'__bank2']
            if digest(origin['path'])!=origin['path_sha256']:raise ValueError('source path frame changed')
            with np.load(origin['path'],allow_pickle=False) as z:
                pose=[*z['xy'][0].tolist(),float(z['heading'][0])]
            arrays,path_config=build_transfer_path(spec,pose,cfg,protocol['straight_prefix_m'])
            pathfile=root/'paths'/(spec['name']+'.npz');np.savez_compressed(pathfile,**arrays)
            theta=pose[2]+(1 if spec['build_direction']=='left' else -1)*np.deg2rad(spec['angle_deg'])/2
            for bank in protocol['bank_indices']:
                inherited=old['contexts'][spec['source_base']+f'__bank{bank}']
                if digest(inherited['initial'])!=inherited['initial_sha256']:raise ValueError('changed full inherited start')
                context=f"{spec['name']}__bank{bank}"
                contexts[context]=dict(initial=inherited['initial'],initial_sha256=inherited['initial_sha256'],
                    path=str(pathfile),path_sha256=digest(pathfile),speed_m_s=spec['speed_m_s'],geometry_name=spec['name'],
                    bank_index=bank,geometry_config=path_config,reference_anchor_pose=pose,
                    rebuild_path_context=True,world_direction=[float(-np.sin(theta)),float(np.cos(theta)),0.])
        cases=transfer_cases(protocol,contexts)
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['protocol.json','common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({c[k]:c[k+'_sha256'] for c in contexts.values() for k in ['initial','path']})
        plan=dict(project='STTW_CONTROL',experiment=protocol['schema'],source_reference_run=str(source),
            source_evaluation_plan_sha256=digest(source/'plan.json'),models=models,contexts=contexts,
            evaluation_roles=TRANSFER_ROLES,declared_cases=240,new_cases=240,cached_cases=0,analysis_cases=240,
            feedback_spec=old['feedback_spec'],horizon_s=10.,lower_dt_s=.005,goal_radius_m=.5,
            additional_report_radii_m=[.1,.2,.3],maximum_active_lower_ticks=480000,maximum_allocated_lower_slots=480000,
            optimizer_updates=0,seed=protocol['seed'],cached_active_lower_ticks=0,cached_allocated_lower_slots=0,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_sha256=sources,sealed_files=sealed,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=protocol['initialization'],information=protocol['events'],stop=protocol['stop'],
            task_rule=old['task_rule'],data_role='New declared joint task geometries/speeds and new seeded pulse times, inherited bank2/4 starts. DEV only, not independent training repeats or a claim of wholly unseen training routes. No nominal profiles, fitting or controller changes.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=240,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 固定Actor反馈新几何240回合',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 新几何240路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
