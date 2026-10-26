"""Fixed three-arm transfer check on real, shared-ancestor entry histories."""
import copy
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .goal_preview_baseline import GOAL_ROLES
from .preference_best import atomic_json,digest


def transform_path(path, original_pose, new_pose):
    """One fixed rigid task-frame transform; never modify a physical state."""
    old=np.asarray(original_pose);new=np.asarray(new_pose);angle=new[2]-old[2]
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    result={k:np.array(v,copy=True) for k,v in path.items()}
    result['xy']=(path['xy']-old[:2])@rotation.T+new[:2]
    result['heading']=path['heading']+angle
    return result,rotation


def inventory(contexts, *, seed):
    rng=np.random.default_rng(seed);cases=[]
    for name,c in contexts.items():
        # One hidden event per complete context/force, shared across alpha/arms.
        for force in (0.,-16.,16.,-32.,32.):
            onset=int(rng.integers(140,241))*4
            for alpha in (0.,.5,1.):
                for role,(source,mask) in GOAL_ROLES.items():
                    event=f'F{force:g}_r0'
                    cases.append(dict(case_id=f'{name}/a{alpha:g}/{event}/{role}',context=name,
                        route=name,geometry_name=c['geometry_name'],prefix_name=c['prefix_name'],
                        alpha=alpha,force_N=force,pulse_onset_task_tick=onset,pulse_duration_s=.1,
                        world_direction=c['world_direction'],requested_speed_m_s=c['speed_m_s'],
                        role=role,actor_role='free_GAE',mode_index=1,profile=0,causal_support=True,
                        feedback_gain_rad=.1,goal_preview=True,source_role=source,
                        source_keep_channels=mask,event_id=event,replicate=0))
    return cases


def prepare(root, *, source, entries):
    import jax
    import jax.numpy as j
    from .forward_entry_contexts import prefix_inventory,task_relative_state
    from .finite_value_evaluation_run import authenticate_models
    from .contact_only_environment import load_environment_contact_only
    from prepare_short_recovery_campaign import register
    root=Path(root).resolve();source=Path(source).resolve();entries=Path(entries).resolve()
    original=json.loads((source/'plan.json').read_text())
    if original['experiment']!='finite_goal_preview_baseline_v1':raise ValueError('fixed goal source required')
    for base,file in [(source,'status.json'),(source,'analysis/status.json'),(entries,'collection/status.json')]:
        if json.loads((base/file).read_text())['phase']!='completed':raise ValueError('complete evidence required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source declaration changed')
    for name,sha in original['sealed_files'].items():
        if digest(name)!=sha:raise ValueError('frozen source changed')
    receipt=json.loads((entries/'collection/receipt.json').read_text())
    entry_protocol=json.loads((entries/'protocol.json').read_text())
    if entry_protocol['source_plan_sha256']!=digest(source/'plan.json') or entry_protocol['source_run']!=str(source):
        raise ValueError('prefix history belongs to a different frozen source')
    if digest(entries/'collection/full_states.pkl')!=receipt['full_states_sha256'] or digest(entries/'protocol.json')!=receipt['protocol_sha256']:
        raise ValueError('prefix state/protocol changed')
    if len(receipt['contexts'])!=6 or not all(c['valid'] for c in receipt['contexts']):
        raise ValueError('all six fixed prefixes required; no selection or replacements')
    if [{k:c[k] for k in ['name','speed_m_s','turn_angle_deg','lower_ticks']} for c in receipt['contexts']]!=prefix_inventory():
        raise ValueError('prefix layout changed')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    declaration=json.loads((source/'source_declaration.json').read_text())
    native=copy.deepcopy(original['models']['goal_full_feedback']);native.pop('source_keep_channels')
    authenticate_models(cfg,dict(free_GAE=native))
    root.mkdir(exist_ok=False);atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW完整进入历史三臂准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:shutil.copyfile(source/name,root/name)
        for name in ['initial_states','paths']:(root/name).mkdir()
        with (entries/'collection/full_states.pkl').open('rb') as f:states=pickle.load(f)
        env,_=load_environment_contact_only(cfg,declaration['source'],'terminal_contract_v2')
        # Geometry templates already fixed by the completed goal comparison.
        templates={c['geometry_name']:c for c in original['contexts'].values()}
        if set(templates)!={'left55_R1p8_v1p1','right120_R1p6_v1p4'}:raise ValueError('fixed two geometries required')
        contexts={};state_receipts=[]
        for index,item in enumerate(receipt['contexts']):
            state=jax.tree.map(lambda a:j.asarray(a[index]),states)
            if int(state.tick)!=320 or int(state.physical.physical_tick)!=1020 or bool(state.physical.failed|state.fault|state.domain_exit):
                raise ValueError('actual complete prefix endpoint invalid')
            relative=task_relative_state(state)
            retained=[k for k in state.__dataclass_fields__ if k!='tick']
            if not all(getattr(relative,k) is getattr(state,k) for k in retained):raise ValueError('dynamic carry changed')
            file=root/'initial_states'/f'{item["name"]}.pkl'
            with file.open('wb') as f:pickle.dump(SimpleNamespace(state=jax.tree.map(lambda a:np.asarray(a)[None],relative)),f)
            _,pose,_=env.physics.observe(state.physical.data);pose=np.asarray(pose)
            state_receipts.append(dict(prefix_name=item['name'],source_index=index,
                source_full_states_sha256=receipt['full_states_sha256'],source_task_tick=320,
                evaluation_task_tick=0,physical_tick=1020,changed_fields=['tick'],
                all_other_fields_retained_by_identity=retained,initial=str(file),sha256=digest(file)))
            for geometry,template in templates.items():
                with np.load(template['path'],allow_pickle=False) as z:path={k:z[k] for k in z.files}
                transformed,rotation=transform_path(path,template['reference_anchor_pose'],pose)
                pathfile=root/'paths'/f'{item["name"]}__{geometry}.npz';np.savez_compressed(pathfile,**transformed)
                direction=rotation@np.asarray(template['world_direction'])[:2]
                name=f'{item["name"]}__{geometry}'
                contexts[name]=dict(initial=str(file),initial_sha256=digest(file),initial_physical_tick=1020,
                    path=str(pathfile),path_sha256=digest(pathfile),prefix_name=item['name'],
                    geometry_name=geometry,speed_m_s=template['speed_m_s'],
                    world_direction=[*direction.tolist(),0.],reference_anchor_pose=pose.tolist(),
                    rebuild_path_context=True,geometry_config=template['geometry_config'])
        cases=inventory(contexts,seed=20261026113)
        if len(cases)!=540:raise ValueError('twelve contexts times three alpha five force three arms required')
        models={r:dict(copy.deepcopy(native),source_keep_channels=mask) for r,(_,mask) in GOAL_ROLES.items()}
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        atomic_json(root/'initial_state_receipt.json',dict(contexts=state_receipts,physics_steps=0,
            task_geometry='Each real endpoint pose defines the new task frame once by a fixed rigid transform of the two templates; identical for all arms, no online reanchoring or physical edit.'))
        module=Path(__file__).parent;files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_entry_history.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        sealed={str(root/name):digest(root/name) for name in ['common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json','initial_state_receipt.json']}
        sealed.update({str(entries/'collection/full_states.pkl'):receipt['full_states_sha256']})
        sealed.update({str(entries/name):digest(entries/name) for name in ['protocol.json','collection/receipt.json']})
        sealed.update({c[k]:c[k+'_sha256'] for c in contexts.values() for k in ['initial','path']})
        plan=dict(project='STTW_CONTROL',experiment='entry_history_confirmation_v1',models=models,contexts=contexts,
            evaluation_roles=list(GOAL_ROLES),goal_preview_comparison=True,declared_cases=540,new_cases=540,
            cached_cases=0,analysis_cases=540,feedback_spec=original['feedback_spec'],horizon_s=10.,lower_dt_s=.005,
            optimizer_updates=0,goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=original['task_rule'],
            maximum_active_lower_ticks=1080000,maximum_allocated_lower_slots=1080000,
            cached_active_lower_ticks=0,cached_allocated_lower_slots=0,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            entry_collection=str(entries),sealed_files=sealed,source_sha256=sources,
            code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization='Full real prefix PathState; change relative task tick320 to0 only, physical1020 retained. Existing rebind changes projection/follower only; raw/lower/history/filter remain.',
            information=original['information'],stop='Fixed540 once. Engineering error stops; no failed-prefix/event replacement, gain/radius/mask/geometry tuning, training or automatic extension.',
            data_role='Prospective correlated real entry-history transfer DEV; shared bank3 ancestor and one frozen Actor, not independent roots/training repeats or final TEST. One fixed task-frame transform per endpoint. No current performance outcomes used to select arms or events.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=540,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW完整进入历史三臂540',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW完整进入历史三臂shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        import traceback
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
