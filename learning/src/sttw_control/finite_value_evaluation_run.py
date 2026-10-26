"""Authenticated fixed final64 six-role evaluation, separate from training."""
import json
import copy
import ast
import os
import pickle
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from .finite_value_evaluation import endpoint_records, evaluation_inventory, make_evaluation_step
from .preference_best import atomic_json, digest


def prepare_candidate_extension(root, *, source, candidates, supported_candidate=None, feedback_protocol=None, support_comparison=None):
    """Supplement frozen candidates only, retaining the completed90 baseline."""
    from .finite_value_evaluation import candidate_extension_cases, supported_candidate_cases, ROLES
    root=Path(root).resolve();source=Path(source).resolve();candidates=Path(candidates).resolve()
    old=json.loads((source/'plan.json').read_text());bank=json.loads((candidates/'plan.json').read_text())
    for directory in [source,candidates]:
        if json.loads((directory/'status.json').read_text())['phase']!='completed':
            raise ValueError('completed source evaluations required')
        if digest(directory/'plan.json')!=json.loads((directory/'prelaunch_receipt.json').read_text())['plan_sha256']:
            raise ValueError('source declaration changed')
    for file,expected in {**old['sealed_files'],**bank['sealed_files']}.items():
        if digest(file)!=expected:raise ValueError('sealed source input changed')
    if digest(source/'common_environment_config.json')!=digest(candidates/'common_environment_config.json'):
        raise ValueError('candidate environment/config differs from reference source')
    if old['models']['parent_plain']['actor_sha256']!=bank['models']['parent_plain']['actor_sha256']:
        raise ValueError('candidates and reference controls have different parent policies')
    module=Path(__file__).parent
    runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py',
        'path_reference.py','path_command_reward.py','closed_loop_kernel.py','teleop_env.py',
        'actuator.py','controller.py','direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py']
    for name in runtime:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed frozen control runtime')
    core=lambda file:next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='make_evaluation_step')
    if core(module/'finite_value_evaluation.py')!=core(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py'):
        raise ValueError('changed evaluation control kernel')
    feedback_spec=None
    if feedback_protocol is not None:
        from .reference_task_feedback import validate_spec
        feedback_protocol=Path(feedback_protocol).resolve()
        feedback_spec=json.loads(feedback_protocol.read_text());validate_spec(feedback_spec)
        if supported_candidate!='free_GAE' or support_comparison is None:raise ValueError('feedback needs declared free_GAE and exact supported baseline')
    roles=['free_GAE','finite_GAE','free_MC'] if supported_candidate is None else [supported_candidate]
    if supported_candidate is not None and supported_candidate!='free_GAE':raise ValueError('undeclared supported candidate')
    models={r:bank['models'][r] for r in roles}
    cfg=json.loads((source/'common_environment_config.json').read_text());authenticate_models(cfg,models)
    original=json.loads((source/'inventory.json').read_text())['cases']
    cases=candidate_extension_cases(original,roles) if supported_candidate is None else supported_candidate_cases(original,supported_candidate)
    if feedback_spec is not None:
        from .reference_task_feedback import feedback_cases
        cases=feedback_cases(original,feedback_spec)
        models={role:dict(models['free_GAE'],feedback_gain_rad=gain) for role,gain in feedback_spec['gains_rad'].items()}
    n=len(cases)
    evaluation_roles=list(ROLES) if supported_candidate is None else [*ROLES[:3],supported_candidate]
    if feedback_spec is not None:evaluation_roles+=list(feedback_spec['gains_rad'])
    records=json.loads((source/'physical/results.json').read_text());indices=json.loads((source/'physical/case_index.json').read_text())
    ids={c['case_id'] for c in original}
    if len(original)!=90 or len(records)!=90 or len(indices)!=90 or len(ids)!=90 or ids!={c['case_id'] for c in records} or ids!={c['case_id'] for c in indices}:
        raise ValueError('complete90 source records required')
    byid={c['case_id']:c for c in records}
    if any(any(byid[c['case_id']][k]!=v for k,v in c.items()) for c in original):
        raise ValueError('cached event metadata differs from original declaration')
    for item in indices:
        file=(source/item['file']).resolve()
        if digest(file)!=item['sha256']:raise ValueError('cached raw trajectory changed')
        item['file']=str(file)
    cache_active=json.loads((source/'status.json').read_text())['active_lower_ticks']
    cache_slots=json.loads((source/'status.json').read_text())['allocated_lower_slots']
    if feedback_spec is not None:
        comparison=Path(support_comparison).resolve();cp=json.loads((comparison/'plan.json').read_text())
        cs=json.loads((comparison/'status.json').read_text())
        if cs['phase']!='completed' or cs['completed_cases']!=30 or cp.get('supported_candidate')!='free_GAE':raise ValueError('complete same-Actor support baseline required')
        if digest(comparison/'plan.json')!=json.loads((comparison/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('support declaration changed')
        for file,expected in cp['sealed_files'].items():
            if digest(file)!=expected:raise ValueError('support baseline sealed input changed')
        if cp['contexts']!=old['contexts'] or digest(comparison/'common_environment_config.json')!=digest(source/'common_environment_config.json'):raise ValueError('support context/config differs')
        if cp['models']['free_GAE']['actor_sha256']!=bank['models']['free_GAE']['actor_sha256']:raise ValueError('support baseline uses different Actor')
        cd=json.loads((comparison/'inventory.json').read_text())['cases']
        if cd!=supported_candidate_cases(original,'free_GAE'):raise ValueError('support events/permissions differ')
        cr=json.loads((comparison/'physical/results.json').read_text());ci=json.loads((comparison/'physical/case_index.json').read_text())
        if len(cr)!=30 or len(ci)!=30 or {x['case_id'] for x in cr}!={x['case_id'] for x in cd} or {x['case_id'] for x in ci}!={x['case_id'] for x in cd}:raise ValueError('support raw cases incomplete')
        cb={x['case_id']:x for x in cr}
        if any(any(cb[c['case_id']][k]!=v for k,v in c.items()) for c in cd):raise ValueError('support actual metadata differs')
        for item in ci:
            file=(comparison/item['file']).resolve()
            if digest(file)!=item['sha256']:raise ValueError('support cached trajectory changed')
            item['file']=str(file)
        original+=cd;records+=cr;indices+=ci
        cache_active+=cs['active_lower_ticks'];cache_slots+=cs['allocated_lower_slots']
    cached_n=len(original)
    root.mkdir(exist_ok=False)
    from prepare_short_recovery_campaign import register
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 冻结中层同事件补评准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json','protocol.json']:
            shutil.copyfile(source/name,root/name)
        source_status=json.loads((source/'status.json').read_text())
        cache=dict(source_run=str(source),source_plan_sha256=digest(source/'plan.json'),
            declared=original,records=records,indices=indices,
            active_lower_ticks=cache_active,allocated_lower_slots=cache_slots)
        atomic_json(root/'cache_receipt.json',cache)
        if feedback_spec is not None:
            shutil.copyfile(feedback_protocol,root/'feedback_protocol.json')
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        plan=dict(project='STTW_CONTROL',experiment='frozen_candidate_reference_extension',
            models=models,contexts=old['contexts'],evaluation_roles=evaluation_roles,declared_cases=n,new_cases=n,cached_cases=cached_n,analysis_cases=n+cached_n,
            supported_candidate=supported_candidate,feedback_spec=feedback_spec,
            maximum_active_lower_ticks=2000*n,maximum_allocated_lower_slots=2000*n,optimizer_updates=0,
            horizon_s=10.,lower_dt_s=.005,goal_radius_m=old['goal_radius_m'],task_rule=old['task_rule'],
            endpoint_receipt=str(candidates/'authenticated_models.json'),endpoint_receipt_sha256=digest(candidates/'authenticated_models.json'),
            source_reference_run=str(source),candidate_source=str(candidates),
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=cache['allocated_lower_slots'],
            sealed_files={str(root/name):digest(root/name) for name in ['inventory.json','cache_receipt.json','authenticated_models.json','common_environment_config.json','source_declaration.json','protocol.json']+(['feedback_protocol.json'] if feedback_spec is not None else [])},
            source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            data_role=('Supplement three frozen final64 candidates on the SAME30 new-geometry DEV events; exact90 reference trajectories reused, not independent repeats.' if supported_candidate is None else 'DEV complementarity probe: free_GAE is ALWAYS deployed with the unchanged measured short-support gate in this run; exact90 parent controls reused. No new training or independent repeat.'),
            information=old['information'],initialization=old['initialization'],
            stop=f'Only{n} missing candidate episodes; no training/baseline rerun/checkpoint selection/threshold tuning/automatic retry/model adoption.')
        if feedback_spec is not None:
            plan.update(data_role='DEV: two preregistered bounded steering feedback gains on the same frozen free_GAE plus unchanged short support; exact120 controls reused, no new training.',support_comparison=str(comparison))
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=n,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json',f'STTW_CONTROL 冻结中层同事件{n}补评',root/'status.json')
        register(root/'ACTIVE_SHAPE.json',f'STTW_CONTROL 冻结中层{n}路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',error=repr(exc),pid=os.getpid()));raise


def prepare_reference_confirmation(root, *, source, protocol_file):
    """Reuse the parent and execution kernel; only declare new task geometries."""
    from .geometric_path import build_path
    from .finite_value_evaluation import confirmation_inventory
    source = Path(source).resolve(); root = Path(root).resolve()
    protocol_file = Path(protocol_file).resolve()
    protocol = json.loads(protocol_file.read_text())
    old = json.loads((source/'plan.json').read_text())
    if json.loads((source/'status.json').read_text())['phase'] != 'completed':
        raise ValueError('completed source evaluation required')
    if digest(source/'plan.json') != json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source evaluation declaration changed')
    for file, expected in old['sealed_files'].items():
        if digest(file) != expected: raise ValueError('source evaluation input changed')
    runtime = ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py',
        'path_reference.py','path_command_reward.py','closed_loop_kernel.py','teleop_env.py',
        'actuator.py','controller.py','direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py']
    module = Path(__file__).parent
    for name in runtime:
        if digest(module/name) != old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed frozen runtime: '+name)
    def core(file):
        return next(ast.dump(n, include_attributes=False) for n in ast.parse(file.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'make_evaluation_step')
    if core(module/'finite_value_evaluation.py') != core(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py'):
        raise ValueError('changed causal reference execution')
    cfg = json.loads((source/'common_environment_config.json').read_text())
    roles = ['parent_plain', 'parent_causal', 'parent_support']
    models = {role: old['models'][role] for role in roles}
    authenticate_models(cfg, models)
    root.mkdir(exist_ok=False)
    from prepare_short_recovery_campaign import register
    atomic_json(root/'prepare_status.json', dict(phase='running', pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 新几何低速参考支持准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name, root/name)
        shutil.copyfile(protocol_file, root/'protocol.json')
        indices = json.loads((source/'physical/case_index.json').read_text())
        contexts = {}; (root/'paths').mkdir()
        for spec in protocol['contexts']:
            origin = old['contexts'][spec['source_context']]
            if digest(origin['initial']) != origin['initial_sha256']:
                raise ValueError('complete starting state changed')
            item = next(r for r in indices if r['case_id'].startswith(spec['source_context']+'/'))
            pose = [*item['initial_xy'], item['initial_yaw']]
            geometry = copy.deepcopy(cfg)
            pc = geometry['path']; pc.update(straight_prefix_m=protocol['straight_prefix_m'],
                turn_angle_rad=float(np.deg2rad(spec['angle_deg'])), turn_radius_m=spec['radius_m'])
            end = pc['straight_prefix_m'] + pc['curvature_transition_length_m'] + pc['turn_angle_rad']*pc['turn_radius_m']
            pc['straight_tail_m'] = 80. - end
            path = build_path(spec['build_direction'], pose, geometry)
            # Common fixed arc-length samples permit true SIMD across geometries.
            grid = np.linspace(0, 80, 4001).astype(np.float32)
            arrays = dict(s=grid, xy=np.column_stack([np.interp(grid,np.asarray(path.s),np.asarray(path.xy)[:,i]) for i in [0,1]]).astype(np.float32),
                heading=np.interp(grid,np.asarray(path.s),np.asarray(path.heading)).astype(np.float32),
                curvature=np.interp(grid,np.asarray(path.s),np.asarray(path.curvature)).astype(np.float32),
                turn_end=np.asarray(path.turn_end), goal=np.asarray(path.goal))
            path_file = root/'paths'/(spec['name']+'.npz'); np.savez_compressed(path_file, **arrays)
            theta = pose[2] + (1 if spec['build_direction'].startswith('left') else -1)*np.deg2rad(spec['angle_deg'])/2
            contexts[spec['name']] = dict(initial=origin['initial'], initial_sha256=origin['initial_sha256'],
                path=str(path_file), path_sha256=digest(path_file), speed_m_s=spec['speed_m_s'],
                geometry_config=geometry['path'], rebuild_path_context=True,
                world_direction=[float(-np.sin(theta)),float(np.cos(theta)),0.])
        cases = confirmation_inventory(protocol, contexts)
        if len(cases) != 90: raise ValueError('declared90 cases required')
        atomic_json(root/'inventory.json', dict(cases=cases))
        atomic_json(root/'authenticated_models.json', models)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        plan=dict(project='STTW_CONTROL',experiment=protocol['experiment'], source_evaluation=str(source),
            source_evaluation_plan_sha256=digest(source/'plan.json'), models=models,contexts=contexts,
            evaluation_roles=roles,declared_cases=len(cases),new_cases=len(cases),cached_cases=0,
            maximum_active_lower_ticks=180000,maximum_allocated_lower_slots=180000,optimizer_updates=0,
            horizon_s=10.,lower_dt_s=.005,goal_radius_m=protocol['goal_radius_m'],task_rule=protocol['task_rule'],
            endpoint_receipt=str(source/'authenticated_models.json'),endpoint_receipt_sha256=digest(source/'authenticated_models.json'),
            seed=protocol['seed'],data_role='New geometry/low requested speed DEV confirmation; inherited physical initial contexts, correlated branches, no independent training repeats.',
            sealed_files={str(root/f):digest(root/f) for f in ['protocol.json','common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json']},
            source_sha256=sources, code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            information=protocol['information'],stop=protocol['stop'],
            initialization='Retain source task0/physical700 physics, ESO, lower state, actuator/filter state and full history. Recompute only task projection and follower preview for the new immutable path; assign original user alpha. No qpos/qvel injection or settle reset.')
        atomic_json(root/'plan.json', plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=90,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 新几何低速三参考90回合',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 新几何低速90路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',error=repr(exc),pid=os.getpid()))
        raise


def authenticate_models(cfg,models):
    # Authenticate executable Actor leaves against the exact recorded checkpoint.
    import torch
    import jax
    from .finite_value_online import FiniteValueLearner
    from .path_command_policy import make_torch_path_network, export_torch_actor
    from .preference_best import config_hash
    for role, model in models.items():
        for key in ['actor','checkpoint']:
            if digest(model[key]) != model[key+'_sha256']:raise ValueError('endpoint artifact changed: '+role)
        saved = torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
        if role in ['free_GAE','finite_GAE','free_MC']:
            if (saved['next_collection'] != 64 or saved['optimizer_steps'] != [248,256,256]
                    or saved['value_contract'] != model['value_contract']):
                raise ValueError('fixed endpoint optimizer/estimator identity mismatch')
            learner = FiniteValueLearner.restore(cfg,saved)
            if learner.fingerprint() != model['fingerprint']:raise ValueError('endpoint fingerprint mismatch')
            actor = learner.actor
        else:
            if saved['schema'] != 'sttw_dual_survival_boundary_v1' or saved['config_sha256'] != config_hash(cfg):
                raise ValueError('parent native source/config mismatch')
            actor = make_torch_path_network(351,2,cfg)
            actor.load_state_dict(saved['actor'],strict=True)
        with Path(model['actor']).open('rb') as stream:deployed = pickle.load(stream)
        expected = export_torch_actor(actor,cfg)
        if jax.tree.structure(deployed) != jax.tree.structure(expected):raise ValueError('Actor export structure mismatch')
        for a,b in zip(jax.tree.leaves(deployed),jax.tree.leaves(expected)):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


def prepare(root, *, training):
    training = Path(training).resolve(); root = Path(root).resolve()
    status = json.loads((training/'status.json').read_text())
    if status.get('phase') != 'completed' or status.get('completed_updates') != 64:
        raise ValueError('complete final64 training required; no evaluation created')
    p = json.loads((training/'plan.json').read_text())
    sha = digest(training/'plan.json')
    if sha != json.loads((training/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('training declaration changed')
    pair_file = training/'checkpoint_pair_0064.json'
    pair = json.loads(pair_file.read_text())
    models = endpoint_records(p,status,pair,plan_sha256=sha)
    for file, expected in p['files_sha256'].items():
        if digest(training/file) != expected:raise ValueError('sealed training input changed: '+file)
    cfg = json.loads((training/'effective_config.json').read_text())
    cases = evaluation_inventory(json.loads((training/'evaluation_cases.json').read_text()))
    authenticate_models(cfg,models)
    root.mkdir(parents=True,exist_ok=False)
    from prepare_short_recovery_campaign import register
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 价值三臂固定终点300评价准备',root/'prepare_status.json')
    try:
        for old,new in [('effective_config.json','common_environment_config.json'),('source_declaration.json','source_declaration.json')]:
            shutil.copyfile(training/old,root/new)
        atomic_json(root/'inventory.json',dict(cases=cases))
        atomic_json(root/'authenticated_models.json',models)
        files=list(Path('learning/src/sttw_control').glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py'),
            Path('docs/evidence/tolerance_progress_20261022/finite_value_online/EVALUATION_PROTOCOL.md')]
        sources={}
        for file in files:
            sources[str(file.resolve())]=digest(file)
            dest=root/'source_at_launch'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        plan=dict(project='STTW_CONTROL',experiment='finite_value_online_v1_fixed64_evaluation',
            training=str(training),training_plan_sha256=sha,endpoint_receipt=str(pair_file),
            endpoint_receipt_sha256=digest(pair_file),models=models,contexts=p['contexts'],
            declared_cases=300,new_cases=300,cached_cases=0,maximum_active_lower_ticks=600000,
            maximum_allocated_lower_slots=600000,optimizer_updates=0,horizon_s=10.,lower_dt_s=.005,
            seed=p['evaluation_seed'],data_role='DEV; two old route contexts; one matched optimization repeat',
            sealed_files={str(root/name):digest(root/name) for name in ['inventory.json','authenticated_models.json','common_environment_config.json','source_declaration.json']},
            source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            information='Actual pulse only enters physics; causal parent observes current measured roll rate before advance. Original alpha retained.',
            stop='300 fixed cases, no best selection, optimizer, threshold changes, automatic retry or budget extension')
        atomic_json(root/'plan.json',plan)
        atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=300,completed_cases=0,
                    active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 偏好抗扰价值三臂固定终点300评价',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 固定终点300路零积分shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),checked_unix=time.time()))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc),checked_unix=time.time()))
        raise


def execute(root, *, shape_only=False):
    root=Path(root).resolve();previous=json.loads((root/'status.json').read_text())
    if previous['phase']!='prepared':raise ValueError('fresh prepared evaluation required; no implicit restart')
    out=root/('shape' if shape_only else 'physical');out.mkdir(exist_ok=False)
    state=dict(previous,phase='running',stage='authenticating',pid=os.getpid())
    status_file=out/'status.json' if shape_only else root/'status.json'
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(status_file,state)
    update()
    try:
        import jax
        import jax.numpy as j
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate
        from .causal_recovery_relay import CausalRelayState,initial_gate as initial_relay
        from .anticipatory_run import flatten_ticks,physical_outcome
        p=json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('evaluation plan changed')
        for file,sha in {**p['sealed_files'],**p['source_sha256'],p['endpoint_receipt']:p['endpoint_receipt_sha256']}.items():
            if digest(file)!=sha:raise ValueError('sealed evaluation input changed: '+file)
        cases=json.loads((root/'inventory.json').read_text())['cases']
        n=p['declared_cases']
        if len(cases)!=n or (p['experiment']=='finite_value_online_v1_fixed64_evaluation' and n!=300):
            raise ValueError('complete declared evaluation cases required')
        cfg=json.loads((root/'common_environment_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        params={}
        for role,model in p['models'].items():
            for key in ['actor','checkpoint']:
                if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen endpoint changed')
            with Path(model['actor']).open('rb') as stream:params[role]=jax.tree.map(j.asarray,pickle.load(stream))
            if p.get('nominal_feedback_spec') is not None:
                params[role]=dict(actor=params[role],feedback_gain_rad=j.float32(model['feedback_gain_rad']),
                                  feedback_kind=j.int32(model['feedback_kind']))
            elif p.get('feedback_spec') is not None:
                params[role]=dict(actor=params[role],feedback_gain_rad=j.float32(model['feedback_gain_rad']))
            if p.get('zero_middle_comparison'):
                params[role]['actor']=dict(native=params[role]['actor'],zero=j.bool_(model['actor_source_zero']))
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2')
        origins={};paths={};context_receipts={}
        for context,c in p['contexts'].items():
            if digest(c['initial'])!=c['initial_sha256'] or digest(c['path'])!=c['path_sha256']:raise ValueError('complete starting context changed')
            with Path(c['initial']).open('rb') as stream:bank=pickle.load(stream).state
            original=jax.tree.map(lambda x:j.asarray(x[0]),bank)
            if int(original.tick)!=0 or int(original.physical.physical_tick)!=700 or bool(original.physical.failed|original.fault|original.domain_exit):raise ValueError('original complete task0/physical700 required')
            origins[context]=original
            with np.load(c['path'],allow_pickle=False) as z:paths[context]=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
            if c.get('rebuild_path_context'):
                from .finite_value_evaluation import rebind_path_context
                origins[context]=rebind_path_context(original,paths[context],j.float32(c['speed_m_s']),env)
                retained=[name for name in original.__dataclass_fields__ if name not in ['projection','follower']]
                if not all(getattr(original,name) is getattr(origins[context],name) for name in retained):
                    raise ValueError('new path initialization modified retained state')
                context_receipts[context]=dict(retained_fields=retained,retained_objects_identical=True,
                    changed_fields=['projection','follower'],physical_tick=700,task_tick=0,
                    current_follower_invalid=bool(origins[context].follower.invalid))
        if context_receipts:atomic_json(out/'path_context_identity.json',context_receipts)
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        states=[origins[c['context']].replace(alpha=j.float32(c['alpha'])) for c in cases]
        carry=CausalRelayState(stack(states),stack([initial_gate() for _ in cases]),j.zeros(n,j.int32),stack([initial_relay() for _ in cases]))
        batch_paths=stack([paths[c['context']] for c in cases])
        if p.get('nominal_feedback_spec') is not None:
            from .nominal_table_ablation import nominal_profile_key
            tables={}
            for key,item in p['nominal_profiles'].items():
                with np.load(item['file'],allow_pickle=False) as z:tables[key]={k:j.asarray(z[k]) for k in z.files}
            parameters=stack([dict(params[c['role']],nominal_table=tables[nominal_profile_key(c)]) for c in cases])
        else:parameters=stack([params[c['role']] for c in cases])
        speeds=j.asarray([c['requested_speed_m_s'] for c in cases],j.float32)
        modes=j.asarray([c['mode_index'] for c in cases],j.int32)
        pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+c['pulse_duration_s'],
                           *[c['force_N']*d for d in c['world_direction']]] for c in cases],j.float32)
        env.set_log_template(states[0],paths[cases[0]['context']],speeds[0])
        if p.get('zero_middle_comparison'):
            from .zero_middle_feedback import make_zero_middle_step
            one=make_zero_middle_step(env,make_path_actor(cfg),horizon_s=10.,spec=p['feedback_spec'])
        elif p.get('initial_state_confirmation'):
            from .nominal_initial_confirmation import make_initial_confirmation_step
            one=make_initial_confirmation_step(env,make_path_actor(cfg),horizon_s=10.,
                spec=p['feedback_spec'],nominal_spec=p['nominal_feedback_spec'])
        elif p.get('nominal_feedback_spec') is not None:
            from .nominal_reference_feedback import make_nominal_evaluation_step
            one=make_nominal_evaluation_step(env,make_path_actor(cfg),horizon_s=10.,spec=p['nominal_feedback_spec'])
        elif p.get('feedback_spec') is not None:
            from .reference_task_feedback import make_feedback_evaluation_step
            one=make_feedback_evaluation_step(env,make_path_actor(cfg),horizon_s=10.,spec=p['feedback_spec'])
        else:one=make_evaluation_step(env,make_path_actor(cfg),horizon_s=10.)
        @jax.jit
        def chunk(carry,parameters,batch_paths,speeds,pulses,modes):
            return jax.lax.scan(lambda st,_:jax.vmap(one)(st,parameters,batch_paths,speeds,pulses,modes),carry,None,length=50)
        if shape_only:
            result=jax.eval_shape(chunk,carry,parameters,batch_paths,speeds,pulses,modes)
            if result[1]['active_tick'].shape!=(50,n,4):raise ValueError('declared lower layout mismatch')
            atomic_json(out/'receipt.json',dict(lower_shape=[50,n,4],physics_steps=0,optimizer_updates=0))
            update(phase='completed',stage='shape_complete');return
        if json.loads((root/'shape/status.json').read_text())['phase']!='completed':raise ValueError('shape must complete first')
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        with (out/'initial_carry.pkl').open('wb') as stream:pickle.dump(jax.device_get(carry),stream)
        logs=[];start=time.monotonic()
        for block in range(10):
            carry,data=chunk(carry,parameters,batch_paths,speeds,pulses,modes);jax.block_until_ready(carry);host,data=jax.device_get((carry,data));data=flatten_ticks(data);logs.append(data)
            np.savez_compressed(out/f'chunk_{block:02d}.npz',**data)
            with (out/'final_carry.pkl').open('wb') as stream:pickle.dump(host,stream)
            state['active_lower_ticks']+=int(data['active_tick'].sum());state['allocated_lower_slots']+=200*n
            if state['active_lower_ticks']>p['maximum_active_lower_ticks'] or state['allocated_lower_slots']>p['maximum_allocated_lower_slots']:raise ValueError('declared evaluation budget exceeded')
            update(stage='collecting',completed_blocks=block+1,elapsed_s=time.monotonic()-start)
            if np.any(host.state.fault):raise RuntimeError('engineering fault; raw retained, no implicit retry')
            if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
        data={k:np.concatenate([x[k] for x in logs]) for k in logs[0]};(out/'cases').mkdir()
        results=[];index=[]
        for lane,c in enumerate(cases):
            raw={k:v[:,lane] for k,v in data.items()};file=out/'cases'/f'case_{lane:03d}.npz';np.savez_compressed(file,**raw)
            outcome=physical_outcome(raw['active_tick'],raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            pose=np.asarray(env.physics.helpers.pose(states[lane].physical.data))
            results.append(dict(c,case_index=lane,physical_outcome=outcome,physical_ticks=int(raw['active_tick'].sum())))
            index.append(dict(case_id=c['case_id'],file=str(file.relative_to(root)),sha256=digest(file),initial_xy=pose[:2].tolist(),initial_yaw=float(pose[2])))
        atomic_json(out/'results.json',results);atomic_json(out/'case_index.json',index)
        if any(r['physical_outcome']=='unknown' for r in results):raise ValueError('unknown endpoints preserved; evaluation incomplete')
        update(phase='completed',stage='physical_complete_analysis_pending',completed_cases=n)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
