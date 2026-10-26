"""Prepare a bounded nominal-preserving feedback mechanism comparison."""
import ast
import copy
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .nominal_reference_feedback import NOMINAL_ROLES, nominal_table
from .preference_best import atomic_json, digest


def probe_cases(source_cases, protocol):
    if (protocol['schema']!='nominal_deviation_and_existing_gate_probe_v1'
            or protocol['roles']!=list(NOMINAL_ROLES)
            or protocol['feedback_gain_rad']!=.1
            or protocol['path_scale_m']!=.5 or protocol['heading_scale_rad']!=.3
            or protocol['goal_radius_m']!=.5):
        raise ValueError('both fixed comparisons and declared scales/radius required')
    events=[c for c in source_cases if c['role']=='free_GAE']
    if len(events)!=54 or len({c['case_id'] for c in events})!=54:
        raise ValueError('exact54 original supported Actor events required')
    result=[]
    for event in events:
        if event['mode_index']!=1 or not event['causal_support']:
            raise ValueError('same original short support required')
        for kind,role in enumerate(NOMINAL_ROLES,1):
            row=copy.deepcopy(event)
            row.update(role=role,actor_role='free_GAE',feedback_kind=kind,feedback_gain_rad=.1,
                       case_id=event['case_id'].rsplit('/',1)[0]+'/'+role)
            result.append(row)
    return result


def prepare_nominal_probe(root, *, source, protocol_file):
    from .finite_value_evaluation_run import authenticate_models
    from .finite_value_evaluation_report import evaluation_inputs
    from prepare_short_recovery_campaign import register
    root=Path(root).resolve();source=Path(source).resolve();protocol_file=Path(protocol_file).resolve()
    old=json.loads((source/'plan.json').read_text())
    if any(json.loads((source/name).read_text())['phase']!='completed' for name in ['status.json','analysis/status.json']):
        raise ValueError('completed source physics and analysis required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source declaration changed')
    for file,expected in old['sealed_files'].items():
        if digest(file)!=expected:raise ValueError('source sealed input changed')
    module=Path(__file__).parent
    runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py',
        'path_reference.py','path_command_reward.py','closed_loop_kernel.py','teleop_env.py',
        'actuator.py','controller.py','direct_command_policy.py','path_command_policy.py',
        'reference_reserve_gate.py','reference_task_feedback.py','reference_reentry.py']
    for name in runtime:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed inherited runtime: '+name)
    def core(file):
        return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name=='make_evaluation_step')
    if core(module/'finite_value_evaluation.py')!=core(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py'):
        raise ValueError('changed original control kernel')
    protocol=json.loads(protocol_file.read_text())
    declared,records,indices,_=evaluation_inputs(source,old)
    cases=probe_cases(declared,protocol)
    if len(declared)!=270 or len(records)!=270 or len(indices)!=270:
        raise ValueError('exact270 inherited controls required')
    byid={r['case_id']:r for r in records};index={r['case_id']:copy.deepcopy(r) for r in indices}
    for c in declared:
        if any(byid[c['case_id']][k]!=v for k,v in c.items()):raise ValueError('cache actual metadata mismatch')
        item=index[c['case_id']];file=(source/item['file']).resolve()
        if digest(file)!=item['sha256']:raise ValueError('raw cache changed')
        item['file']=str(file)
    cfg=json.loads((source/'common_environment_config.json').read_text())
    fields={f['name']:f['scale'] for f in cfg['network']['context_fields']}
    if any(fields.get(name)!=scale for name,scale in [('signed_cross_track_error_m',.2),
            ('sin_path_heading_error',1.),('cos_path_heading_error',1.)]):
        raise ValueError('nominal table clipping must match original observation scales')
    base=old['models']['free_GAE'];authenticate_models(cfg,{'free_GAE':base})
    models={role:dict(copy.deepcopy(base),feedback_gain_rad=.1,feedback_kind=kind)
            for kind,role in enumerate(NOMINAL_ROLES,1)}
    initial_file=source/'physical/initial_carry.pkl'
    with initial_file.open('rb') as f:initial=pickle.load(f)
    source_cases=json.loads((source/'inventory.json').read_text())['cases']
    tables={};profiles={}
    for c in declared:
        if c['role']!='free_GAE' or c['force_N']:continue
        key=f"{c['context']}/a{c['alpha']:g}"
        row=byid[c['case_id']];item=index[c['case_id']]
        if row['physical_outcome']!='survived' or row['physical_ticks']!=2000:
            raise ValueError('complete nominal source response required')
        with np.load(item['file'],allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        # The existing feedback log records the actual initial observation before
        # any action. Its sibling role starts from the same sealed full state.
        anchor_id=c['case_id'].rsplit('/',1)[0]+'/feedback_010'
        anchor=index[anchor_id]
        with np.load(anchor['file'],allow_pickle=False) as z:
            if not z['actor_action_valid'][0]:raise ValueError('initial observable anchor unavailable')
            ey0=float(z['feedback_path_error_m'][0]);eh0=float(z['feedback_heading_error_rad'][0])
        lane=next(i for i,x in enumerate(source_cases) if x['context']==c['context'] and x['alpha']==c['alpha'] and x['role']=='free_GAE')
        if int(initial.state.tick[lane])!=0:raise ValueError('true initial state required')
        progress0=float(initial.state.projection.progress[lane])
        np.testing.assert_allclose(ey0,np.clip(float(initial.state.projection.cross_track[lane]),-1,1),atol=1e-6)
        tables[key]=nominal_table(raw['path_progress'],raw['path_cross_track'],raw['path_heading_error'],
            initial_progress=progress0,initial_ey=ey0,initial_eh=eh0)
        profiles[key]=dict(nominal_case=c['case_id'],nominal_raw_sha256=item['sha256'],
            initial_observation_case=anchor_id,initial_observation_raw_sha256=anchor['sha256'],
            initial_carry_sha256=digest(initial_file),initial_lane=lane,
            samples=2001,nominal_collection_lower_ticks=2000,
            min_progress_m=progress0,max_progress_m=float(tables[key]['progress'][-1]))
    if len(tables)!=6:raise ValueError('six original-alpha nominal profiles required')
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 名义增量与同门控反馈准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        shutil.copyfile(protocol_file,root/'protocol.json')
        (root/'profiles').mkdir()
        for i,(key,table) in enumerate(tables.items()):
            file=root/'profiles'/f'profile_{i}.npz';np.savez_compressed(file,**table)
            profiles[key].update(file=str(file),sha256=digest(file))
        atomic_json(root/'nominal_profiles.json',profiles)
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        cache=dict(declared=declared,records=records,indices=list(index.values()),
            active_lower_ticks=sum(r['physical_ticks'] for r in records),allocated_lower_slots=540000,
            source_run=str(source),source_plan_sha256=digest(source/'plan.json'))
        atomic_json(root/'cache_receipt.json',cache)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['common_environment_config.json','source_declaration.json','protocol.json',
               'nominal_profiles.json','inventory.json','authenticated_models.json','cache_receipt.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({p['file']:p['sha256'] for p in profiles.values()})
        plan=dict(project='STTW_CONTROL',experiment=protocol['schema'],models=models,contexts=old['contexts'],
            evaluation_roles=old['evaluation_roles']+list(NOMINAL_ROLES),declared_cases=108,new_cases=108,
            cached_cases=270,analysis_cases=378,feedback_spec=old['feedback_spec'],
            nominal_feedback_spec=protocol,nominal_profiles=profiles,
            horizon_s=10.,lower_dt_s=.005,maximum_active_lower_ticks=216000,maximum_allocated_lower_slots=216000,
            optimizer_updates=0,goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],
            task_rule='First measured entry into original XY goal disk before contact/fault; no terminal speed/heading/hold.',
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=540000,
            endpoint_receipt=str(source/'authenticated_models.json'),endpoint_receipt_sha256=digest(source/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=old['initialization'],information='Original alpha, observable clipped errors and current path progress; zero-only fixed nominal table. No current/future force or onset query. Gate variant reuses original measured-rate gate.',
            stop=protocol['stop'],data_role='Retrospective same-event DEV mechanism probe. Both variants must run nominal as well as disturbed episodes. Nominal acquisition already cost12000 lower ticks, included within270 inherited controls; new routes/alphas/initial states require additional nominal collection. No independent generalization claim.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=108,completed_cases=0,
            active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 名义增量与同门控反馈108回合',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 名义增量108路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
