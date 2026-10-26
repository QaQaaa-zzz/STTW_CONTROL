"""New pulse times/durations for frozen feedback and matched strong controls."""
import ast
import copy
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np

from .preference_best import atomic_json, digest


def confirmation_cases(protocol, contexts, source_cases):
    """Duration pairs share an onset; no controller sees event parameters."""
    if protocol['schema']!='reference_feedback_time_duration_confirmation_v1':
        raise ValueError('explicit confirmation protocol required')
    roles=protocol['roles']
    if roles!=['parent_plain','parent_causal','free_GAE','feedback_005','feedback_010']:
        raise ValueError('all five frozen comparisons required')
    rng=np.random.default_rng(protocol['seed'])
    cases=[]
    for context,spec in contexts.items():
        for alpha in protocol['alphas']:
            excluded={c['pulse_onset_task_tick']//4 for c in source_cases
                      if c['context']==context and c['alpha']==alpha and c['force_N']}
            allowed=[tick for tick in range(*protocol['onset_upper_tick_range']) if tick not in excluded]
            if not allowed:
                raise ValueError('no previously unused onset available')
            for force in protocol['forces_N']:
                onset=int(rng.choice(allowed))*4
                for duration in protocol['durations_s']:
                    ticks=round(duration/.005)
                    if ticks%4 or not np.isclose(ticks*.005,duration) or onset+ticks>2000:
                        raise ValueError('pulse must align to the original upper grid and horizon')
                    event=f'F{force:g}_d{round(duration*1000):03d}_r0'
                    for role in roles:
                        gain={'feedback_005':.05,'feedback_010':.1}.get(role,0.)
                        cases.append(dict(case_id=f'{context}/a{alpha:g}/{event}/{role}',
                            context=context,route=context,alpha=alpha,force_N=force,
                            pulse_onset_task_tick=onset,pulse_duration_s=duration,
                            world_direction=spec['world_direction'],requested_speed_m_s=spec['speed_m_s'],
                            role=role,actor_role='free_GAE' if role in ['feedback_005','feedback_010'] else role,
                            mode_index=0 if role=='parent_plain' else 1,profile=0,
                            causal_support=role!='parent_plain',feedback_gain_rad=gain,
                            event_id=event,replicate=0))
    if len(cases)!=240 or len({r['case_id'] for r in cases})!=240:
        raise ValueError('exact240 newly declared disturbed cases required')
    return cases


def prepare_confirmation(root, *, source, protocol_file):
    from .finite_value_evaluation_run import authenticate_models
    from .finite_value_evaluation_report import evaluation_inputs
    from .reference_task_feedback import validate_spec
    root=Path(root).resolve();source=Path(source).resolve();protocol_file=Path(protocol_file).resolve()
    old=json.loads((source/'plan.json').read_text())
    reference=Path(old['source_reference_run'])
    parent=json.loads((reference/'plan.json').read_text())
    for directory,plan in [(source,old),(reference,parent)]:
        if json.loads((directory/'status.json').read_text())['phase']!='completed':
            raise ValueError('completed source control evaluation required')
        if digest(directory/'plan.json')!=json.loads((directory/'prelaunch_receipt.json').read_text())['plan_sha256']:
            raise ValueError('source declaration changed')
        for file,expected in plan['sealed_files'].items():
            if digest(file)!=expected:raise ValueError('source sealed input changed')
    if json.loads((source/'analysis/status.json').read_text())['phase']!='completed':
        raise ValueError('complete source analysis required')
    protocol=json.loads(protocol_file.read_text());spec=old['feedback_spec'];validate_spec(spec)
    if protocol['goal_radius_m']!=old['goal_radius_m'] or protocol['horizon_s']!=10.:
        raise ValueError('unchanged original goal and horizon required')
    module=Path(__file__).parent
    runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py',
        'path_reference.py','path_command_reward.py','closed_loop_kernel.py','teleop_env.py',
        'actuator.py','controller.py','direct_command_policy.py','path_command_policy.py',
        'reference_reserve_gate.py','reference_task_feedback.py','reference_reentry.py']
    for name in runtime:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed source control runtime: '+name)
    core=lambda file:next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='make_evaluation_step')
    if core(module/'finite_value_evaluation.py')!=core(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py'):
        raise ValueError('changed original execution kernel')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    if digest(source/'common_environment_config.json')!=digest(reference/'common_environment_config.json'):
        raise ValueError('parent/feedback configuration mismatch')
    models={role:copy.deepcopy(parent['models'][role]) for role in ['parent_plain','parent_causal']}
    models['free_GAE']=copy.deepcopy(old['models']['feedback_005'])
    authenticate_models(cfg,models)
    models.update({role:copy.deepcopy(models['free_GAE']) for role in spec['gains_rad']})
    for role,model in models.items():model['feedback_gain_rad']=spec['gains_rad'].get(role,0.)
    declared,records,indices,_=evaluation_inputs(source,old)
    cases=confirmation_cases(protocol,old['contexts'],declared)
    zeros=[c for c in declared if not c['force_N'] and c['role'] in protocol['roles']]
    zeroids={c['case_id'] for c in zeros}
    if len(zeros)!=30 or len(zeroids)!=30:
        raise ValueError('exact30 nominal source trajectories required')
    records=[r for r in records if r['case_id'] in zeroids]
    indices=[r for r in indices if r['case_id'] in zeroids]
    if len(records)!=30 or len(indices)!=30:raise ValueError('complete nominal cache required')
    byid={r['case_id']:r for r in records}
    if any(any(byid[c['case_id']][k]!=v for k,v in c.items()) for c in zeros):
        raise ValueError('nominal actual metadata mismatch')
    for item in indices:
        file=(source/item['file']).resolve()
        if digest(file)!=item['sha256']:raise ValueError('nominal raw changed')
        item['file']=str(file)
    root.mkdir(exist_ok=False)
    from prepare_short_recovery_campaign import register
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 冻结反馈新时刻时长准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        shutil.copyfile(protocol_file,root/'protocol.json')
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        # Cache cost describes these30 reused traces, not the entire historical
        # acquisition campaign, whose sunk cost remains separately recorded.
        cache=dict(declared=zeros,records=records,indices=indices,
            active_lower_ticks=sum(r['physical_ticks'] for r in records),allocated_lower_slots=60000,
            source_run=str(source),source_plan_sha256=digest(source/'plan.json'),
            original_campaign_actual_lower_ticks=json.loads((source/'status.json').read_text())['active_lower_ticks']+old['cached_active_lower_ticks'])
        atomic_json(root/'cache_receipt.json',cache)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        plan=dict(project='STTW_CONTROL',experiment=protocol['schema'],models=models,contexts=old['contexts'],
            evaluation_roles=protocol['roles'],declared_cases=240,new_cases=240,cached_cases=30,analysis_cases=270,
            feedback_spec=spec,supported_candidate='free_GAE',horizon_s=10.,lower_dt_s=.005,
            maximum_active_lower_ticks=480000,maximum_allocated_lower_slots=480000,optimizer_updates=0,
            goal_radius_m=old['goal_radius_m'],task_rule=old['task_rule'],seed=protocol['seed'],
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=cache['allocated_lower_slots'],
            endpoint_receipt=str(source/'authenticated_models.json'),endpoint_receipt_sha256=digest(source/'authenticated_models.json'),
            source_reference_run=str(source),source_parent_reference_run=str(reference),
            source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files={str(root/name):digest(root/name) for name in ['inventory.json','authenticated_models.json',
                'cache_receipt.json','protocol.json','common_environment_config.json','source_declaration.json']},
            source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            information=old['information'],initialization=old['initialization'],stop=protocol['stop'],
            data_role='New pulse times and .06/.14s durations; frozen gains and policies, same inherited physical starts/geometries. DEV confirmation, no independent training/initial-state repeats. Exact30 nominal traces reused.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=240,completed_cases=0,
            active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 冻结反馈新时刻时长240回合',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 新时刻时长240路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',error=repr(exc),pid=os.getpid()));raise
