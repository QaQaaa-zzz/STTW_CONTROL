"""Fixed reference-feedback ablations with exact cached controls."""
import ast
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

from .preference_best import atomic_json,digest

SHARED_ROLE='feedback_shared'


def nominal_profile_key(case):
    return f"{case['context']}/a{case.get('nominal_profile_alpha',case['alpha']):g}"


def shared_cases(cases,protocol):
    if (protocol['schema']!='nominal_table_alpha_index_ablation_v1'
            or protocol['executed_alphas']!=[0.,1.] or protocol['nominal_profile_alpha']!=.5
            or protocol['feedback_gain_rad']!=.1 or protocol['goal_radius_m']!=.5):
        raise ValueError('fixed endpoint/shared-table protocol required')
    result=[]
    for original in cases:
        if original['role']!='feedback_nominal' or original['alpha'] not in protocol['executed_alphas']:continue
        if original['feedback_kind']!=1 or original['feedback_gain_rad']!=.1 or original['mode_index']!=1:
            raise ValueError('unchanged nominal feedback and short support required')
        case=copy.deepcopy(original)
        case.update(role=SHARED_ROLE,nominal_profile_alpha=.5,
                    case_id=original['case_id'].rsplit('/',1)[0]+'/'+SHARED_ROLE)
        result.append(case)
    if len(result)!=40 or len({r['case_id'] for r in result})!=40:
        raise ValueError('exact40 endpoint episodes required; alpha.5 has no intervention')
    return result


def weak_feedback_cases(cases,protocol):
    if (protocol['schema']!='fixed_weak_feedback_real_initial_v1'
            or protocol['executed_alphas']!=[0.,.5,1.]
            or protocol['feedback_gain_rad']!=.05 or protocol['goal_radius_m']!=.5):
        raise ValueError('fixed .05 feedback, all original alphas and goal required')
    result=[]
    for original in cases:
        if original['role']!='feedback_010':continue
        if original['feedback_kind']!=3 or original['feedback_gain_rad']!=.1 or original['mode_index']!=1:
            raise ValueError('original constant feedback and short support required')
        case=copy.deepcopy(original)
        case.update(role='feedback_005',feedback_gain_rad=.05,
                    case_id=original['case_id'].rsplit('/',1)[0]+'/feedback_005')
        result.append(case)
    if len(result)!=60 or len({r['case_id'] for r in result})!=60:
        raise ValueError('exact60 full-alpha episodes required')
    return result


def prepare_shared_table(root,*,source,protocol_file):
    return _prepare_ablation(root,source=source,protocol_file=protocol_file,weak=False)


def prepare_weak_feedback(root,*,source,protocol_file):
    return _prepare_ablation(root,source=source,protocol_file=protocol_file,weak=True)


def _prepare_ablation(root,*,source,protocol_file,weak):
    from .finite_value_evaluation_report import evaluation_inputs
    from .nominal_initial_confirmation import authenticate_initial_models
    from prepare_short_recovery_campaign import register
    root=Path(root).resolve();source=Path(source).resolve();protocol_file=Path(protocol_file).resolve()
    old=json.loads((source/'plan.json').read_text());protocol=json.loads(protocol_file.read_text())
    if any(json.loads((source/f).read_text())['phase']!='completed' for f in ['status.json','analysis/status.json']):
        raise ValueError('completed real-initial-state confirmation required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source declaration changed')
    for file,expected in old['sealed_files'].items():
        if digest(file)!=expected:raise ValueError('frozen source input changed')
    module=Path(__file__).parent
    runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
        'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
        'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
        'reference_reentry.py','nominal_reference_feedback.py']
    for name in runtime:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed frozen runtime: '+name)
    def body(file,name):
        return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    for name,symbol in [('finite_value_evaluation.py','make_evaluation_step'),
            ('nominal_initial_confirmation.py','InitialFeedbackActor'),
            ('nominal_initial_confirmation.py','make_initial_confirmation_step')]:
        if body(module/name,symbol)!=body(source/'source_at_launch/learning/src/sttw_control'/name,symbol):
            raise ValueError('changed frozen feedback/physics kernel: '+symbol)
    cfg=json.loads((source/'common_environment_config.json').read_text())
    authenticate_initial_models(cfg,old['models'])
    declared,records,indices,_=evaluation_inputs(source,old)
    if len(declared)!=300:raise ValueError('complete300 source required')
    cases=weak_feedback_cases(declared,protocol) if weak else shared_cases(declared,protocol)
    role='feedback_005' if weak else SHARED_ROLE
    n_new=60 if weak else 40
    n_cache=300 if weak else 200
    label='固定0.05弱参考反馈对照' if weak else '固定公共alpha名义表消融'
    cached=[c for c in declared if c['alpha'] in protocol['executed_alphas']]
    ids={c['case_id'] for c in cached}
    cached_records=[r for r in records if r['case_id'] in ids]
    cached_indices=[copy.deepcopy(r) for r in indices if r['case_id'] in ids]
    if len(cached)!=n_cache or len(cached_records)!=n_cache or len(cached_indices)!=n_cache:
        raise ValueError('complete declared source controls required')
    byid={r['case_id']:r for r in cached_records}
    for c in cached:
        if any(byid[c['case_id']][k]!=v for k,v in c.items()):raise ValueError('cache actual metadata mismatch')
    for item in cached_indices:
        file=(source/item['file']).resolve()
        if digest(file)!=item['sha256']:raise ValueError('cache raw changed')
        item['file']=str(file)
    model=copy.deepcopy(old['models']['feedback_010' if weak else 'feedback_nominal'])
    if weak:model.update(feedback_gain_rad=.05,feedback_kind=3)
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL '+label+'准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        shutil.copyfile(protocol_file,root/'protocol.json')
        atomic_json(root/'inventory.json',dict(cases=cases))
        atomic_json(root/'authenticated_models.json',{role:model})
        atomic_json(root/'nominal_profiles.json',old['nominal_profiles'])
        cache=dict(declared=cached,records=cached_records,indices=cached_indices,
            active_lower_ticks=sum(r['physical_ticks'] for r in cached_records),allocated_lower_slots=n_cache*2000,
            source_run=str(source),source_plan_sha256=digest(source/'plan.json'),
            original_full300_actual_lower_ticks=json.loads((source/'status.json').read_text())['active_lower_ticks'])
        atomic_json(root/'cache_receipt.json',cache)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['common_environment_config.json','source_declaration.json','protocol.json','inventory.json',
               'authenticated_models.json','nominal_profiles.json','cache_receipt.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({p['file']:p['sha256'] for p in old['nominal_profiles'].values()})
        plan=dict(project='STTW_CONTROL',experiment=protocol['schema'],models={role:model},
            contexts=old['contexts'],evaluation_roles=old['evaluation_roles']+[role],
            declared_cases=n_new,new_cases=n_new,cached_cases=n_cache,analysis_cases=n_new+n_cache,
            feedback_spec=old['feedback_spec'],nominal_feedback_spec=old['nominal_feedback_spec'],
            initial_state_confirmation=True,nominal_profiles=old['nominal_profiles'],
            horizon_s=10.,lower_dt_s=.005,maximum_active_lower_ticks=n_new*2000,maximum_allocated_lower_slots=n_new*2000,
            optimizer_updates=0,goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=old['task_rule'],
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=n_cache*2000,
            endpoint_receipt=str(source/'authenticated_models.json'),endpoint_receipt_sha256=digest(source/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=old['initialization'],information=old['information'],stop=protocol['stop'],
            data_role=('Same-event DEV fixed .05 weak-feedback control, full three original alphas. Exact300 previous controls reused; no profile fitting, gain sweep or training.' if weak else 'Same-event DEV mechanism ablation. Original Actor/user alpha0/1 stays fixed; only nominal table selector becomes .5. Alpha.5 is excluded because it has no intervention, not an independent repeated sample. Reuse exact200 endpoint controls from prior300; no profile fitting or gain search.'))
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=n_new,completed_cases=0,
            active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL '+label+str(n_new)+'回合',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL '+label+str(n_new)+'路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
