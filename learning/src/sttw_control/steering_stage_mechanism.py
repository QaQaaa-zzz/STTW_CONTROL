"""Privileged onset-stage intervention for mechanism diagnosis, not deployment."""
import copy
import numpy as np

STAGE_ROLES=('neural_before_only','neural_after_only')


def stage_cases(original):
    events=[c for c in original if c['role']=='goal_steer_feedback']
    if len(events)!=180:raise ValueError('all180 original steering events required')
    cases=[]
    for event in events:
        if event['pulse_onset_task_tick']%4:raise ValueError('upper boundary onset required')
        for role in STAGE_ROLES:
            c=copy.deepcopy(event)
            c.update(role=role,stage_mechanism=True,
                     stage_late_neural=role=='neural_after_only',
                     case_id=c['case_id'].rsplit('/',1)[0]+'/'+role)
            c.pop('source_keep_channels')
            cases.append(c)
    return cases


def make_stage_step(env,actor,*,horizon_s,spec):
    import jax.numpy as j
    from .goal_preview_baseline import make_goal_step
    advance=make_goal_step(env,actor,horizon_s=horizon_s,spec=spec)
    def step(carry,parameters,path,speed,pulse,mode):
        before=carry.state.tick<parameters['switch_task_tick']
        keep=j.where(before,~parameters['stage_late_neural'],parameters['stage_late_neural'])
        masked=dict(parameters,actor=dict(native=parameters['actor']['native'],
                                         keep_channels=j.stack([j.bool_(False),keep])))
        end,logs=advance(carry,masked,path,speed,pulse,mode)
        logs.update(stage_before=j.repeat(before,4),
                    stage_switch_task_tick=j.repeat(parameters['switch_task_tick'],4))
        return end,logs
    return step


def expected_mask(raw,case):
    active=np.asarray(raw['active_tick'],bool)
    before=np.asarray(raw['physical_tick'])<case['pulse_onset_task_tick']
    np.testing.assert_array_equal(raw['stage_before'][active],before[active])
    np.testing.assert_array_equal(raw['stage_switch_task_tick'][active],np.full(active.sum(),case['pulse_onset_task_tick']))
    keep=np.where(before,not case['stage_late_neural'],case['stage_late_neural'])
    return np.column_stack([np.zeros(len(keep),bool),keep])


def verify_stage_log(raw,case,config,spec,path):
    from .goal_preview_baseline import verify_goal_log
    return verify_goal_log(raw,case,config,spec,path,expected_keep=expected_mask(raw,case))


def prepare(root,source):
    import ast,json,os,shutil,subprocess,traceback
    from pathlib import Path
    from .preference_best import atomic_json,digest
    from .finite_value_evaluation_report import evaluation_inputs
    from .finite_value_evaluation_run import authenticate_models
    from prepare_short_recovery_campaign import register
    root=Path(root).resolve();source=Path(source).resolve()
    old=json.loads((source/'plan.json').read_text())
    if old['experiment']!='entry_history_confirmation_v1':raise ValueError('complete fixed entry-history source required')
    for name in ['status.json','analysis/status.json']:
        if json.loads((source/name).read_text())['phase']!='completed':raise ValueError('completed source required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source plan changed')
    for file,sha in old['sealed_files'].items():
        if digest(file)!=sha:raise ValueError('source input changed')
    module=Path(__file__).parent
    numeric=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
        'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
        'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
        'reference_reentry.py','geometric_path.py','middle_channel_ablation.py']
    for name in numeric:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:raise ValueError('numerical source changed: '+name)
    def tree(file,name):
        return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    for file,names in [('finite_value_evaluation.py',['make_evaluation_step','rebind_path_context']),
                       ('goal_preview_baseline.py',['goal_follower','GoalPreviewEnv','make_goal_step'])]:
        snapshot=source/'source_at_launch/learning/src/sttw_control'/file
        for name in names:
            if tree(module/file,name)!=tree(snapshot,name):raise ValueError('original numerical kernel changed: '+name)
    cfg=json.loads((source/'common_environment_config.json').read_text())
    native=copy.deepcopy(old['models']['goal_steer_feedback']);native.pop('source_keep_channels')
    authenticate_models(cfg,dict(free_GAE=native))
    declared,records,indices,_=evaluation_inputs(source,old)
    selected={'goal_zero_feedback','goal_steer_feedback'}
    cached=[c for c in declared if c['role'] in selected];ids={c['case_id'] for c in cached}
    if len(cached)!=360 or len(ids)!=360:raise ValueError('all360 original endpoint cases required')
    byid={r['case_id']:r for r in records}
    if any(any(byid[c['case_id']][k]!=v for k,v in c.items()) for c in cached):raise ValueError('cached metadata differs')
    cached_records=[r for r in records if r['case_id'] in ids]
    cached_indices=[copy.deepcopy(i) for i in indices if i['case_id'] in ids]
    for item in cached_indices:
        file=(source/item['file']).resolve()
        if digest(file)!=item['sha256']:raise ValueError('cached raw changed')
        item['file']=str(file)
    cases=stage_cases(declared)
    root.mkdir(exist_ok=False);atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW神经转向前后阶段机制准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:shutil.copyfile(source/name,root/name)
        models={role:dict(copy.deepcopy(native),source_keep_channels=[False,True],stage_late_neural=role=='neural_after_only') for role in STAGE_ROLES}
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        cache=dict(declared=cached,records=cached_records,indices=cached_indices,
            active_lower_ticks=sum(r['physical_ticks'] for r in cached_records),allocated_lower_slots=720000,
            source_run=str(source),source_plan_sha256=digest(source/'plan.json'))
        atomic_json(root/'cache_receipt.json',cache)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_steering_stage.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json','cache_receipt.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
        plan=dict(project='STTW_CONTROL',experiment='steering_stage_mechanism_v1',models=models,contexts=old['contexts'],
            evaluation_roles=['goal_zero_feedback',*STAGE_ROLES,'goal_steer_feedback'],goal_preview_comparison=True,
            steering_stage_mechanism=True,declared_cases=360,new_cases=360,cached_cases=360,analysis_cases=720,
            feedback_spec=old['feedback_spec'],horizon_s=10.,lower_dt_s=.005,optimizer_updates=0,
            goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=old['task_rule'],
            maximum_active_lower_ticks=720000,maximum_allocated_lower_slots=720000,
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=720000,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=old['initialization'],information='Exact planned onset tick is privileged experimental mask intervention only; it is not an actor input or deployable disturbance detector. Same synthetic switch also occurs in no-force branches.',
            stop='All180 fixed events per hybrid once; no onset scan, state/filter/history reset, event selection, full-network family, training or automatic extension.',
            data_role='Four-stage-combination causal mechanism diagnosis under same complete task0, not a proposed deployable switch or independent final TEST. Each prefix generates its own real onset state; no state swap. Shared bank3 ancestor, one Actor;360 historical endpoints are not new trials. Numerical replay variation accepted; no new exact replay gate.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=360,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW神经转向前后机制360新＋360历史',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW神经转向前后机制shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc());atomic_json(root/'prepare_status.json',dict(phase='error',error=repr(exc),pid=os.getpid()));raise
