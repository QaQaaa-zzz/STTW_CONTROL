"""Passive observation history logging; no detector actuation or new policy."""
import copy


def attach_sensor_log(advance, env, horizon_s):
    import jax.numpy as j
    from .causal_reference_baseline import (LATEST_ESO, PREVIOUS_ESO,
        PREVIOUS_FRAME_MASK, LATEST_FRAME_MASK)
    def step(carry, parameters, path, speed, pulse, mode):
        obs, _, fault = env.observation(carry.state, path, speed, horizon_s)
        latest, previous = obs[LATEST_ESO], obs[PREVIOUS_ESO]
        valid = (~fault & j.all(j.isfinite(obs)) & (obs[PREVIOUS_FRAME_MASK] > .5)
                 & (obs[LATEST_FRAME_MASK] > .5) & (j.abs(latest) < 5)
                 & (j.abs(previous) < 5))
        end, logs = advance(carry, parameters, path, speed, pulse, mode)
        # Pure observation values never enter advance, gate or action.
        logs.update(probe_eso_latest=j.repeat(latest, 4),
                    probe_eso_previous=j.repeat(previous, 4),
                    probe_eso_signal=j.repeat(-(latest-previous), 4),
                    probe_eso_history_valid=j.repeat(valid, 4))
        return end, logs
    return step


def prepare(root, source):
    import ast, json, os, shutil, subprocess
    from pathlib import Path
    from .preference_best import atomic_json, digest
    from .finite_value_evaluation_run import authenticate_models
    from prepare_short_recovery_campaign import register
    root, source = Path(root).resolve(), Path(source).resolve()
    old = json.loads((source/'plan.json').read_text())
    if old['experiment'] != 'entry_history_confirmation_v1':
        raise ValueError('original complete entry-history source required')
    for file in ['status.json', 'analysis/status.json']:
        if json.loads((source/file).read_text())['phase'] != 'completed':
            raise ValueError('completed original source required')
    if digest(source/'plan.json') != json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source plan changed')
    for file, sha in old['sealed_files'].items():
        if digest(file) != sha: raise ValueError('source input changed')
    module = Path(__file__).parent
    numeric = ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py',
        'path_reference.py','path_command_reward.py','closed_loop_kernel.py','teleop_env.py',
        'actuator.py','controller.py','direct_command_policy.py','path_command_policy.py',
        'reference_reserve_gate.py','reference_task_feedback.py','reference_reentry.py',
        'geometric_path.py','middle_channel_ablation.py','causal_reference_baseline.py']
    for name in numeric:
        if digest(module/name) != old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('numerical source changed: '+name)
    def tree(file, name):
        return next(ast.dump(n, include_attributes=False) for n in ast.parse(file.read_text()).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name)
    for file, names in [('finite_value_evaluation.py',['make_evaluation_step','rebind_path_context']),
                        ('goal_preview_baseline.py',['goal_follower','GoalPreviewEnv','make_goal_step'])]:
        snapshot=source/'source_at_launch/learning/src/sttw_control'/file
        for name in names:
            if tree(module/file,name) != tree(snapshot,name):
                raise ValueError('original numerical kernel changed: '+name)
    original=json.loads((source/'inventory.json').read_text())['cases']
    cases=[copy.deepcopy(c) for c in original if c['role']=='goal_zero_feedback' and c['alpha']==.5]
    if len(cases)!=60 or len({c['case_id'] for c in cases})!=60:
        raise ValueError('all60 zero-neural alpha0.5 cases required')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    model=copy.deepcopy(old['models']['goal_zero_feedback'])
    native=copy.deepcopy(model);native.pop('source_keep_channels')
    authenticate_models(cfg,dict(free_GAE=native))
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW被动ESO信号60基线准备',root/'prepare_status.json')
    for name in ['common_environment_config.json','source_declaration.json']:
        shutil.copyfile(source/name,root/name)
    atomic_json(root/'inventory.json',dict(cases=cases))
    models=dict(goal_zero_feedback=model)
    atomic_json(root/'authenticated_models.json',models)
    sources={}
    for file in list(module.glob('*.py'))+[Path('learning/cli/evaluate_passive_sensor.py')]:
        sources[str(file.resolve())]=digest(file)
        dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd())
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    sealed={str(root/name):digest(root/name) for name in ['common_environment_config.json',
        'source_declaration.json','inventory.json','authenticated_models.json']}
    sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
    plan=dict(copy.deepcopy(old),experiment='passive_sensor_probe_v1',models=models,
        evaluation_roles=['goal_zero_feedback'],passive_sensor_probe=True,
        declared_cases=60,new_cases=60,cached_cases=0,analysis_cases=60,
        maximum_active_lower_ticks=120000,maximum_allocated_lower_slots=120000,
        cached_active_lower_ticks=0,cached_allocated_lower_slots=0,
        source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
        sealed_files=sealed,source_sha256=sources,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        information='Passive observation history only; original alpha0.5 zero-neural control and events retained. Existing normalized ESO threshold0.015 used only in later timing audit; no new action/gate/threshold fitting.',
        stop='Exactly60 once, no actuation, new threshold, onset scan, optimizer or automatic extension. Engineering error retained and stops.',
        data_role='Exploratory missing-telemetry collection on seen related DEV; not independent confirmation, new control gain or exact replay gate.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=60,completed_cases=0,
        active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0))
    register(root/'ACTIVE_RUN.json','STTW被动ESO信号60基线诊断',root/'status.json')
    register(root/'ACTIVE_SHAPE.json','STTW被动ESO信号shape',root/'shape/status.json')
