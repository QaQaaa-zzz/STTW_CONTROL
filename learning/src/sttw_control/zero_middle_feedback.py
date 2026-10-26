"""Explicit zero-middle counterfactual; preserve the complete lower control stack."""
import copy
import numpy as np
import jax.numpy as j

ZERO_ROLES = ('zero_middle', 'zero_middle_feedback')


def source_latent(native, zero):
    return j.where(zero, j.zeros_like(native), native)


class SourceActor:
    def __init__(self, actor):
        self.actor = actor

    def apply(self, parameters, observation):
        return source_latent(self.actor.apply(parameters['native'], observation), parameters['zero'])


def zero_cases(cases):
    originals = [c for c in cases if c['role'] == 'free_GAE']
    if len(originals) != 60 or len({c['case_id'] for c in originals}) != 60:
        raise ValueError('complete60 native source events required')
    result = []
    for original in originals:
        if original['mode_index'] != 1 or not original['causal_support'] or original['feedback_gain_rad'] != 0:
            raise ValueError('same short support and no source feedback required')
        for role, gain in [('zero_middle', 0.), ('zero_middle_feedback', .1)]:
            case = copy.deepcopy(original)
            case.update(role=role, actor_role='zero_middle', actor_source_zero=True,
                        feedback_gain_rad=gain, case_id=original['case_id'].rsplit('/', 1)[0]+'/'+role)
            result.append(case)
    return result


def make_zero_middle_step(env, actor, *, horizon_s, spec):
    from .reference_task_feedback import make_feedback_evaluation_step
    from .reference_reentry import actor_permission
    advance = make_feedback_evaluation_step(env, SourceActor(actor), horizon_s=horizon_s, spec=spec)

    def step(carry, parameters, path, speed, pulse, mode):
        s = carry.state
        stopped = s.physical.failed | s.fault | (s.tick >= round(horizon_s/env.cc.dt))
        enabled = actor_permission(s.domain_exit, s.follower.invalid, carry.clean_blocks, j.bool_(False), stopped)
        obs, _, _ = env.observation(s, path, speed, horizon_s)
        native = actor.apply(parameters['actor']['native'], j.where(enabled, obs, j.zeros_like(obs)))
        end, logs = advance(carry, parameters, path, speed, pulse, mode)
        legal = logs['actor_action_valid']
        # Preserve original_actor_latent as the actual frozen NN output. The
        # counterfactual source is separate and shared by execution and logging.
        logs.update(source_actor_latent=logs['original_actor_latent'],
                    original_actor_latent=j.where(legal[:, None], j.tile(native, (4, 1)), j.nan),
                    actor_source_zero=j.repeat(parameters['actor']['zero'], 4))
        return end, logs
    return step


def verify_zero_middle(raw, case, config, spec):
    from .reference_task_feedback import verify_feedback_log
    legal = np.asarray(raw['actor_action_valid'], bool)
    zero = case['actor_source_zero']
    np.testing.assert_array_equal(raw['actor_source_zero'], zero)
    expected = np.zeros_like(raw['original_actor_latent'][legal]) if zero else raw['original_actor_latent'][legal]
    np.testing.assert_array_equal(raw['source_actor_latent'][legal], expected)
    verify_feedback_log(dict(raw, original_actor_latent=raw['source_actor_latent']), case, config, spec)
    if zero:
        np.testing.assert_array_equal(raw['feedback_actor_latent'][legal, 0], 0.)
    if case['feedback_gain_rad'] == 0:
        np.testing.assert_array_equal(raw['feedback_requested_steer_rad'][legal], 0.)


def prepare(root, source):
    """Seal only missing120 arms; native120 remain exact, authenticated cache."""
    import ast
    import json
    import os
    import shutil
    import subprocess
    from pathlib import Path
    from .preference_best import atomic_json, digest
    from .finite_value_evaluation_report import evaluation_inputs
    from .finite_value_evaluation_run import authenticate_models
    from prepare_short_recovery_campaign import register
    root, source = Path(root).resolve(), Path(source).resolve()
    old = json.loads((source/'plan.json').read_text())
    if old['experiment'] != 'frozen_actor_feedback_geometry_transfer_v1':
        raise ValueError('the fixed completed transfer panel is required')
    for name in ['status.json', 'analysis/status.json']:
        if json.loads((source/name).read_text())['phase'] != 'completed':
            raise ValueError('source panel must be complete')
    if digest(source/'plan.json') != json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source plan changed')
    for file, sha in old['sealed_files'].items():
        if digest(file) != sha:raise ValueError('changed source input')
    module = Path(__file__).parent
    runtime = ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
        'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
        'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
        'reference_reentry.py','geometric_path.py']
    for name in runtime:
        if digest(module/name) != old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed numerical runtime: '+name)
    def body(file, symbol):
        return next(ast.dump(n, include_attributes=False) for n in ast.parse(file.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == symbol)
    for name, symbol in [('finite_value_evaluation.py','make_evaluation_step'),
                          ('finite_value_evaluation.py','rebind_path_context')]:
        if body(module/name,symbol) != body(source/'source_at_launch/learning/src/sttw_control'/name,symbol):
            raise ValueError('changed execution/initial-state kernel')
    cfg = json.loads((source/'common_environment_config.json').read_text())
    authenticate_models(cfg,{r:old['models'][r] for r in ['parent_causal','free_GAE']})
    for alias,native in [('feedback_010','free_GAE'),('parent_feedback_010','parent_causal')]:
        for key in ['actor','actor_sha256','checkpoint','checkpoint_sha256','fingerprint','value_contract']:
            if old['models'][alias][key] != old['models'][native][key]:
                raise ValueError('feedback alias changed native identity')
    declared, records, indices, _ = evaluation_inputs(source,old)
    cases = zero_cases(declared)
    cached = [c for c in declared if c['role'] in ['free_GAE','feedback_010']]
    ids = {c['case_id'] for c in cached}
    cached_records = [r for r in records if r['case_id'] in ids]
    cached_indices = [copy.deepcopy(r) for r in indices if r['case_id'] in ids]
    if len(cached) != 120 or len(cached_records) != 120 or len(cached_indices) != 120:
        raise ValueError('exact120 native controls required')
    byid = {r['case_id']:r for r in cached_records}
    for c in cached:
        if any(byid[c['case_id']][k] != v for k,v in c.items()):raise ValueError('cache metadata mismatch')
    for item in cached_indices:
        file = (source/item['file']).resolve()
        if digest(file) != item['sha256']:raise ValueError('cache raw changed')
        item['file'] = str(file)
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 零神经中层强对照准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        models = {}
        for role,gain in [('zero_middle',0.),('zero_middle_feedback',.1)]:
            models[role] = dict(copy.deepcopy(old['models']['free_GAE']), actor_source_zero=True,
                               feedback_gain_rad=gain, weight_role='native authentication/logging only; source latent is explicitly zero')
        atomic_json(root/'inventory.json',dict(cases=cases))
        atomic_json(root/'authenticated_models.json',models)
        cache = dict(declared=cached,records=cached_records,indices=cached_indices,
                     active_lower_ticks=sum(r['physical_ticks'] for r in cached_records),
                     allocated_lower_slots=240000,source_run=str(source),source_plan_sha256=digest(source/'plan.json'))
        atomic_json(root/'cache_receipt.json',cache)
        files = list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py'),Path('learning/cli/evaluate_zero_middle.py')]
        sources = {str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest = root/'source_at_launch'/file.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names = ['common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json','cache_receipt.json']
        sealed = {str(root/name):digest(root/name) for name in names}
        sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
        plan = dict(project='STTW_CONTROL',experiment='zero_neural_middle_fixed_feedback_v1',
            models=models,contexts=old['contexts'],evaluation_roles=['zero_middle','free_GAE','zero_middle_feedback','feedback_010'],
            zero_middle_comparison=True,declared_cases=120,new_cases=120,cached_cases=120,analysis_cases=240,
            feedback_spec=old['feedback_spec'],horizon_s=10.,lower_dt_s=.005,optimizer_updates=0,
            goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=old['task_rule'],
            maximum_active_lower_ticks=240000,maximum_allocated_lower_slots=240000,
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=240000,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=old['initialization'],information=old['information'],
            stop='Run only the missing120 zero-source cases once, stop on engineering fault. No gains, thresholds, paths, supports, training or automatic retries added after outcomes.',
            data_role='Same-event DEV neural-middle presence x fixed-feedback factorial. Complete lower local300, nominal follower, reference filters/permissions, short support and user alpha remain. Zero source has no upper preference selection; it is not pure ECBC or a no-learning system. Native weights are untouched and logged, but do not drive zero-source actions. Exact120 native controls reused. No independent training replicates or algorithm novelty proof.')
        atomic_json(root/'plan.json',plan)
        atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=120,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 零神经中层固定反馈120回合',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 零神经中层120路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise

