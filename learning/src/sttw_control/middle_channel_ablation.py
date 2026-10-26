"""Counterfactual neural reference channels under unchanged geometric feedback."""
import copy
import numpy as np
import jax.numpy as j

CHANNEL_ROLES = ('speed_only_feedback', 'steer_only_feedback')
CHANNEL_MASKS = dict(speed_only_feedback=[True, False], steer_only_feedback=[False, True])


def select_channels(native, keep):
    return j.where(keep, native, j.zeros_like(native))


class ChannelActor:
    def __init__(self, actor):
        self.actor = actor

    def apply(self, parameters, observation):
        return select_channels(self.actor.apply(parameters['native'], observation), parameters['keep_channels'])


def channel_cases(cases):
    originals = [c for c in cases if c['role'] == 'feedback_010']
    if len(originals) != 60 or len({c['case_id'] for c in originals}) != 60:
        raise ValueError('complete60 both-channel events required')
    result = []
    for original in originals:
        if (original['mode_index'] != 1 or not original['causal_support']
                or original['feedback_gain_rad'] != .1 or original.get('actor_source_zero', False)):
            raise ValueError('native both-channel fixed feedback/short support required')
        for role, mask in CHANNEL_MASKS.items():
            case = copy.deepcopy(original)
            case.update(role=role, source_keep_channels=list(mask),
                        case_id=original['case_id'].rsplit('/', 1)[0]+'/'+role)
            result.append(case)
    return result


def make_channel_step(env, actor, *, horizon_s, spec):
    from .reference_task_feedback import make_feedback_evaluation_step
    from .reference_reentry import actor_permission
    advance = make_feedback_evaluation_step(env, ChannelActor(actor), horizon_s=horizon_s, spec=spec)

    def step(carry, parameters, path, speed, pulse, mode):
        s = carry.state
        stopped = s.physical.failed | s.fault | (s.tick >= round(horizon_s/env.cc.dt))
        enabled = actor_permission(s.domain_exit, s.follower.invalid, carry.clean_blocks, j.bool_(False), stopped)
        obs, _, _ = env.observation(s, path, speed, horizon_s)
        native = actor.apply(parameters['actor']['native'], j.where(enabled, obs, j.zeros_like(obs)))
        end, logs = advance(carry, parameters, path, speed, pulse, mode)
        legal = logs['actor_action_valid']
        logs.update(source_actor_latent=logs['original_actor_latent'],
                    original_actor_latent=j.where(legal[:, None], j.tile(native, (4, 1)), j.nan),
                    source_keep_channels=j.tile(parameters['actor']['keep_channels'], (4, 1)))
        return end, logs
    return step


def verify_channel_log(raw, case, config, spec):
    from .reference_task_feedback import verify_feedback_log
    legal = np.asarray(raw['actor_action_valid'], bool)
    mask = np.asarray(case['source_keep_channels'], bool)
    if mask.tolist() != CHANNEL_MASKS[case['role']]:raise ValueError('undeclared neural channel mask')
    np.testing.assert_array_equal(raw['source_keep_channels'], np.tile(mask, (len(legal), 1)))
    expected = np.where(mask, raw['original_actor_latent'][legal], 0.)
    np.testing.assert_array_equal(raw['source_actor_latent'][legal], expected)
    verify_feedback_log(dict(raw, original_actor_latent=raw['source_actor_latent']), case, config, spec)


def prepare(root, source):
    import ast
    import json
    import os
    import shutil
    import subprocess
    import traceback
    from pathlib import Path
    from .preference_best import atomic_json, digest
    from .finite_value_evaluation_report import evaluation_inputs
    from .finite_value_evaluation_run import authenticate_models
    from .zero_middle_feedback import verify_alias_identity
    from prepare_short_recovery_campaign import register
    root, source = Path(root).resolve(), Path(source).resolve()
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 神经参考通道留出准备',root/'prepare_status.json')
    try:
        old = json.loads((source/'plan.json').read_text())
        if old['experiment'] != 'zero_neural_middle_fixed_feedback_v1':
            raise ValueError('completed zero/both-source comparison required')
        for name in ['status.json','analysis/status.json']:
            if json.loads((source/name).read_text())['phase'] != 'completed':raise ValueError('incomplete source')
        if digest(source/'plan.json') != json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
            raise ValueError('changed source plan')
        for file, sha in old['sealed_files'].items():
            if digest(file) != sha:raise ValueError('changed source input')
        module = Path(__file__).parent
        runtime = ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
            'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
            'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
            'reference_reentry.py','geometric_path.py','zero_middle_feedback.py']
        for name in runtime:
            if digest(module/name) != old['source_sha256'][str((module/name).resolve())]:
                raise ValueError('changed numerical runtime: '+name)
        def body(file, symbol):
            return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
                        if isinstance(n,ast.FunctionDef) and n.name==symbol)
        for symbol in ['make_evaluation_step','rebind_path_context']:
            if body(module/'finite_value_evaluation.py',symbol) != body(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py',symbol):
                raise ValueError('changed common execution/initialization kernel')
        cfg = json.loads((source/'common_environment_config.json').read_text())
        native_root = Path(old['source_reference_run'])
        if digest(native_root/'plan.json') != old['source_evaluation_plan_sha256']:
            raise ValueError('changed native source plan')
        native_plan = json.loads((native_root/'plan.json').read_text())
        native = native_plan['models']['free_GAE']
        authenticate_models(cfg,dict(free_GAE=native))
        verify_alias_identity(old['models']['zero_middle_feedback'],native)
        declared, records, indices, _ = evaluation_inputs(source,old)
        cases = channel_cases(declared)
        cached = [c for c in declared if c['role'] in ['zero_middle_feedback','feedback_010']]
        if any(c['feedback_gain_rad']!=.1 or c['mode_index']!=1 or not c['causal_support'] for c in cached):
            raise ValueError('same fixed feedback and support required in both cached controls')
        ids = {c['case_id'] for c in cached}
        cached_records = [r for r in records if r['case_id'] in ids]
        cached_indices = [copy.deepcopy(r) for r in indices if r['case_id'] in ids]
        if len(cached)!=120 or len(cached_records)!=120 or len(cached_indices)!=120:
            raise ValueError('exact120 complete both/none controls required')
        byid = {r['case_id']:r for r in cached_records}
        for c in cached:
            if any(byid[c['case_id']][k]!=v for k,v in c.items()):raise ValueError('cache metadata mismatch')
        for item in cached_indices:
            file = (source/item['file']).resolve()
            if digest(file)!=item['sha256']:raise ValueError('cache raw changed')
            item['file']=str(file)
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        models = {role:dict(copy.deepcopy(native),source_keep_channels=mask,feedback_gain_rad=.1)
                  for role,mask in CHANNEL_MASKS.items()}
        atomic_json(root/'inventory.json',dict(cases=cases))
        atomic_json(root/'authenticated_models.json',models)
        cache=dict(declared=cached,records=cached_records,indices=cached_indices,
                   active_lower_ticks=sum(r['physical_ticks'] for r in cached_records),allocated_lower_slots=240000,
                   source_run=str(source),source_plan_sha256=digest(source/'plan.json'))
        atomic_json(root/'cache_receipt.json',cache)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py'),Path('learning/cli/evaluate_middle_channels.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json','cache_receipt.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
        plan=dict(project='STTW_CONTROL',experiment='middle_reference_channel_feedback_v1',models=models,
            contexts=old['contexts'],evaluation_roles=['zero_middle_feedback',*CHANNEL_ROLES,'feedback_010'],
            channel_ablation=True,declared_cases=120,new_cases=120,cached_cases=120,analysis_cases=240,
            feedback_spec=old['feedback_spec'],horizon_s=10.,lower_dt_s=.005,optimizer_updates=0,
            goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=old['task_rule'],
            maximum_active_lower_ticks=240000,maximum_allocated_lower_slots=240000,
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=240000,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=old['initialization'],information=old['information'],
            stop='Only missing120 speed-only/steer-only full episodes once; stop on engineering fault. No gain/threshold/support/route sweep, training, extension or automatic retry.',
            data_role='DEV neural reference channel2x2 under fixed .10 geometric feedback. Same original alpha, full task0 state, lower local300, nominal follower, short support, permissions and filters. Mask only native neural source channels before feedback; steering feedback and safety speed support remain. Both/none controls reused exactly, not120 new independent repeats. Channel interactions do not establish a novel learning algorithm.')
        atomic_json(root/'plan.json',plan)
        atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=120,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 固定反馈神经通道留出120',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 神经通道留出120路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
