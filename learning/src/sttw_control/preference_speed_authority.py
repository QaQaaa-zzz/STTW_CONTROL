"""One fixed exploratory rule: scale physical neural speed offset by 1-alpha.

This changes neither the user's preference nor the lower controller. It is not
a safety certificate, and retaining zero NN speed correction is not the same
as enforcing the actual requested speed.
"""
import jax.numpy as j
import numpy as np

AUTHORITY_ROLE = 'preference_speed_authority'


def alpha_context_index(config):
    fields = config['network']['context_fields']
    names = [f['name'] for f in fields]
    if names.count('alpha') != 1:
        raise ValueError('exactly one existing user-alpha context field required')
    i = names.index('alpha')
    if not np.isfinite(fields[i]['scale']) or fields[i]['scale'] <= 0:
        raise ValueError('positive finite alpha scale required')
    return i-len(fields), fields[i]['scale']


def scale_physical_speed(native, alpha):
    """Keep steering; map speed via atanh((1-alpha)*tanh(native_speed))."""
    normalized = (1-alpha)*j.tanh(native[0])
    # Interior inverse only: protect float32 rounding near alpha=0. Endpoints
    # bypass this inverse so saturated native outputs retain exact identity.
    limit = j.nextafter(j.ones_like(normalized), j.zeros_like(normalized))
    interior = j.arctanh(j.clip(normalized, -limit, limit))
    speed = j.where(alpha == 0, native[0], j.where(alpha == 1, 0., interior))
    scaled = native.at[0].set(speed)
    valid = (alpha >= 0) & (alpha <= 1) & j.isfinite(alpha) & j.all(j.isfinite(native))
    return j.where(valid, scaled, j.full_like(native, j.nan))


class PreferenceSpeedActor:
    def __init__(self, actor, config):
        self.actor = actor
        self.alpha_index, self.alpha_scale = alpha_context_index(config)

    def apply(self, parameters, observation):
        # The original observation, including original alpha, reaches the NN.
        native = self.actor.apply(parameters, observation)
        alpha = observation[self.alpha_index]*self.alpha_scale
        return scale_physical_speed(native, alpha)


def make_authority_step(env, actor, *, horizon_s, spec):
    from .reference_task_feedback import make_feedback_evaluation_step
    from .reference_reentry import actor_permission
    source = PreferenceSpeedActor(actor, env.config)
    advance = make_feedback_evaluation_step(env, source, horizon_s=horizon_s, spec=spec)

    def step(carry, parameters, path, speed, pulse, mode):
        state = carry.state
        stopped = state.physical.failed | state.fault | (state.tick >= round(horizon_s/env.cc.dt))
        enabled = actor_permission(state.domain_exit, state.follower.invalid,
                                   carry.clean_blocks, j.bool_(False), stopped)
        obs, _, _ = env.observation(state, path, speed, horizon_s)
        safe = j.where(enabled, obs, j.zeros_like(obs))
        native = actor.apply(parameters['actor'], safe)
        alpha = safe[source.alpha_index]*source.alpha_scale
        end, logs = advance(carry, parameters, path, speed, pulse, mode)
        legal = logs['actor_action_valid']
        logs.update(source_actor_latent=logs['original_actor_latent'],
                    original_actor_latent=j.where(legal[:, None], j.tile(native, (4, 1)), j.nan),
                    speed_authority_user_alpha=j.where(legal, alpha, j.nan))
        return end, logs
    return step


def verify_authority_log(raw, case, config, spec):
    """Independent NumPy reconstruction in physical target coordinates."""
    from .reference_task_feedback import verify_feedback_log
    legal = np.asarray(raw['actor_action_valid'], bool)
    if np.any(legal):
        alpha = float(case['alpha'])
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('user alpha out of range')
        np.testing.assert_allclose(raw['speed_authority_user_alpha'][legal], alpha, atol=1e-7, rtol=0)
        native = np.asarray(raw['original_actor_latent'])[legal]
        source = np.asarray(raw['source_actor_latent'])[legal]
        if not np.all(np.isfinite(native)) or not np.all(np.isfinite(source)):
            raise ValueError('nonfinite legal neural reference')
        c = config['residual_interface']
        def speed(z):
            a = np.tanh(z)
            return a*np.where(a >= 0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s'])
        np.testing.assert_allclose(speed(source[:,0]),(1-alpha)*speed(native[:,0]),atol=2e-7,rtol=2e-6)
        np.testing.assert_array_equal(source[:,1],native[:,1])
        if alpha == 0:
            np.testing.assert_array_equal(source,native)
        if alpha == 1:
            np.testing.assert_array_equal(source[:,0],np.zeros(len(source)))
    verify_feedback_log(dict(raw,original_actor_latent=raw['source_actor_latent']),case,config,spec)


def midpoint_cases(declared):
    import copy
    source=[c for c in declared if c['role']=='feedback_010' and c['alpha']==.5]
    if len(source)!=20 or len({c['case_id'] for c in source})!=20:
        raise ValueError('twenty unique original midpoint events required')
    cases=[]
    for c in source:
        if c['feedback_gain_rad']!=.1 or c['mode_index']!=1 or not c['causal_support']:
            raise ValueError('same fixed feedback and short support required')
        new=copy.deepcopy(c)
        new.update(role=AUTHORITY_ROLE,preference_speed_authority=True,
                   case_id=c['case_id'].rsplit('/',1)[0]+'/'+AUTHORITY_ROLE)
        cases.append(new)
    return cases


def prepare_midpoint(root, source):
    """Seal twenty new midpoint trials and forty exactly reused controls."""
    import ast
    import copy
    import json
    import os
    import shutil
    import subprocess
    import traceback
    from pathlib import Path
    from .preference_best import atomic_json,digest
    from .finite_value_evaluation_report import evaluation_inputs
    from .finite_value_evaluation_run import authenticate_models
    from prepare_short_recovery_campaign import register
    root,source=Path(root).resolve(),Path(source).resolve()
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 固定偏好速度权限准备',root/'prepare_status.json')
    try:
        old=json.loads((source/'plan.json').read_text())
        if old['experiment']!='middle_reference_channel_feedback_v1':
            raise ValueError('complete original channel panel required')
        for name in ['status.json','analysis/status.json']:
            if json.loads((source/name).read_text())['phase']!='completed':raise ValueError('incomplete source')
        if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
            raise ValueError('source plan changed')
        for file,expected in old['sealed_files'].items():
            if digest(file)!=expected:raise ValueError('source input changed: '+file)
        module=Path(__file__).parent
        runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
            'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
            'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
            'reference_reentry.py','geometric_path.py','middle_channel_ablation.py']
        for name in runtime:
            if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
                raise ValueError('changed numerical runtime: '+name)
        def body(file,symbol):
            return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
                        if isinstance(n,ast.FunctionDef) and n.name==symbol)
        for symbol in ['make_evaluation_step','rebind_path_context']:
            if body(module/'finite_value_evaluation.py',symbol)!=body(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py',symbol):
                raise ValueError('changed common execution/initialization kernel')
        cfg=json.loads((source/'common_environment_config.json').read_text())
        alpha_context_index(cfg)
        native=copy.deepcopy(old['models']['steer_only_feedback']);native.pop('source_keep_channels')
        authenticate_models(cfg,dict(free_GAE=native))
        declared,records,indices,_=evaluation_inputs(source,old)
        cases=midpoint_cases(declared)
        cached=[c for c in declared if c['alpha']==.5 and c['role'] in ['feedback_010','steer_only_feedback']]
        if len(cached)!=40:raise ValueError('complete forty midpoint controls required')
        ids={c['case_id'] for c in cached};byid={r['case_id']:r for r in records}
        for c in cached:
            if any(byid[c['case_id']][k]!=v for k,v in c.items()):raise ValueError('cache definition changed')
        cached_records=[r for r in records if r['case_id'] in ids]
        cached_indices=[copy.deepcopy(r) for r in indices if r['case_id'] in ids]
        for item in cached_indices:
            file=(source/item['file']).resolve()
            if digest(file)!=item['sha256']:raise ValueError('cache raw changed')
            item['file']=str(file)
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        models={AUTHORITY_ROLE:native}
        atomic_json(root/'inventory.json',dict(cases=cases))
        atomic_json(root/'authenticated_models.json',models)
        cache=dict(declared=cached,records=cached_records,indices=cached_indices,
                   active_lower_ticks=sum(r['physical_ticks'] for r in cached_records),allocated_lower_slots=80000,
                   source_run=str(source),source_plan_sha256=digest(source/'plan.json'))
        atomic_json(root/'cache_receipt.json',cache)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_preference_speed_authority.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['common_environment_config.json','source_declaration.json','inventory.json','authenticated_models.json','cache_receipt.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
        plan=dict(project='STTW_CONTROL',experiment='preference_speed_authority_midpoint_v1',models=models,
            contexts=old['contexts'],evaluation_roles=['feedback_010','steer_only_feedback',AUTHORITY_ROLE],
            preference_speed_authority=True,declared_cases=20,new_cases=20,cached_cases=40,analysis_cases=60,
            feedback_spec=old['feedback_spec'],horizon_s=10.,lower_dt_s=.005,optimizer_updates=0,
            goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=old['task_rule'],
            maximum_active_lower_ticks=40000,maximum_allocated_lower_slots=40000,
            cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=80000,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization=old['initialization'],information=old['information'],
            stop='Twenty midpoint trials once; engineering error stops. No function, mask, gain or threshold sweep, training or automatic retry.',
            data_role='Post-data fixed R3 opportunity check, same correlated DEV events. Alpha=.5 only, original user preference unchanged. Forty exact historical controls, not new independent trials. No trajectory interpolation. New events only after useful outcome; R3 is a practical baseline, not a Transactions innovation claim.')
        atomic_json(root/'plan.json',plan)
        atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=20,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 固定偏好速度权限中点20',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 固定偏好速度权限shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
