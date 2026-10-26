"""Fixed nominal tables on pre-existing real prepared physical starts."""
import ast
import copy
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .preference_best import atomic_json,digest

ROLES=('parent_plain','parent_causal','free_GAE','feedback_010','feedback_nominal')


def authenticate_initial_models(config,models):
    from .finite_value_evaluation_run import authenticate_models
    authenticate_models(config,{r:models[r] for r in ['parent_plain','parent_causal','free_GAE']})
    # Native checkpoint format belongs to free_GAE, not to its runtime aliases.
    for role in ['feedback_010','feedback_nominal']:
        for key in ['actor','actor_sha256','checkpoint','checkpoint_sha256','fingerprint','value_contract']:
            if models[role][key]!=models['free_GAE'][key]:
                raise ValueError('feedback alias differs from authenticated native Actor: '+role+'/'+key)


def initial_cases(protocol, contexts, previous_cases):
    if (protocol['schema']!='nominal_feedback_real_initial_confirmation_v1'
            or protocol['roles']!=list(ROLES) or protocol['bank_indices']!=[2,4]
            or protocol['alphas']!=[0.,.5,1.] or protocol['forces_N']!=[0.,-16.,16.,-32.,32.]
            or protocol['pulse_duration_s']!=.1 or protocol['goal_radius_m']!=.5):
        raise ValueError('fixed initial-state confirmation protocol required')
    rng=np.random.default_rng(protocol['seed']);onsets={};cases=[]
    for context, spec in contexts.items():
        base=spec['nominal_profile_context']
        for alpha in protocol['alphas']:
            excluded={c['pulse_onset_task_tick']//4 for c in previous_cases if
                      c['context']==base and c['alpha']==alpha and c['force_N']}
            allowed=[v for v in range(*protocol['onset_upper_tick_range']) if v not in excluded]
            if not allowed:raise ValueError('unused onset required')
            for force in protocol['forces_N']:
                key=(base,alpha,force)
                if key not in onsets:onsets[key]=int(rng.choice(allowed))*4
                onset=onsets[key]
                event=f'F{force:g}_r0'
                for role in ROLES:
                    kind=1 if role=='feedback_nominal' else 3 if role=='feedback_010' else 0
                    cases.append(dict(case_id=f'{context}/a{alpha:g}/{event}/{role}',context=context,route=context,
                        nominal_profile_context=base,bank_index=spec['bank_index'],alpha=alpha,force_N=force,
                        pulse_onset_task_tick=onset,pulse_duration_s=.1,world_direction=spec['world_direction'],
                        requested_speed_m_s=spec['speed_m_s'],role=role,
                        actor_role=role if role in ['parent_plain','parent_causal'] else 'free_GAE',
                        mode_index=0 if role=='parent_plain' else 1,profile=0,causal_support=role!='parent_plain',
                        feedback_gain_rad=.1 if kind else 0.,feedback_kind=kind,event_id=event,replicate=0))
    if len(cases)!=300 or len({c['case_id'] for c in cases})!=300:
        raise ValueError('exact300 unique cases required')
    return cases


def make_initial_confirmation_step(env,actor,*,horizon_s,spec,nominal_spec):
    """Reuse BOTH frozen feedback kernels; this adapter only selects their kind."""
    import jax
    import jax.numpy as j
    from .reference_task_feedback import make_feedback_evaluation_step
    from .nominal_reference_feedback import make_nominal_evaluation_step,lookup_nominal
    constant=make_feedback_evaluation_step(env,actor,horizon_s=horizon_s,spec=spec)
    nominal=make_nominal_evaluation_step(env,actor,horizon_s=horizon_s,spec=nominal_spec)
    def step(carry,parameters,path,speed,pulse,mode):
        def constant_branch(_):
            end,logs=constant(carry,parameters,path,speed,pulse,mode)
            active=logs['active_tick']
            _,_,covered=lookup_nominal(carry.state.projection.progress,parameters['nominal_table'])
            logs.update(feedback_effective_gain_rad=j.where(active,parameters['feedback_gain_rad'],j.nan),
                nominal_progress_m=j.where(active,carry.state.projection.progress,j.nan),
                nominal_profile_covered=active&covered,
                nominal_path_error_m=j.where(active,0.,j.nan),nominal_heading_error_rad=j.where(active,0.,j.nan))
            return end,logs
        return jax.lax.cond(parameters['feedback_kind']==1,
            lambda _:nominal(carry,parameters,path,speed,pulse,mode),constant_branch,None)
    return step


def verify_initial_feedback(raw,case,config,plan):
    from .nominal_reference_feedback import verify_nominal_feedback
    from .reference_task_feedback import verify_feedback_log
    profile=plan['nominal_profiles'][f"{case['context']}/a{case['alpha']:g}"]
    with np.load(profile['file'],allow_pickle=False) as z:table={k:z[k] for k in z.files}
    if case['feedback_kind']==1:
        verify_nominal_feedback(raw,case,config,plan['nominal_feedback_spec'],table)
    else:
        verify_feedback_log(raw,case,config,plan['feedback_spec'])
    active=np.asarray(raw['active_tick'],bool)
    ticks=np.flatnonzero(active&(raw['physical_tick']%4==0))
    later=ticks[ticks>0]
    np.testing.assert_allclose(raw['nominal_progress_m'][later],raw['path_progress'][later-1],atol=2e-6)
    if len(ticks):
        np.testing.assert_allclose(raw['nominal_progress_m'][ticks[0]],
            plan['contexts'][case['context']]['initial_progress_m'],atol=2e-6)


def prepare_initial_confirmation(root,*,source,protocol_file):
    import jax
    import jax.numpy as j
    from .contact_only_environment import load_environment_contact_only
    from .geometric_path import Path as GeometricPath
    from prepare_short_recovery_campaign import register
    root=Path(root).resolve();source=Path(source).resolve();protocol_file=Path(protocol_file).resolve()
    old=json.loads((source/'plan.json').read_text());protocol=json.loads(protocol_file.read_text())
    if any(json.loads((source/f).read_text())['phase']!='completed' for f in ['status.json','analysis/status.json']):
        raise ValueError('completed nominal mechanism probe required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source plan changed')
    for file,expected in old['sealed_files'].items():
        if digest(file)!=expected:raise ValueError('frozen source input changed')
    module=Path(__file__).parent
    runtime=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
        'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
        'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
        'reference_reentry.py','nominal_reference_feedback.py']
    for name in runtime:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:
            raise ValueError('changed frozen source runtime: '+name)
    def core(file):
        return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body
                    if isinstance(n,ast.FunctionDef) and n.name=='make_evaluation_step')
    if core(module/'finite_value_evaluation.py')!=core(source/'source_at_launch/learning/src/sttw_control/finite_value_evaluation.py'):
        raise ValueError('original evaluation kernel changed')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    declaration=json.loads((source/'source_declaration.json').read_text())
    reference=Path(old['source_reference_run']);ref=json.loads((reference/'plan.json').read_text())
    if digest(reference/'plan.json')!=old['source_evaluation_plan_sha256']:raise ValueError('reference control identity changed')
    models={role:copy.deepcopy(ref['models'][role]) for role in ROLES if role!='feedback_nominal'}
    models['feedback_nominal']=copy.deepcopy(old['models']['feedback_nominal'])
    for role,model in models.items():model.update(feedback_kind=1 if role=='feedback_nominal' else 3 if role=='feedback_010' else 0,
        feedback_gain_rad=.1 if role in ['feedback_nominal','feedback_010'] else 0.)
    authenticate_initial_models(cfg,models)
    previous=json.loads((reference/'inventory.json').read_text())['cases']
    earlier=Path(ref['source_reference_run'])/'inventory.json'
    previous+=json.loads(earlier.read_text())['cases']
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 真实进入状态固定名义表准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        shutil.copyfile(protocol_file,root/'protocol.json')
        env,_=load_environment_contact_only(cfg,declaration['source'],'terminal_contract_v2')
        bank_file=Path(declaration['source']['prepared_bank'])
        with bank_file.open('rb') as f:bank=pickle.load(f)
        (root/'initial_states').mkdir();contexts={};profiles={};receipts=[]
        for base,c in old['contexts'].items():
            with np.load(c['path'],allow_pickle=False) as z:path=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
            for index in protocol['bank_indices']:
                name=f'{base}__bank{index}'
                snapshot=jax.tree.map(lambda a:j.asarray(a[index]),bank)
                state=env.reset(snapshot,path,c['speed_m_s'],alpha=0.)
                if bool(state.physical.failed|state.fault|state.domain_exit) or int(state.tick)!=0 or int(state.physical.physical_tick)!=700:
                    raise ValueError('declared real start invalid; do not silently exclude or replace it')
                # Standard task reset changes task references and creates fresh
                # task-level lower history exactly as in the established reset.
                for field in ['data','controller','actuator','governor']:
                    for a,b in zip(jax.tree.leaves(getattr(snapshot,field)),jax.tree.leaves(getattr(state.physical,field))):
                        np.testing.assert_array_equal(a,b)
                initial=root/'initial_states'/f'{name}.pkl'
                with initial.open('wb') as f:pickle.dump(SimpleNamespace(state=jax.tree.map(lambda a:np.asarray(a)[None],state)),f)
                measured,pose,speed=env.physics.observe(state.physical.data)
                contexts[name]=dict(copy.deepcopy(c),initial=str(initial),initial_sha256=digest(initial),
                    rebuild_path_context=False,nominal_profile_context=base,bank_index=index,
                    initial_progress_m=float(state.projection.progress))
                for alpha in protocol['alphas']:
                    profiles[f'{name}/a{alpha:g}']=copy.deepcopy(old['nominal_profiles'][f'{base}/a{alpha:g}'])
                receipts.append(dict(context=name,bank_index=index,prepared_bank=str(bank_file),
                    prepared_bank_sha256=declaration['source']['prepared_bank_sha256'],
                    physical_fields_retained_exact=['data','controller','actuator','governor'],
                    task_initialization='Existing env.reset on real prepared Snapshot; no physics advance/qpos/qvel edits or path relocation.',
                    pose=np.asarray(pose).tolist(),speed_m_s=float(speed),roll=float(measured[0]),
                    task_tick=0,physical_tick=700))
        cases=initial_cases(protocol,contexts,previous)
        atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
        atomic_json(root/'initial_state_receipt.json',dict(cases=receipts,physics_steps=0))
        atomic_json(root/'nominal_profiles.json',profiles)
        files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py')]
        sources={str(f.resolve()):digest(f) for f in files}
        for file in files:
            dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        names=['common_environment_config.json','source_declaration.json','protocol.json','inventory.json',
               'authenticated_models.json','initial_state_receipt.json','nominal_profiles.json']
        sealed={str(root/name):digest(root/name) for name in names}
        sealed.update({p['file']:p['sha256'] for p in profiles.values()})
        sealed.update({c['initial']:c['initial_sha256'] for c in contexts.values()})
        sealed.update({str(file.resolve()):digest(file) for file in [reference/'inventory.json',earlier]})
        plan=dict(project='STTW_CONTROL',experiment=protocol['schema'],models=models,contexts=contexts,
            evaluation_roles=list(ROLES),declared_cases=300,new_cases=300,cached_cases=0,analysis_cases=300,
            feedback_spec=old['feedback_spec'],nominal_feedback_spec=old['nominal_feedback_spec'],
            initial_state_confirmation=True,nominal_profiles=profiles,horizon_s=10.,lower_dt_s=.005,
            maximum_active_lower_ticks=600000,maximum_allocated_lower_slots=600000,optimizer_updates=0,
            goal_radius_m=.5,additional_report_radii_m=[.1,.2,.3],task_rule=old['task_rule'],seed=protocol['seed'],
            cached_active_lower_ticks=0,cached_allocated_lower_slots=0,
            endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
            source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
            sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            initialization='Standard task reset from prescribed real prepared physical bank2/4, retaining full physical/controller/actuator/governor state. Original paths fixed; no injected state. Bank3 reconstruction bridge checked separately, no new integration.',
            information=old['information'],stop=protocol['stop'],
            data_role='Prospective DEV confirmation on two previously unused physical starts for this mechanism, paired correlated branches; not independent training seeds or broad unseen-route generalization. Original six bank3 nominal tables remain byte-identical; no new nominal fitting.')
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=300,completed_cases=0,
            active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 真实进入状态固定名义表300回合',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 真实进入状态300路shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise
