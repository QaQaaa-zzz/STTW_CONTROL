"""Explicit valid-domain handback opportunity, retaining all real closed-loop state."""
import json
import math
from pathlib import Path
import jax.numpy as j
from flax import struct
from .preference_best import atomic_json,digest


@struct.dataclass
class ReentryEvaluationState:
    state: object
    gate: object
    clean_blocks: object


def actor_permission(ever_exited,current_invalid,clean_blocks,zero_mode,stopped):
    ever_exited=j.asarray(ever_exited,bool);invalid=j.asarray(current_invalid,bool)
    return (~ever_exited|(j.asarray(clean_blocks)>=16))&~invalid&~j.asarray(zero_mode,bool)&~j.asarray(stopped,bool)


def next_clean_blocks(previous,active,invalid,failure):
    clean=j.all(active)&~j.any(invalid)&~j.any(failure)
    return j.where(clean,j.minimum(j.asarray(previous)+1,16),0).astype(j.int32)


def reentry_cases(original,*,roles=None,reference_speed_floor_m_s=None,reference_profiles=None):
    roles=dict(task_only='task_reentry',physical_credit='physical_reentry') if roles is None else dict(roles)
    supported={'parent','task_only','physical_credit','floor','causal_eso_allalpha'}
    if not roles or not set(roles)<=supported or len(set(roles.values()))!=len(roles):
        raise ValueError('unique declared Actor-based roles required')
    if reference_profiles is not None:
        reference_profiles=tuple(reference_profiles)
        if not reference_profiles or len(set(reference_profiles))!=len(reference_profiles) or not set(reference_profiles)<={6,7}:
            raise ValueError('unique explicit early/constant support profiles required')
    result=[]
    for c in original:
        if c['role'] not in roles:continue
        new=dict(c,source_case_id=c['case_id'],case_id=roles[c['role']]+'/'+c['case_id'],
            role=roles[c['role']],stage='reentry')
        if reference_speed_floor_m_s is not None:
            floor=float(reference_speed_floor_m_s)
            if not math.isfinite(floor) or floor<float(c['reference_speed_floor_m_s']):
                raise ValueError('finite reference floor must not widen original bounds')
            new['reference_speed_floor_m_s']=floor
        if reference_profiles is None:result.append(new)
        else:
            for profile in reference_profiles:
                result.append(dict(new,profile=profile,role=new['role']+f'_profile{profile}',
                    case_id=new['role']+f'_profile{profile}/'+c['case_id']))
    expected={(role,alpha,event) for role in roles for alpha in [.5,.75]
              for event in ['zero','t380_F-32','t380_F+32','t580_F-32','t580_F+32']}
    actual={(c['role'],c['alpha'],c['event_id']) for c in original if c['role'] in roles}
    if len(result)!=10*len(roles)*(len(reference_profiles) if reference_profiles is not None else 1) or actual!=expected or any(c['alpha']!=c['state_alpha'] for c in result):
        raise ValueError('exact declaredroles x twoalpha x fiveevents required')
    return result


def prepare_reentry(source,root,*,roles=None,reference_speed_floor_m_s=None,reference_profiles=None,declared_cases=None):
    """Seal declared handback cases and their exact factual baseline references."""
    from .hidden_time_confirmation import authenticate_saved_endpoint
    source=Path(source).resolve();root=Path(root).resolve()
    status=json.loads((source/'timing/status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=55:raise ValueError('complete right55 source required')
    prior=json.loads((source/'plan.json').read_text());oldcases=json.loads((source/'inventory.json').read_text())['cases']
    routes={c['route'] for c in oldcases}
    if len(routes)!=1 or not routes<={'right90_R1p1','left90_R1p1'} or (roles is None and declared_cases is None and routes!={'right90_R1p1'}):
        raise ValueError('one declared original route required')
    original=Path(json.loads((source.parent/'plan.json').read_text())['source_run'])
    training=Path(json.loads((source.parent/'plan.json').read_text())['source_training_run'])
    auth,receipt=authenticate_saved_endpoint(original,training)
    for name in ['common_environment_config.json','source_declaration.json','authenticated_models.json','inventory.json','origins.json']:
        if digest(source/name)!=prior['sealed_files'][str(source/name)]:raise ValueError('source input drift '+name)
    if digest(prior['source_initial_state'])!=prior['source_initial_state_sha256'] or digest(prior['path_file'])!=prior['path_sha256']:
        raise ValueError('complete task0 source/path drift')
    if reference_speed_floor_m_s is not None:
        cfg=json.loads((source/'common_environment_config.json').read_text())
        lo,hi=cfg['residual_interface']['speed_reference_bounds_m_s']
        if not math.isfinite(float(reference_speed_floor_m_s)) or not lo<=float(reference_speed_floor_m_s)<=hi:
            raise ValueError('reference floor must stay inside original configured bounds')
    if declared_cases is None:
        cases=reentry_cases(oldcases,roles=roles,reference_speed_floor_m_s=reference_speed_floor_m_s,reference_profiles=reference_profiles)
    else:
        from .prospective_preference import validate_common_origin_cases
        if roles is not None or reference_speed_floor_m_s is not None or reference_profiles is not None:
            raise ValueError('explicit cases cannot be mixed with implicit case transformations')
        cases=[dict(c) for c in declared_cases]
        validate_common_origin_cases(cases,oldcases)
    count=len(cases)
    # New roles share source native Actors and normalization. Only policy eligibility changes.
    for actor in {c['actor_role'] for c in cases}:
        auth['models'][actor]=dict(auth['models'][actor],runtime_kind='reentry')
    index={c['case_id']:c for c in json.loads((source/'timing/case_index.json').read_text())}
    results={c['case_id']:c for c in json.loads((source/'timing/results.json').read_text())}
    reuse=[]
    for c in cases:
        if declared_cases is not None:continue
        key=c['source_case_id'];item=dict(index[key]);file=source/item['file']
        if digest(file)!=item['sha256']:raise ValueError('baseline raw drift')
        item['file']=str(file.resolve());reuse.append(dict(case=results[key],index=item))
    root.mkdir(parents=True,exist_ok=False)
    for name in ['common_environment_config.json','source_declaration.json']:(root/name).write_bytes((source/name).read_bytes())
    atomic_json(root/'authenticated_models.json',auth);atomic_json(root/'historical_identity_receipt.json',receipt)
    atomic_json(root/'inventory.json',dict(project='STTW_CONTROL',cases=cases,source_run=str(source)))
    atomic_json(root/'reuse_cases.json',reuse)
    sources=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/evaluate_anticipatory_reference.py').resolve()]
    archive=root/'source_snapshot';archive.mkdir()
    for i,p in enumerate(sorted(sources)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    plan=dict(project='STTW_CONTROL',data_role='DEV',purpose='Does real-history originalalphaActor handback after16cleanblocks add useful task recovery without newfalls?',
        maximum_active_lower_ticks=2000*count,optimizer_updates=0,batch_size=20,steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,
        stage_counts=dict(reentry=count),logical_cases=count+len(reuse),inherited_cases=len(reuse),
        comparison_baseline_by_role={c['role']:old['case']['role'] for c,old in zip(cases,reuse)},
        sealed_files={str((root/n).resolve()):digest(root/n) for n in ['common_environment_config.json','source_declaration.json','authenticated_models.json','historical_identity_receipt.json','inventory.json','reuse_cases.json']},
        source_sha256={str(p.resolve()):digest(p) for p in sources},source_initial_state=prior['source_initial_state'],source_initial_state_sha256=prior['source_initial_state_sha256'],
        path_file=prior['path_file'],path_sha256=prior['path_sha256'],
        source_baseline_status_sha256=digest(source/'timing/status.json'),
        contract='Actual complete task0/physical700 and originalalpha; history/ESO/filter/actuator/projection/clocks never reset. Cumulative exit remains; policy eligibility separate,16clean complete blocks, currentinvalid immediately blocks nextupperboundary. Same oldreference bounds/rates and frozenplant.',
        information='No futurepulse metadata to eligibility orActor; actualhistory/currentgeometry only. Event schedule physicalintegrator only.',
        stop_rule='Declared newcases only/contact/fault/horizon; no automatic retries, extra waiting thresholds, rollgates, geometryL/reward or training.',
        inference='Common-interface opportunity, not newmethod or safetycertification. Anyuseful gain must then faceparent/floor/ESO with SAME handbackpermission andleft confirmation beforeadoption.')
    if reference_speed_floor_m_s is not None:
        plan.update(reference_speed_floor_m_s=float(reference_speed_floor_m_s),
            purpose='Does the SAME existing reference floor on learned Actors retain task recovery and improve survival, with shared handback permission?')
    if reference_profiles is not None:
        plan.update(reference_profiles=list(reference_profiles),
            purpose='Early versus constant speed support through the SAME original correction filter; no change to reference floor or user alpha.')
    if declared_cases is not None:
        plan.update(allow_task0_alpha_context_override=True,
            purpose='Prospective new timing and five user-alpha contexts, common complete origin; frozen reference rules.',
            initial_context='Source slot0/sourcealpha0; only upper user-alpha assigned explicitly at task0. Physics, lower/history/filter/actuator clocks retained.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=count,completed_cases=0,inherited_cases=len(reuse),active_lower_ticks=0,optimizer_updates=0))


def prepare_shared_reentry_campaign(source,pilot,output):
    """Declare missing80 only; preserve130 exact completed factual inputs."""
    import ast
    source=Path(source).resolve();pilot=Path(pilot).resolve();output=Path(output).resolve()
    a=json.loads((source/'status.json').read_text());b=json.loads((pilot/'reentry/status.json').read_text())
    if a['phase']!='completed' or a['completed_cases']!=110 or b['phase']!='completed' or b['completed_cases']!=20:
        raise ValueError('complete110 plus20 pilot required, no implicit recovery')
    oldplan=json.loads((pilot/'plan.json').read_text());module=Path(__file__).parent
    for name in ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','causal_reference_baseline.py','path_reference.py','path_command_reward.py']:
        if digest(module/name)!=oldplan['source_sha256'][str((module/name).resolve())]:raise ValueError('changed handback/control kernel '+name)
    archived=next((pilot/'source_snapshot').glob('*_reference_reentry.py'))
    def core(path):
        tree=ast.parse(path.read_text())
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['actor_permission','next_clean_blocks','ReentryEvaluationState']}
    if core(Path(__file__))!=core(archived):raise ValueError('changed gate/history contract')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'status.json',dict(phase='preparing',declared_cases=80,completed_cases=0,active_lower_ticks=0,optimizer_updates=0))
    roles=dict(parent='parent_reentry',task_only='task_reentry',physical_credit='physical_reentry',floor='floor_reentry',causal_eso_allalpha='eso_reentry')
    prepare_reentry(source/'left',output/'left',roles=roles)
    prepare_reentry(source/'right',output/'right',roles={k:roles[k] for k in ['parent','floor','causal_eso_allalpha']})
    cache=[]
    for directory,stage in [(source/'left','timing'),(source/'right','timing'),(pilot,'reentry')]:
        index={x['case_id']:x for x in json.loads((directory/stage/'case_index.json').read_text())}
        rows=json.loads((directory/stage/'results.json').read_text())
        for row in rows:
            item=dict(index[row['case_id']]);raw=directory/item['file']
            if digest(raw)!=item['sha256']:raise ValueError('cached factual array drift')
            item['file']=str(raw.resolve());cache.append(dict(case=row,index=item,origin_run=str(directory),origin_stage=stage))
    assert len(cache)==130
    new=[]
    for short in ['left','right']:
        cases=json.loads((output/short/'inventory.json').read_text())['cases']
        new += [dict(case=c,planned_run=str(output/short),planned_stage='reentry') for c in cases]
    keys=lambda rows:[(x['case']['route'],x['case']['case_id']) for x in rows]
    if len(new)!=80 or len(set(keys(cache)+keys(new)))!=210:raise ValueError('duplicate or incomplete210-case identity inventory')
    atomic_json(output/'cached_cases.json',cache);atomic_json(output/'planned_new_cases.json',new)
    atomic_json(output/'kernel_identity.json',dict(pilot_plan_sha256=digest(pilot/'plan.json'),gate_ast_unchanged=True,
        frozen_runtime_modules={name:digest(module/name) for name in ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','causal_reference_baseline.py','path_reference.py','path_command_reward.py']},
        source_pilot_code='c3a18cb',physical_steps=0,optimizer_updates=0))
    atomic_json(output/'plan.json',dict(project='STTW_CONTROL',data_role='DEV',purpose='Same handback permission for strong rules, plusleft confirmation; no new gate/weights/reward.',
        source_run=str(source),source_pilot=str(pilot),new_actual_cases=80,inherited_actual_cases=130,logical_cases=210,maximum_active_lower_ticks=160000,optimizer_updates=0,
        route_new_counts=dict(left=50,right=30),stage='reentry',route_order=['left','right'],
        route_plan_sha256={short:digest(output/short/'plan.json') for short in ['left','right']},
        cached_inventory_sha256=digest(output/'cached_cases.json'),planned_inventory_sha256=digest(output/'planned_new_cases.json'),kernel_identity_sha256=digest(output/'kernel_identity.json'),
        stop_rule='One pass of missing80 only; faults stop remaining stages. No automatic retry/extension, no waiting/roll/reward tuning or model adoption.',
        inference='RelatedDEV branches/commonorigin andsingleparent model. Originalzero remains10unique. Newmethods must compare SAMEhandbackpermission; interface gain not byitself novel method.'))
    atomic_json(output/'status.json',dict(phase='prepared',declared_cases=80,completed_cases=0,inherited_actual_cases=130,logical_cases=210,active_lower_ticks=0,optimizer_updates=0))
