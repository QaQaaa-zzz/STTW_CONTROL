"""Explicit valid-domain handback opportunity, retaining all real closed-loop state."""
import json
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


def reentry_cases(original):
    roles=dict(task_only='task_reentry',physical_credit='physical_reentry')
    result=[]
    for c in original:
        if c['role'] not in roles:continue
        result.append(dict(c,source_case_id=c['case_id'],case_id=roles[c['role']]+'/'+c['case_id'],
            role=roles[c['role']],stage='reentry'))
    if len(result)!=20:raise ValueError('exact twoactors x twoalpha x fiveevents required')
    return result


def prepare_reentry(source,root):
    """Seal20 new actors-enabled cases and20 exact factual baseline references."""
    from .hidden_time_confirmation import authenticate_saved_endpoint
    source=Path(source).resolve();root=Path(root).resolve()
    status=json.loads((source/'timing/status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=55:raise ValueError('complete right55 source required')
    prior=json.loads((source/'plan.json').read_text());oldcases=json.loads((source/'inventory.json').read_text())['cases']
    if {c['route'] for c in oldcases}!={'right90_R1p1'}:raise ValueError('declared right route only')
    original=Path(json.loads((source.parent/'plan.json').read_text())['source_run'])
    training=Path(json.loads((source.parent/'plan.json').read_text())['source_training_run'])
    auth,receipt=authenticate_saved_endpoint(original,training)
    for name in ['common_environment_config.json','source_declaration.json','authenticated_models.json','inventory.json','origins.json']:
        if digest(source/name)!=prior['sealed_files'][str(source/name)]:raise ValueError('source input drift '+name)
    if digest(prior['source_initial_state'])!=prior['source_initial_state_sha256'] or digest(prior['path_file'])!=prior['path_sha256']:
        raise ValueError('complete task0 source/path drift')
    cases=reentry_cases(oldcases)
    # New roles share source native Actors and normalization. Only policy eligibility changes.
    auth['models']['task_only']=dict(auth['models']['task_only'],runtime_kind='reentry')
    auth['models']['physical_credit']=dict(auth['models']['physical_credit'],runtime_kind='reentry')
    index={c['case_id']:c for c in json.loads((source/'timing/case_index.json').read_text())}
    results={c['case_id']:c for c in json.loads((source/'timing/results.json').read_text())}
    reuse=[]
    for c in cases:
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
        maximum_active_lower_ticks=40000,optimizer_updates=0,batch_size=20,steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,
        stage_counts=dict(reentry=20),logical_cases=40,inherited_cases=20,comparison_baseline_by_role=dict(task_reentry='task_only',physical_reentry='physical_credit'),
        sealed_files={str((root/n).resolve()):digest(root/n) for n in ['common_environment_config.json','source_declaration.json','authenticated_models.json','historical_identity_receipt.json','inventory.json','reuse_cases.json']},
        source_sha256={str(p.resolve()):digest(p) for p in sources},source_initial_state=prior['source_initial_state'],source_initial_state_sha256=prior['source_initial_state_sha256'],
        path_file=prior['path_file'],path_sha256=prior['path_sha256'],
        source_baseline_status_sha256=digest(source/'timing/status.json'),
        contract='Actual complete task0/physical700 and originalalpha; history/ESO/filter/actuator/projection/clocks never reset. Cumulative exit remains; policy eligibility separate,16clean complete blocks, currentinvalid immediately blocks nextupperboundary. Same oldreference bounds/rates and frozenplant.',
        information='No futurepulse metadata to eligibility orActor; actualhistory/currentgeometry only. Event schedule physicalintegrator only.',
        stop_rule='20new only/contact/fault/horizon; no automatic retries, extra waiting thresholds, rollgates, geometryL/reward or training.',
        inference='Common-interface opportunity, not newmethod or safetycertification. Anyuseful gain must then faceparent/floor/ESO with SAME handbackpermission andleft confirmation beforeadoption.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=20,completed_cases=0,inherited_cases=20,active_lower_ticks=0,optimizer_updates=0))
