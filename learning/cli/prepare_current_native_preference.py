"""Prepare eight missing current-weight traces in H75's native protocol."""
import argparse
import copy
import gzip
import hashlib
import io
import json
import pickle
import shutil
import subprocess
import tarfile
from pathlib import Path

from sttw_control.preference_best import atomic_json,digest


def prepare(root):
    from sttw_control.finite_value_evaluation_run import authenticate_models
    source=Path('runs/preference_full_tail_length75_post_v1/moderate/evaluation').resolve()
    previous=Path('runs/lineage_nominal_common_interface_v1/analysis_corrected_roles_v1/h75_native_preference_v1').resolve()
    old=json.loads((source/'manifest.json').read_text());baseline=json.loads((previous/'review.json').read_text())
    if digest(source/'manifest.json')!=baseline['source_manifest_sha256']:raise ValueError('native source changed')
    current_run=Path('runs/entry_history_confirmation_v1').resolve()
    plan=json.loads((current_run/'plan.json').read_text());model=copy.deepcopy(plan['models']['goal_full_feedback'])
    model.pop('source_keep_channels')
    cfg=json.loads((current_run/'common_environment_config.json').read_text())
    authenticate_models(cfg,dict(free_GAE=model))
    for field in ['feedback_gain_rad','feedback_kind']:model.pop(field,None)
    for row in baseline['records']:
        if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('native H75 cached raw changed')
    decl=copy.deepcopy(old['declaration']);decl.update(experiment_id='current_native_preference_v1',
        question='Does the current frozen middle Actor retain H75 native high-demand preference behavior?',
        policies={'current':model},alphas=[0.,1.],forces_N=[0.],onsets_s=[1.],maximum_lower_ticks=16000,
        data_role='Native-interface DEV diagnostic:8 missing current-weight nominal traces;8 historical H75 traces reused. Shared source bank, no independent repeat or new control mechanism.',
        stop_rule='Exactly8 frozen current trajectories, no optimizer, feedback, support, goal-preview rule, checkpoint selection, retry or extension.',
        adoption_status='not_applicable',training_steps=0)
    # Original tracked evaluator defaults; optional recent features are absent.
    for name in ['preference_switching','recovery_intervention','teaching_records','diagnostic_physical_stop','record_policy_inputs']:
        decl.pop(name,None)
    for name in ['training_campaign','training_declaration_sha256','baseline_cached']:
        decl.pop(name,None)
    decl['initialization']='Original native prepared-bank snapshot and env.reset semantics; frozen current middle weights, no optimizer or extra support.'
    root=Path(root).resolve();root.mkdir(exist_ok=False)
    atomic_json(root/'declaration.json',decl)
    runtime=root/'native_runtime';runtime.mkdir()
    commit=old['code_revision'];raw=subprocess.check_output(['git','archive','--format=tar',commit,'learning'])
    archive=gzip.compress(raw,mtime=0);(root/'native_source.tar.gz').write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:tar.extractall(runtime,filter='data')
    for file in runtime.rglob('*'):
        if file.is_file():file.chmod(0o444)
    atomic_json(root/'cache_receipt.json',dict(source_run=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        original_records=baseline['records'],historical_cases=8,historical_actual_lower_ticks=16000,
        historical_allocated_lower_slots=16000,deduplication='fixed F0/onset1, original four bend/speed contexts, alpha0/1'))
    sources={str(f.resolve()):digest(f) for f in runtime.rglob('*.py')}
    atomic_json(root/'plan.json',dict(project='STTW_CONTROL',experiment='current_native_preference_v1',
        question=decl['question'],current_model=model,native_manifest=str(source/'manifest.json'),
        source_manifest_sha256=digest(source/'manifest.json'),native_tracked_code_commit=commit,
        native_source_archive_sha256=hashlib.sha256(archive).hexdigest(),native_runtime_files=sources,
        archived_runtime_scope='Tracked source at original native manifest commit; no claim original execution lacked untracked patches or exact physical replay.',
        initial_source=decl['source'],config=decl['config'],new_cases=8,cached_cases=8,analysis_cases=16,
        maximum_new_lower_ticks=16000,maximum_new_allocated_lower_slots=16000,optimizer_updates=0,
        initialization='Same original source prepared-bank complete snapshot; original env.reset/path/history semantics for each own alpha. No lower reset injection or borrowed other-policy prefix.',
        baseline_cache='cache_receipt.json',feedback=False,short_support=False,goal_preview=False,
        evaluation_semantics='Original native ten-second task/physical termination and goal-section/hold diagnostics; continuous errors and full same-user endpoint costs retained; no requirement every case succeeds.',
        declared_fields_only=True,stop=decl['stop_rule'],execution_status='prepared',adoption_status='not_applicable'))
    atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json'),declaration_sha256=digest(root/'declaration.json'),cache_sha256=digest(root/'cache_receipt.json')))
    atomic_json(root/'prepare_status.json',dict(phase='completed',physics_steps=0,optimizer_updates=0,authenticated_cached_cases=8))
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_RUN.json','STTW当前模型原生八场景',root/'evaluation/status.json')
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
