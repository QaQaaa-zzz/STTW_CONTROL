"""A fixed nominal-selected two-process disturbance probe, no retuning."""
import json,time
from pathlib import Path
from .preference_best import atomic_json,digest

SELECTED=('sequence_26','constant_slow')


def probe_cases(templates):
    if len(templates)!=8 or len({(r['alpha'],r['event_id']) for r in templates})!=8 or any(not r['force_N'] for r in templates):
        raise ValueError('eight matched nonzero original events required')
    return [dict(t,case_id=f'{role}/alpha{t["alpha"]}/{t["event_id"]}',role=role,actor_role=role,
        profile=0,reference_speed_floor_m_s=1.,zero_mode=False,stage='sequence_probe') for role in SELECTED for t in templates]


def prepare_sequence_probe(source,baselines,root):
    source=source.resolve();baselines=baselines.resolve();root=root.resolve()
    status=json.loads((source/'status.json').read_text())
    stage=json.loads((source/'sequence_nominal/status.json').read_text())
    if status['phase']!='completed' or not status.get('analysis_numeric_complete') or stage['completed_cases']!=74:
        raise ValueError('complete analyzed nominal74 required')
    sp=json.loads((source/'plan.json').read_text());bp=json.loads((baselines/'plan.json').read_text())
    bs=json.loads((baselines/'status.json').read_text())
    if bs['phase']!='completed' or bs['completed_cases']!=80:raise ValueError('complete matched80 required')
    exempt={'anticipatory_run.py','anticipatory_analysis.py','evaluate_anticipatory_reference.py','analyze_anticipatory_reference.py'}
    for plan in [sp,bp]:
        for path,sha in plan['source_sha256'].items():
            if Path(path).name not in exempt and digest(path)!=sha:raise ValueError('cached physical source drift '+path)
        for path,sha in plan['sealed_files'].items():
            if digest(path)!=sha:raise ValueError('cached identity drift '+path)
    for key in ['source_initial_state','source_initial_state_sha256','path_file','path_sha256']:
        if sp[key]!=bp[key]:raise ValueError('initial/path contract mismatch')
    if (source/'common_environment_config.json').read_bytes()!=(baselines/'common_environment_config.json').read_bytes():
        raise ValueError('runtime config mismatch')
    rr=json.loads((source/'sequence_nominal/analysis/results.json').read_text())
    for alpha in [0.,.5]:
        rows=[r for r in rr if r['alpha']==alpha]
        seq=min([r for r in rows if r.get('sequence_kind')=='sequence'],key=lambda r:r['scored_cost'])
        simple=min([r for r in rows if r.get('sequence_kind') in ('constant','zero')],key=lambda r:r['scored_cost'])
        if (seq['role'],simple['role'])!=SELECTED:raise ValueError('declared nominal selection changed')
    inv=json.loads((baselines/'inventory.json').read_text())['cases']
    cases=probe_cases([r for r in inv if r['stage']=='disturbed' and r['role']=='parent'])
    reuse=[]
    def inherit(run,stage,roles):
        rows=json.loads((run/stage/'results.json').read_text())
        idx={r['case_id']:r for r in json.loads((run/stage/'case_index.json').read_text())}
        for r in rows:
            if r['role'] not in roles:continue
            item=idx[r['case_id']];file=run/item['file']
            if digest(file)!=item['sha256']:raise ValueError('raw cache drift')
            reuse.append(dict(case=r,index=dict(item,file=str(file.resolve())),source_run=str(run)))
    for stage in ['nominal','disturbed']:inherit(baselines,stage,['parent','task_only','physical_credit','floor','causal_eso','zero_middle'])
    inherit(source,'sequence_nominal',SELECTED)
    if len(reuse)!=54 or len({r['case']['case_id'] for r in reuse})!=54:raise ValueError('54 unique inherited cases required')
    models=json.loads((source/'authenticated_models.json').read_text())['models']
    models={k:models[k] for k in SELECTED}
    for model in models.values():
        if digest(model['actor'])!=model['actor_sha256']:raise ValueError('fixed selected process changed')
    root.mkdir(parents=True,exist_ok=False)
    for name in ['common_environment_config.json','source_declaration.json']:(root/name).write_bytes((source/name).read_bytes())
    atomic_json(root/'authenticated_models.json',dict(models=models))
    atomic_json(root/'inventory.json',dict(project='STTW_CONTROL',cases=cases,new_actual_cases=16,inherited_cases=54,logical_cases=70))
    atomic_json(root/'reuse_cases.json',reuse)
    files=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/evaluate_anticipatory_reference.py').resolve(),Path('learning/cli/prepare_sequence_probe.py').resolve()]
    archive=root/'source_snapshot';archive.mkdir()
    for i,p in enumerate(sorted(files)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    names=['common_environment_config.json','source_declaration.json','authenticated_models.json','inventory.json','reuse_cases.json']
    plan=dict(project='STTW_CONTROL',data_role='DEV',purpose='Does nominal-selected full-task reference coordination preserve useful physical outcomes at its original task cost beyond constant and trained controls?',
        stage_counts=dict(sequence_probe=16),logical_cases=70,inherited_cases=54,maximum_active_lower_ticks=32000,optimizer_updates=0,
        batch_size=16,steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,allow_task0_alpha_context_override=False,
        nominal_source=str(source),baseline_source=str(baselines),selection=SELECTED,
        selection_basis='Postnominal exploratory choice by lowest original cost in sequence/constant families, fixed before any selected-process force outcome. Original nominal safe_task labels remain failed; no new threshold.',
        source_initial_state=sp['source_initial_state'],source_initial_state_sha256=sp['source_initial_state_sha256'],path_file=sp['path_file'],path_sha256=sp['path_sha256'],
        sealed_files={str((root/name).resolve()):digest(root/name) for name in names},source_sha256={str(p.resolve()):digest(p) for p in files},
        contract=sp['contract'],information='Process uses current actual progress only. No actual future force/onset/direction input; common world-time force changes actual entering state, not matched disturbed state.',
        stop_rule='16 actual episodes only; no optimizer, parameter/threshold retuning, additional force/candidate library, automatic retry or horizon change.',
        limitation='One previously used DEV context; same process for bothalpha, no newly established alpha-conditioned selection or continuous force threshold. No method adoption from eventwise best.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=16,new_actual_cases=16,inherited_cases=54,logical_cases=70,
        completed_cases=0,active_lower_ticks=0,optimizer_updates=0,checked_unix=time.time()))
