"""Fixed nominal search -> freeze decisions -> complete event evaluation."""
import copy
import json
import os
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .reference_bias_optimization import minimize_nominal_cost
from .reference_bias_search import batch_slices,freeze_selected,evaluation_schedule


def costs(out):
    files=list((out/'searches').glob('search_*/generation_*/status.json'))+list((out/'evaluation').glob('batch_*/status.json'))
    records=[json.loads(f.read_text()) for f in files]
    return dict(active_lower_ticks=sum(r.get('active_lower_ticks',0) for r in records),
                allocated_lower_slots=sum(r.get('allocated_lower_slots',0) for r in records),
                recorded_physical_batches=len(records),optimizer_updates=0)


def prepare(root):
    root=Path(root).resolve();out=root/'production'
    if out.exists():raise ValueError('fresh production only; no implicit resume')
    for f in ['engineering/recovered_numeric/status.json','engineering/analysis_recovered_numeric/status.json','shape_batch8/status.json']:
        if json.loads((root/f).read_text())['phase']!='completed':raise ValueError('complete wiring/recovery/analysis/eight-lane shape required')
    parent=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('parent plan changed')
    for file,sha in {**parent['sealed_files'],**parent['source_sha256']}.items():
        if digest(file)!=sha:raise ValueError('original parent input/control source changed '+file)
    searches=json.loads((root/'search_inventory.json').read_text());events=json.loads((root/'evaluation_events.json').read_text())
    if len(searches)!=24 or len(events)!=60:raise ValueError('original24/60 fixed contract required')
    for key in {(s['context'],s['alpha']) for s in searches}:
        arms=[s for s in searches if (s['context'],s['alpha'])==key]
        if len(arms)!=2 or {s['method'] for s in arms}!={'cem','random'} or len({s['seed'] for s in arms})!=1:raise ValueError('unmatched search arms')
    out.mkdir()
    files=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/run_reference_bias.py').resolve(),Path('learning/cli/analyze_reference_bias_campaign.py').resolve()]
    sources={str(f.resolve()):digest(f) for f in files};archive=out/'source';archive.mkdir()
    for i,f in enumerate(files):(archive/f'{i:03d}_{f.name}').write_bytes(f.read_bytes())
    # Freeze every relevant baseline now, not according to optimized outcomes.
    cache=[]
    event_table={(e['context'],e['alpha'],e['event_id']):e for e in events}
    for run,roles in [(Path(parent['source']),['goal_zero_feedback','goal_steer_feedback','goal_full_feedback']),
                      (Path(parent['source']).parent/'steering_stage_mechanism_v1',['neural_after_only'])]:
        if json.loads((run/'analysis/status.json').read_text())['phase']!='completed':raise ValueError('baseline incomplete')
        bp=json.loads((run/'plan.json').read_text())
        if digest(run/'plan.json')!=json.loads((run/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('baseline plan changed')
        if digest(run/'common_environment_config.json')!=digest(root/'common_environment_config.json'):raise ValueError('baseline config differs')
        for role in roles:
            if bp['models'][role]['actor_sha256']!=parent['model']['actor_sha256']:raise ValueError('baseline Actor differs')
        for name,c in parent['contexts'].items():
            for f in ['initial_sha256','path_sha256','speed_m_s','world_direction']:
                if bp['contexts'][name][f]!=c[f]:raise ValueError('baseline context identity differs')
        for r in json.loads((run/'analysis/results.json').read_text()):
            if r['context'] in parent['contexts'] and r['role'] in roles:
                e=event_table[(r['context'],r['alpha'],r['event_id'])]
                for field in ['force_N','pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s','feedback_gain_rad','causal_support']:
                    if r[field]!=e[field]:raise ValueError('baseline event or authority differs: '+field)
                if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('baseline raw changed')
                cache.append(dict(r))
    if len(cache)!=240:raise ValueError('all240 baseline traces required')
    atomic_json(out/'cached_baselines.json',cache)
    contract=dict(parent_plan_sha256=digest(root/'plan.json'),search_inventory_sha256=digest(root/'search_inventory.json'),
        event_inventory_sha256=digest(root/'evaluation_events.json'),source_sha256=sources,cached_baselines_sha256=digest(out/'cached_baselines.json'),
        declared_search_episodes=1536,declared_evaluation_episodes=120,declared_cached_episodes=240,
        maximum_active_lower_ticks=3312000,maximum_allocated_lower_slots=3312000,
        inherited_engineering_active_lower_ticks=32000,opportunity_project_maximum_new_lower_including_engineering=3344000,
        evaluation_real_batch_sizes=[16]*7+[8],search_policy='Original fixed CEM/random4x16, nominal-only objective and feasibility from parent; all candidate/incumbent repeats charged.',
        selection='Actual feasible nominal witness; freeze all24 references+source SHA before any final event. One reference per(context,originalalpha,method), no realized event in selection.',
        analysis='All120 new+240 frozen cached events; original goal0.5/.75/1m before contact, physical survival separately, original-alpha cross-score and paired losses, full XY and total computation costs. Privileged after-only marked.',
        stop='Fixed contract only; no retries/extra candidates/events/time/alpha/force/knots. No control adoption; no best selected from final evaluation.',
        data_role='Seen shared-ancestor DEV and per-task offline nominal search, not unseen test or independent training repeats.',
        production_ready=True)
    atomic_json(out/'contract.json',contract);atomic_json(out/'seal.json',dict(contract_sha256=digest(out/'contract.json')))
    atomic_json(out/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,completed_searches=0,evaluation_cases=0))
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_PRODUCTION.json','STTW名义定向与随机参考优化对照',out/'status.json')


def execute(root):
    from .reference_bias_runtime import BiasRuntime
    root=Path(root).resolve();out=root/'production';status_file=out/'status.json'
    previous=json.loads(status_file.read_text())
    recovery_file=out/'recovery_binding.json'
    recovery=json.loads(recovery_file.read_text()) if recovery_file.exists() else None
    if previous['phase']!='prepared' and not (previous['phase']=='error' and recovery):
        raise ValueError('one prepared production or explicitly raw-recovered audit error only')
    contract=json.loads((out/'contract.json').read_text())
    if not contract.get('production_ready'):raise ValueError('production implementation not ready')
    if digest(out/'contract.json')!=json.loads((out/'seal.json').read_text())['contract_sha256']:raise ValueError('contract changed')
    sources=dict(contract['source_sha256'])
    if recovery:
        if recovery['original_contract_sha256']!=digest(out/'contract.json'):
            raise ValueError('recovery belongs to another contract')
        if previous!=recovery['original_status']:raise ValueError('recovery error identity changed')
        if previous['completed_searches'] or previous['evaluation_cases'] or previous['recorded_physical_batches']!=1:
            raise ValueError('only the single completed first physical batch may be recovered')
        for file,change in recovery['source_changes'].items():
            if change['old_sha256']!=sources.get(file):raise ValueError('source ancestry changed')
            sources[file]=change['new_sha256']
    for file,sha in sources.items():
        if digest(file)!=sha:raise ValueError('production source changed '+file)
    for file,key in [('plan.json','parent_plan_sha256'),('search_inventory.json','search_inventory_sha256'),('evaluation_events.json','event_inventory_sha256')]:
        if digest(root/file)!=contract[key]:raise ValueError('original declaration changed')
    if digest(out/'cached_baselines.json')!=contract['cached_baselines_sha256']:raise ValueError('baseline catalog changed')
    state=dict(previous,phase='running',stage='initializing',pid=os.getpid());start=time.monotonic()
    def update(**kw):
        state.update(costs(out),**kw,elapsed_s=time.monotonic()-start)
        if state['active_lower_ticks']>3312000 or state['allocated_lower_slots']>3312000:raise ValueError('fixed total exceeded')
        atomic_json(status_file,state)
        root_status=json.loads((root/'status.json').read_text())
        root_status.update(state,production_ready=True,production_execution_status=state['phase'],
                           total_new_lower_ticks_including_engineering=state['active_lower_ticks']+32000)
        atomic_json(root/'status.json',root_status)
    update()
    try:
        runtime=BiasRuntime(root)
        searches=json.loads((root/'search_inventory.json').read_text());events=json.loads((root/'evaluation_events.json').read_text());selected=[]
        for index,search in enumerate(searches):
            job=out/'searches'/f'search_{index:02d}'
            if index==0 and recovery:
                if json.loads((job/'search.json').read_text())!=search:raise ValueError('recovered search identity changed')
            else:job.mkdir(parents=True,exist_ok=False)
            atomic_json(job/'search.json',search);all_rows=[];generation=0
            nominal=next(e for e in events if e['context']==search['context'] and e['alpha']==search['alpha'] and e['force_N']==0)
            def evaluate(vectors):
                nonlocal generation
                cases=[]
                for lane in range(16):
                    c=copy.deepcopy(nominal);c.update(case_id=search['search_id']+f'/g{generation}/c{lane:02d}',role=search['method'],generation=generation,candidate_index=lane)
                    cases.append(c)
                rows=runtime.rollout(cases,vectors,job/f'generation_{generation:02d}')
                all_rows.extend(rows);generation+=1
                update(stage='nominal_search',search_index=index,generation=generation)
                return np.asarray([r['objective'] if r['nominal_feasible'] else np.inf for r in rows])
            outcome=minimize_nominal_cost(evaluate,seed=search['seed'],method=search['method'])
            decision=freeze_selected(search,outcome,all_rows)
            if digest(decision['nominal_raw_file'])!=decision['nominal_raw_sha256']:raise ValueError('selected nominal raw changed')
            selected.append(decision)
            atomic_json(job/'optimizer_trace.json',dict(outcome,vector=outcome['vector'].tolist()))
            atomic_json(job/'selected.json',decision)
            update(stage='nominal_search',completed_searches=index+1)
        atomic_json(out/'selected_references.json',selected)
        selected_sha=digest(out/'selected_references.json')
        cases,vectors=evaluation_schedule(events,selected)
        atomic_json(out/'evaluation_inventory.json',dict(cases=cases,vectors=vectors.tolist(),selected_references_sha256=selected_sha))
        atomic_json(out/'selection_frozen.json',dict(selected_references_sha256=selected_sha,searches_complete=24,search_cost=costs(out),final_event_cases_executed=0))
        results=[]
        for i,sl in enumerate(batch_slices(120)):
            if digest(out/'selected_references.json')!=selected_sha:raise ValueError('nominal choices changed after seeing final events')
            results.extend(runtime.rollout(cases[sl],vectors[sl],out/'evaluation'/f'batch_{i:02d}'))
            update(stage='frozen_event_evaluation',evaluation_cases=len(results))
        atomic_json(out/'evaluation_results.json',results)
        update(phase='completed',stage='numeric_complete_analysis_pending',evaluation_cases=120)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc())
        update(phase='error',error=repr(exc));raise
