"""Seal a finite full-task pure-reference opportunity screen, without training."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .full_task_reference_sequence import candidate_library


def sequence_cases(templates,names):
    if len(templates)!=2 or {r['alpha'] for r in templates}!={0.,.5} or len(names)!=37 or len(set(names))!=37:
        raise ValueError('two original alpha templates and37 distinct processes required')
    return [dict(t,case_id=f'{name}/alpha{t["alpha"]}/zero',role=name,actor_role=name,
        sequence_index=i,sequence_kind='sequence' if i<32 else 'zero' if i==32 else 'constant',
        profile=0,reference_speed_floor_m_s=1.,zero_mode=False,event_id='zero',force_N=0.,stage='sequence_nominal')
        for i,name in enumerate(names) for t in templates]


def prepare_full_sequence(source,root):
    source=source.resolve();root=root.resolve()
    state=json.loads((source/'status.json').read_text())
    if state['phase']!='completed' or state['completed_cases']!=80:raise ValueError('completed80 source required')
    old=json.loads((source/'plan.json').read_text())
    for name,sha in old['sealed_files'].items():
        if digest(name)!=sha:raise ValueError('source identity drift '+name)
    # Only dispatch/analysis extension points changed after completed sources;
    # physical kernel, Actor, mapping, reference and reward sources still match.
    dispatch={'anticipatory_run.py','anticipatory_analysis.py','evaluate_anticipatory_reference.py','analyze_anticipatory_reference.py'}
    for name,sha in old['source_sha256'].items():
        if Path(name).name not in dispatch and digest(name)!=sha:raise ValueError('physical/Actor cached source drift '+name)
    for name,key in [(old['source_initial_state'],'source_initial_state_sha256'),(old['path_file'],'path_sha256')]:
        if digest(name)!=old[key]:raise ValueError('complete initial/path drift')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    lib=candidate_library(cfg)
    inventory=json.loads((source/'inventory.json').read_text())['cases']
    templates=[r for r in inventory if r['stage']=='nominal' and r['role']=='parent']
    rows=sequence_cases(templates,lib['names'])
    # Do not quietly assume independent original source lanes were identical.
    import jax
    with Path(old['source_initial_state']).open('rb') as f:initial=pickle.load(f).state
    equal=[]
    for parent_lane,other_lane in [(6,8),(6,10),(7,9),(7,11)]:
        aa=jax.tree.leaves(jax.tree.map(lambda x:np.asarray(x[parent_lane]),initial))
        bb=jax.tree.leaves(jax.tree.map(lambda x:np.asarray(x[other_lane]),initial))
        if any(not np.array_equal(a,b,equal_nan=True) for a,b in zip(aa,bb)):
            raise ValueError('baseline original complete lane mismatch')
        equal.append([parent_lane,other_lane])
    auth=json.loads((source/'authenticated_models.json').read_text())
    model_files={}
    for role in ['parent','task_only','physical_credit']:
        model=auth['models'][role]
        for kind in ['actor','checkpoint']:
            if digest(model[kind])!=model[kind+'_sha256']:raise ValueError('baseline model drift')
            model_files[model[kind]]=model[kind+'_sha256']
    cached=json.loads((source/'nominal/results.json').read_text())
    idx={x['case_id']:x for x in json.loads((source/'nominal/case_index.json').read_text())}
    reuse=[]
    for r in cached:
        if r['role'] not in ['parent','task_only','physical_credit','floor','causal_eso','zero_middle']:continue
        item=idx[r['case_id']];file=source/item['file']
        if digest(file)!=item['sha256']:raise ValueError('cached baseline raw drift')
        reuse.append(dict(case=r,index=dict(item,file=str(file.resolve())),source_run=str(source)))
    if len(reuse)!=10:raise ValueError('ten baseline traces: ESO onlyalpha0; uniquezero required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ['common_environment_config.json','source_declaration.json']:(root/name).write_bytes((source/name).read_bytes())
    np.savez(root/'candidate_targets.npz',targets=lib['targets'])
    atomic_json(root/'candidate_definitions.json',dict(names=lib['names'],kind=lib['kind'],seed=lib['seed'],
        knot_progress='prefix-1.5,prefix,(prefix+turn_end)/2,turn_end,turn_end+1.5',
        state_lanes_verified_equal=equal,full_horizon_reference_authority=True,original_source=str(source)))
    models={}
    for i,name in enumerate(lib['names']):
        file=root/(name+'_targets.pkl')
        with file.open('wb') as f:pickle.dump(dict(targets=lib['targets'][i]),f)
        models[name]=dict(actor=str(file),actor_sha256=digest(file),runtime_kind='reference_sequence',
            kind='fixed_reference_process_not_a_trained_Actor')
    atomic_json(root/'authenticated_models.json',dict(models=models,inherited_models=auth['models']))
    atomic_json(root/'reuse_cases.json',reuse)
    atomic_json(root/'inventory.json',dict(project='STTW_CONTROL',cases=rows,new_actual_cases=74,inherited_cases=10,
        logical_cases=84,scope='One known DEV context, zero force only; no deployment or safety claim'))
    files=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/evaluate_anticipatory_reference.py').resolve(),
        Path('learning/cli/prepare_full_task_sequence.py').resolve()]
    archive=root/'source_snapshot';archive.mkdir()
    for i,p in enumerate(sorted(files)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    names=['common_environment_config.json','source_declaration.json','candidate_targets.npz','candidate_definitions.json',
        'authenticated_models.json','inventory.json','reuse_cases.json']
    plan=dict(project='STTW_CONTROL',data_role='DEV',purpose='Is there useful full-task finite reference coordination beyond strong trained and constant controls before adding disturbance objectives?',
        stage_counts=dict(sequence_nominal=74),logical_cases=84,inherited_cases=10,maximum_active_lower_ticks=148000,
        optimizer_updates=0,batch_size=16,steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,
        allow_task0_alpha_context_override=False,source_run=str(source),
        source_initial_state=old['source_initial_state'],source_initial_state_sha256=old['source_initial_state_sha256'],
        path_file=old['path_file'],path_sha256=old['path_sha256'],
        sealed_files={**model_files,**{str((root/name).resolve()):digest(root/name) for name in names}},
        source_sha256={str(p.resolve()):digest(p) for p in files},
        contract='Same complete source, original user alpha/path/speed,50Hz sample hold and four5ms physical ticks; spline/constant/zero reference processes all execute across cumulative Actor domain exit; contact/fault/10s stop. Original Actor baselines retain their original live-zero fallback.',
        limitation='Reference processes have full-horizon geometric support, unlike old Actor domain. Same rights constant/zero comparisons required; no tolerance target is used. Two useralpha runs kept separate; no assumed crossalpha physics dedup.',
        stop_rule='74 actual episodes only, no automatic retry, candidate expansion, parameter fitting or optimizer. No eventwise future-informed winner is a deployment method.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=74,logical_cases=84,inherited_cases=10,
        completed_cases=0,active_lower_ticks=0,optimizer_updates=0,checked_unix=time.time()))
