"""Authenticate and prepare a bounded historical endpoint comparison."""
import json
import time
from pathlib import Path
from .preference_best import atomic_json,digest


def historical_cases(templates):
    if len(templates)!=5 or len({r['event_id'] for r in templates})!=5:
        raise ValueError('five unique previously declared events required')
    combinations=[('dual',a,f'dual_a{int(a)}') for a in [0.,1.]]
    combinations += [(role,a,role) for role in ['bc','v3'] for a in [0.,1.]]
    combinations += [('parent',1.,'parent')]
    return [dict(t,case_id=f'history/{role}/alpha{alpha}/{t["event_id"]}',
        role=role,actor_role=actor,alpha=alpha,state_alpha=alpha,stage='endpoints',profile=0,zero_mode=False)
        for role,alpha,actor in combinations for t in templates]


def prepare_history(source,root,registry):
    from .path_command_policy import validate_actor_parameters
    import pickle
    source=source.resolve();root=root.resolve();registry=registry.resolve()
    status=json.loads((source/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=80:
        raise ValueError('complete80 source collection required')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    source_plan=json.loads((source/'plan.json').read_text())
    inventory=json.loads((source/'inventory.json').read_text())
    templates=[r for r in inventory['cases'] if r['role']=='parent' and r['alpha']==0.]
    rows=historical_cases(templates)
    reg=json.loads(registry.read_text());frozen=Path(reg['root'])
    if digest(frozen/'manifest.json')!=reg['manifest_sha256']:raise ValueError('frozen registry drift')
    manifest=json.loads((frozen/'manifest.json').read_text())
    nodes={r['id']:r for r in manifest['nodes']}
    names=dict(dual_a0='DUAL_COVERAGE_U200_A0',dual_a1='DUAL_COVERAGE_U200_A1',
               bc='DISTILLED_BC',v3='DISTILLED_V3_U325',parent='FULLTAIL_PPO75_U77')
    models={};compatibility=[]
    for role,suffix in names.items():
        identifier='STTW_MIDDLE_'+suffix;record=reg['records'][identifier];node=nodes[identifier]
        for kind in ['actor','checkpoint']:
            if digest(record[kind])!=record[kind+'_sha256']:raise ValueError('changed frozen model '+identifier)
        cp=frozen/node['resolved_config']['path']
        if digest(cp)!=node['resolved_config']['sha256']:raise ValueError('changed frozen config '+identifier)
        actor_cfg=json.loads(cp.read_text())
        for key in ['history_frames','frame_fields','mask_fields','actor_dim','context_fields','hidden','activation','actions']:
            if actor_cfg['network'][key]!=cfg['network'][key]:raise ValueError('incompatible observation/network field '+identifier+'/'+key)
        with Path(record['actor']).open('rb') as f:weights=pickle.load(f)
        validate_actor_parameters(weights,actor_cfg)
        models[role]=dict(record,actor_config=actor_cfg,frozen_id=identifier,
            actor_training_config_path=str(cp),actor_training_config_sha256=digest(cp))
        compatibility.append(dict(role=role,frozen_id=identifier,
            original_interface=actor_cfg['residual_interface'],common_evaluation_interface=cfg['residual_interface'],
            original_radius=actor_cfg['path']['turn_radius_m'],common_radius=cfg['path']['turn_radius_m']))
    # Verify exact model, environment and physical kernel compatibility before
    # inheriting the five factual H75 alpha0 traces, without replaying them.
    old_models=json.loads((source/'authenticated_models.json').read_text())['models']
    if old_models['parent']['actor_sha256']!=models['parent']['actor_sha256']:
        raise ValueError('H75 cache has different Actor')
    for name,sha in source_plan['source_sha256'].items():
        if Path(name).name in ['anticipatory_run.py','evaluate_anticipatory_reference.py']:continue
        if digest(name)!=sha:raise ValueError('cached physical/Actor source drift '+name)
    reuse=[]
    for stage in ['nominal','disturbed']:
        originals=json.loads((source/stage/'results.json').read_text())
        index={r['case_id']:r for r in json.loads((source/stage/'case_index.json').read_text())}
        for r in originals:
            if r['role']!='parent' or r['alpha']!=0.:continue
            item=index[r['case_id']];file=source/item['file']
            if digest(file)!=item['sha256']:raise ValueError('cached raw drift')
            reuse.append(dict(case=r,index=dict(item,file=str(file.resolve())),source_run=str(source)))
    if len(reuse)!=5:raise ValueError('exactly five inherited parent cases required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ['common_environment_config.json','source_declaration.json']:
        (root/name).write_bytes((source/name).read_bytes())
    atomic_json(root/'authenticated_models.json',dict(models=models,compatibility=compatibility,
        registry=str(registry),registry_sha256=digest(registry),manifest_sha256=reg['manifest_sha256']))
    atomic_json(root/'inventory.json',dict(project='STTW_CONTROL',cases=rows,
        scientific_scope='Historical fixed endpoints transferred to common runtime. Not a pure architecture ablation.',
        new_actual_cases=35,inherited_cases=5,logical_comparisons=40))
    atomic_json(root/'reuse_cases.json',reuse)
    files=list(Path(__file__).parent.glob('*.py'))+[
        Path('learning/cli/evaluate_anticipatory_reference.py').resolve(),
        Path('learning/cli/prepare_middle_lineage_evaluation.py').resolve()]
    archive=root/'source_snapshot';archive.mkdir()
    for i,p in enumerate(sorted(files)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    plan=dict(project='STTW_CONTROL',purpose='Check whether frozen dual/distilled/V3 endpoints offer reusable task-balance assets before developing another control method.',
        data_role='DEV',stage_counts=dict(endpoints=35),logical_cases=40,inherited_cases=5,
        maximum_active_lower_ticks=70000,optimizer_updates=0,batch_size=16,
        steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,
        allow_task0_alpha_context_override=True,
        context_change='Only task0 user alpha set to requested endpoint; physical/ESO/actuator/history copied intact. No mid-episode alpha switching.',
        source_initial_state=source_plan['source_initial_state'],source_initial_state_sha256=source_plan['source_initial_state_sha256'],
        path_file=source_plan['path_file'],path_sha256=source_plan['path_sha256'],
        sealed_files={str((root/name).resolve()):digest(root/name) for name in
            ['common_environment_config.json','source_declaration.json','authenticated_models.json','inventory.json','reuse_cases.json']},
        source_sha256={str(p.resolve()):digest(p) for p in files},
        source_run=str(source),source_plan_sha256=digest(source/'plan.json'),
        contract=source_plan['contract'],
        stop_rule='35 actual episodes only, no optimizer, candidate expansion, automatic retries or horizon extension.',
        limitation='Different training histories and reference limits; common-runtime endpoint capability, not an architecture-only causal comparison.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=35,logical_cases=40,inherited_cases=5,
        completed_cases=0,active_lower_ticks=0,optimizer_updates=0,checked_unix=time.time()))
