"""Prepare the fixed10-episode causal expert-reference borrowing screen."""
import json
import pickle
import time
from pathlib import Path
from .preference_best import atomic_json,digest


def relay_cases(templates):
    if len(templates)!=5 or len({r['event_id'] for r in templates})!=5:
        raise ValueError('five unique previously declared events required')
    return [dict(t,case_id=f'relay_dual{a}/alpha0/{t["event_id"]}',role=f'relay_dual{a}',
        actor_role=f'relay_dual{a}',alpha=0.,state_alpha=0.,profile=0,zero_mode=False,
        teacher_alpha_context=float(a),stage='expert_relay') for a in [0,1] for t in templates]


def prepare_relay(source,history,root):
    from .path_command_policy import validate_actor_parameters
    source=source.resolve();history=history.resolve();root=root.resolve()
    status=json.loads((source/'status.json').read_text())
    hs=json.loads((history/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=80 or hs['phase']!='completed' or hs['completed_cases']!=35:
        raise ValueError('complete80 and complete35 lineage sources required')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    if json.loads((history/'common_environment_config.json').read_text())!=cfg:raise ValueError('common runtime drift')
    source_plan=json.loads((source/'plan.json').read_text())
    for name,sha in source_plan['sealed_files'].items():
        if digest(name)!=sha:raise ValueError('source declaration drift '+name)
    for name,sha in source_plan['source_sha256'].items():
        if Path(name).name in ['anticipatory_run.py','evaluate_anticipatory_reference.py']:continue
        if digest(name)!=sha:raise ValueError('cached physical/Actor source drift '+name)
    inventory=json.loads((source/'inventory.json').read_text())
    rows=relay_cases([r for r in inventory['cases'] if r['role']=='parent' and r['alpha']==0.])
    auth=json.loads((history/'authenticated_models.json').read_text())
    selected={k:auth['models'][k] for k in ['parent','dual_a0','dual_a1']};original_files={}
    weights={}
    for role,m in selected.items():
        for kind in ['actor','checkpoint']:
            if digest(m[kind])!=m[kind+'_sha256']:raise ValueError('changed frozen '+role+'/'+kind)
            original_files[m[kind]]=m[kind+'_sha256']
        if digest(m['actor_training_config_path'])!=m['actor_training_config_sha256']:
            raise ValueError('changed frozen Actor config')
        original_files[m['actor_training_config_path']]=m['actor_training_config_sha256']
        with Path(m['actor']).open('rb') as f:weights[role]=pickle.load(f)
        validate_actor_parameters(weights[role],m['actor_config'])
    old_models=json.loads((source/'authenticated_models.json').read_text())['models']
    if old_models['parent']['actor_sha256']!=selected['parent']['actor_sha256']:
        raise ValueError('H75 baseline identity drift')
    reuse=[]
    for stage in ['nominal','disturbed']:
        rr=json.loads((source/stage/'results.json').read_text())
        index={r['case_id']:r for r in json.loads((source/stage/'case_index.json').read_text())}
        for r in rr:
            if r['role'] not in ['parent','causal_eso'] or r['alpha']!=0.:continue
            item=index[r['case_id']];file=source/item['file']
            if digest(file)!=item['sha256']:raise ValueError('cached raw drift')
            reuse.append(dict(case=r,index=dict(item,file=str(file.resolve())),source_run=str(source)))
    if len(reuse)!=10:raise ValueError('ten baseline traces required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ['common_environment_config.json','source_declaration.json']:
        (root/name).write_bytes((source/name).read_bytes())
    import numpy as np
    models={}
    for a in [0,1]:
        role=f'relay_dual{a}';teacher=selected[f'dual_a{a}'];file=root/(role+'_parameter_bundle.pkl')
        with file.open('wb') as f:pickle.dump(dict(base=weights['parent'],teacher=weights[f'dual_a{a}'],teacher_alpha=np.float32(a)),f)
        models[role]=dict(actor=str(file),actor_sha256=digest(file),runtime_kind='causal_expert',
            actor_config=selected['parent']['actor_config'],teacher_actor_config=teacher['actor_config'],
            component_models=dict(base=selected['parent'],teacher=teacher),
            bundle_kind='Unchanged two frozen Actor trees and teacher query context; no fitted or blended weights.')
    atomic_json(root/'authenticated_models.json',dict(models=models))
    atomic_json(root/'inventory.json',dict(project='STTW_CONTROL',cases=rows,new_actual_cases=10,inherited_cases=10,
        logical_cases=20,user_alpha_fixed=0.,gate='Original causal_reference_baseline; no threshold/duration/direction changes.'))
    atomic_json(root/'reuse_cases.json',reuse)
    files=list(Path(__file__).parent.glob('*.py'))+[
        Path('learning/cli/evaluate_anticipatory_reference.py').resolve(),Path('learning/cli/prepare_expert_relay.py').resolve()]
    archive=root/'source_snapshot';archive.mkdir()
    for i,p in enumerate(sorted(files)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    plan=dict(project='STTW_CONTROL',purpose='Test whether short causal borrowing from historical fixed endpoints adds useful task/physical outcomes beyond the original ESO rule.',
        data_role='DEV',stage_counts=dict(expert_relay=10),logical_cases=20,inherited_cases=10,
        maximum_active_lower_ticks=20000,optimizer_updates=0,batch_size=16,steps_per_chunk=50,max_chunks=10,
        horizon_s=10.,lower_dt_s=.005,allow_task0_alpha_context_override=False,user_alpha_fixed=0.,
        source_initial_state=source_plan['source_initial_state'],source_initial_state_sha256=source_plan['source_initial_state_sha256'],
        path_file=source_plan['path_file'],path_sha256=source_plan['path_sha256'],
        source_run=str(source),lineage_source=str(history),
        sealed_files={**original_files,**{str((root/name).resolve()):digest(root/name) for name in
            ['common_environment_config.json','source_declaration.json','authenticated_models.json','inventory.json','reuse_cases.json']}},
        source_sha256={str(p.resolve()):digest(p) for p in files},
        contract=source_plan['contract'],
        control='Original ESO first event and right-turn negative-innovation condition, fixed80ticks/.4s. Publish both teacher latent channels through original correction_tick; handback to H75. User alpha0 remains unchanged. Teacher query context0/1 is internal only.',
        information='Only actual current/past observation to gate/Actors; actual force/onset enters physics only.',
        stop_rule='Ten actual episodes only, no optimizer, retuning, teacher library expansion, horizon extension or automatic retry.',
        limitation='Related one-context DEV screen; cross-policy state transfer unvalidated before rollout. Both-channel teacher vs speed-only ESO difference must be disclosed.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=10,logical_cases=20,inherited_cases=10,
        completed_cases=0,active_lower_ticks=0,optimizer_updates=0,checked_unix=time.time()))
