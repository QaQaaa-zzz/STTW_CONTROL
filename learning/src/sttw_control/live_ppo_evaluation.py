"""Prepare the predeclared fixed64 DEV comparison; never select checkpoints."""
import json
import shutil
from pathlib import Path
from .preference_best import atomic_json,digest


def evaluation_inventory(declared,context,*,reference_floor):
    rows=[]
    for c in declared:
        if c['context']!=context:continue
        event='zero' if c['force_N']==0 else f"t{c['pulse_onset_task_tick']}_F{c['force_N']:g}_rep{c['replicate']}"
        role=c['actor_role'];key=f"{role}/a{c['alpha']:g}/{event}"
        rows.append(dict(c,case_id=key,role=role,actor_role=role,event_id=event,
            state_alpha=c['alpha'],requested_speed_m_s=c['speed'],source_batch_slot=0,
            reference_speed_floor_m_s=reference_floor,stage='reentry',zero_mode=False))
    if len(rows)!=100 or len({r['case_id'] for r in rows})!=100:
        raise ValueError('exact100 unique route cases required')
    return rows


def prepare(training,output):
    """Seal separately, only once the entire paired training is terminal."""
    training=Path(training).resolve();output=Path(output).resolve()
    status=json.loads((training/'status.json').read_text())
    if status['phase']!='completed' or status['completed_updates']!=64:
        raise ValueError('complete paired fixed64 training required; no partial/best selection')
    plan=json.loads((training/'plan.json').read_text())
    assert digest(training/'plan.json')==json.loads((training/'prelaunch_receipt.json').read_text())['plan_sha256']
    declared=json.loads((training/'evaluation_cases.json').read_text())
    assert digest(training/'evaluation_cases.json')==plan['files_sha256']['evaluation_cases.json']
    end=json.loads((training/'checkpoint_pair_0064.json').read_text())
    assert end['completed_updates']==64 and end['plan_sha256']==digest(training/'plan.json')
    models={x['arm']:x for x in end['models']}
    assert set(models)=={'plain','support'}
    all_models={r:dict(plan['model']) for r in ['parent_plain','parent_support']}
    all_models.update({f'trained_{n}':dict(m) for n,m in models.items()})
    cfg=json.loads((training/'effective_config.json').read_text())
    assert digest(training/'effective_config.json')==plan['files_sha256']['effective_config.json']
    for m in all_models.values():
        for k in ['actor','checkpoint']:assert digest(m[k])==m[k+'_sha256']
        m.update(runtime_kind='reentry',actor_config=cfg)
    sources=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/evaluate_anticipatory_reference.py').resolve()]
    output.mkdir(parents=True,exist_ok=False)
    route_plans={}
    for context,source in plan['contexts'].items():
        for k in ['initial','path']:assert digest(source[k])==source[k+'_sha256']
        root=output/context;root.mkdir()
        shutil.copyfile(training/'effective_config.json',root/'common_environment_config.json')
        shutil.copyfile(training/'source_declaration.json',root/'source_declaration.json')
        cases=evaluation_inventory(declared,context,reference_floor=cfg['residual_interface']['speed_reference_bounds_m_s'][0])
        atomic_json(root/'authenticated_models.json',dict(models=all_models))
        atomic_json(root/'inventory.json',dict(project='STTW_CONTROL',cases=cases))
        atomic_json(root/'reuse_cases.json',[])
        sealed=['common_environment_config.json','source_declaration.json','authenticated_models.json','inventory.json','reuse_cases.json']
        p=dict(project='STTW_CONTROL',data_role='DEV_fixed64',purpose='Ordinary live PPO support adaptation versus its own frozen source under common permissions',
            maximum_active_lower_ticks=200000,optimizer_updates=0,batch_size=50,steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,
            stage_counts={'reentry':100},logical_cases=100,inherited_cases=0,allow_task0_alpha_context_override=True,
            comparison_baseline_role='parent_plain',comparison_baseline_by_role={'trained_plain':'parent_plain','trained_support':'parent_support','parent_support':'parent_plain'},
            source_initial_state=source['initial'],source_initial_state_sha256=source['initial_sha256'],path_file=source['path'],path_sha256=source['path_sha256'],
            sealed_files={str(root/n):digest(root/n) for n in sealed},source_sha256={str(f.resolve()):digest(f) for f in sources},
            training_plan_sha256=digest(training/'plan.json'),final_pair_sha256=digest(training/'checkpoint_pair_0064.json'),
            contract='Originalalpha task0/physical700; meanActor; common16clean handback; all reference limits/history intact; no future pulse input',
            stop_rule='Single100 cases/route, no automatic retry/extra force or horizon; known contacts retained; unknown stops',
            inference='Two old routes and one related physical origin; event times matched across four roles within alpha, not across differentalpha. Independent random DEV schedule, not geometry generalization or training-seed replication.')
        atomic_json(root/'plan.json',p);atomic_json(root/'status.json',dict(phase='prepared',declared_cases=100,completed_cases=0,active_lower_ticks=0,optimizer_updates=0))
        route_plans[context]=dict(path=str(root),plan_sha256=digest(root/'plan.json'))
    atomic_json(output/'plan.json',dict(project='STTW_CONTROL',training=str(training),routes=route_plans,
        declared_cases=200,maximum_active_lower_ticks=400000,optimizer_updates=0,
        evaluation_schedule_sha256=digest(training/'evaluation_cases.json'),stage='reentry',analysis_status='not_started',
        note='Use common fivealpha analysis; older anticipatory_analysis only supports twoalpha XY and must not be called blindly.'))
    atomic_json(output/'status.json',dict(phase='prepared',declared_cases=200,completed_cases=0,optimizer_updates=0))
