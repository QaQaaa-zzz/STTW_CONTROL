"""One fixed DEV permission-factor comparison; no search or optimizer."""
import copy
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest

CASE_FIELDS=('case_id','context','route','geometry_name','prefix_name','alpha','force_N',
    'pulse_onset_task_tick','pulse_duration_s','world_direction','requested_speed_m_s',
    'role','actor_role','mode_index','profile','causal_support','feedback_gain_rad',
    'goal_preview','source_role','source_keep_channels','event_id','replicate')
CHANGED_OLD_SOURCES={'anticipatory_reference.py','reference_bias_runtime.py',
    'reference_bias_optimization.py','goal_preview_baseline.py','reference_task_feedback.py'}


def build_inventory(baselines):
    if len(baselines)!=108 or any(r['role']!='full' for r in baselines):
        raise ValueError('all108 authenticated full baseline rows required')
    groups={};cases=[]
    for row in baselines:
        base={k:copy.deepcopy(row[k]) for k in CASE_FIELDS}
        key=(base['context'],base['alpha'])
        events=groups.setdefault(key,set())
        event=(base['event_id'],base['force_N'])
        if event in events:raise ValueError('duplicate baseline event')
        events.add(event)
        if base['mode_index']!=1 or not base['causal_support'] or base['source_keep_channels']!=[True,True]:
            raise ValueError('original full neural channels and causal support required')
        candidate=dict(base,role='support_permission',support_independent_permission=True,
                       case_id=base['case_id'].rsplit('/',1)[0]+'/support_permission')
        cases.append(candidate)
    if len(groups)!=12 or {key[1] for key in groups}!={0.,.5,1.} or len({key[0] for key in groups})!=4:
        raise ValueError('allfour contexts and three original alphas required')
    wanted={('nominal',0.)}|{(f'{phase}_F{force:g}',float(force)) for phase in ['early','late'] for force in [-16,16,-32,32]}
    if any(events!=wanted for events in groups.values()):raise ValueError('fixed9 events per task required')
    return cases


def prepare(root,parent):
    from .reference_preservation_confirmation import audit_pulse
    root=Path(root).resolve();parent=Path(parent).resolve()
    if root.exists():raise ValueError('fresh run only')
    root.mkdir(parents=True)
    old=json.loads((parent/'plan.json').read_text())
    if digest(parent/'plan.json')!=json.loads((parent/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('old parent identity changed')
    for f,sha in old['sealed_files'].items():
        if digest(f)!=sha:raise ValueError('frozen input changed '+f)
    differences={f:dict(old=sha,current=digest(f)) for f,sha in old['source_sha256'].items() if digest(f)!=sha}
    if any(Path(f).name not in CHANGED_OLD_SOURCES for f in differences):raise ValueError('undeclared old runtime change')
    confirmation=parent/'preservation_timecheck_v1'
    state=json.loads((confirmation/'status.json').read_text())
    if state.get('phase')!='completed' or state.get('completed_physical_cases')!=240:
        raise ValueError('completed new-time parent required')
    rows=json.loads((confirmation/'analysis/results.json').read_text())
    baselines=[r for r in rows if r['role']=='full']
    cases=build_inventory(baselines)
    originals={x['case']['case_id']:x['case'] for x in json.loads((confirmation/'physical_inventory.json').read_text())}
    for row in baselines:
        if row['force_N'] and {k:row[k] for k in CASE_FIELDS}!=originals.get(row['case_id']):
            raise ValueError('baseline event differs from executed original physical inventory')
    # Verify every inherited raw and pulse, rather than trusting a summary rate.
    for row in baselines:
        if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('baseline raw changed')
        with np.load(row['raw_file']) as z:audit_pulse(z,row)
    for name in ['common_environment_config.json','source_declaration.json']:
        shutil.copy2(parent/name,root/name)
    atomic_json(root/'baseline_rows.json',baselines);atomic_json(root/'candidate_cases.json',cases)
    files=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/run_support_permission_factor.py').resolve()]
    source=root/'source';source.mkdir()
    for i,f in enumerate(files):shutil.copy2(f,source/f'{i:03d}_{f.name}')
    plan=dict(project='STTW_CONTROL',experiment='support_permission_factor_v1',
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_identity='working source SHA snapshot below includes declared uncommitted changes',
        question='Does retaining already-active analytic speed support during geometric NN disable improve survival/arrival?',
        parent=str(parent),parent_plan_sha256=digest(parent/'plan.json'),
        baseline_results_sha256=digest(confirmation/'analysis/results.json'),
        baseline_execution_binding=str(confirmation/'execution_binding.json'),
        old_source_differences=differences,source_sha256={str(f.resolve()):digest(f) for f in files},
        sealed_files={**old['sealed_files'],**{str((root/n).resolve()):digest(root/n) for n in
            ['common_environment_config.json','source_declaration.json','baseline_rows.json','candidate_cases.json']}},
        contexts=old['contexts'],model=old['model'],feedback_spec=old['feedback_spec'],
        horizon_s=10.,lower_dt_s=.005,real_batch_sizes=[108],new_episodes=108,
        maximum_new_actual_lower_ticks=216000,maximum_allocated_lower_slots=216000,optimizer_updates=0,
        inherited_costs=dict(parent_total_actual_lower_ticks=3780700,inherited_baseline_episodes=108),
        initialization='same complete task0 states; no reset at pulse, weights frozen, no optimizer',
        primary_evaluation='arrival before contact/fault within original10s worldgoal1m; full-window survival separate',
        sensitivity_radii_m=[.5,.75,1.25],data_role='seen DEV permission-factor diagnosis, not final independent TEST',
        stop='engineering error retains partial cost and stops; no implicit retry; no threshold/duration sweep',
        adoption='use all gain/loss and original-alpha task cost; positive first upgrades a simple baseline, not a novelty claim',
        information='controller receives current measured state and original references only; no force/onset query',
        implementation_status='prepared; actual108 shape and numerical execution pending')
    atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',stage='shape_pending',completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    return plan


def inputs(root):
    root=Path(root).resolve();p=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('changed fixed plan')
    for f,sha in {**p['source_sha256'],**p['sealed_files']}.items():
        if digest(f)!=sha:raise ValueError('changed sealed source/input '+f)
    cases=json.loads((root/'candidate_cases.json').read_text())
    if cases!=build_inventory(json.loads((root/'baseline_rows.json').read_text())):
        raise ValueError('candidate changes more than declared permission factor')
    return root,p,cases


def shape(root):
    import jax
    from .reference_bias_runtime import BiasRuntime
    root,p,cases=inputs(root);out=root/'shape';out.mkdir(exist_ok=False)
    runtime=BiasRuntime(root)
    args,_,_=runtime.batch_inputs(cases,np.zeros((108,6)),allowed_real_batch_sizes=(108,))
    end,logs=jax.eval_shape(runtime.chunk,*args)
    if logs['support_permission_override'].shape!=(50,108,4):raise ValueError('actual batch shape changed')
    atomic_json(out/'receipt.json',dict(batch=108,physics_steps=0,optimizer_updates=0,
        plan_sha256=digest(root/'plan.json'),scope='All108 real inputs/model shape trace, not numerical integration or timing',
        permission_log_shape=list(logs['support_permission_override'].shape)))
    atomic_json(root/'status.json',dict(phase='prepared',stage='execution_ready',completed_cases=0,
        active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def execute(root):
    from .reference_bias_runtime import BiasRuntime
    root,p,cases=inputs(root)
    if json.loads((root/'status.json').read_text()).get('stage')!='execution_ready':
        raise ValueError('fresh prepared shape-checked run required')
    receipt=json.loads((root/'shape/receipt.json').read_text())
    if receipt['plan_sha256']!=digest(root/'plan.json') or receipt['batch']!=108 or receipt['physics_steps']:
        raise ValueError('shape identity changed')
    state=dict(phase='running',stage='initializing',pid=os.getpid(),completed_cases=0,optimizer_updates=0)
    start=time.monotonic();atomic_json(root/'status.json',state)
    def save():
        batch=root/'physical/batch_00/status.json'
        physical=json.loads(batch.read_text()) if batch.exists() else {}
        for k in ['active_lower_ticks','allocated_lower_slots']:state[k]=physical.get(k,0)
        state['elapsed_s']=time.monotonic()-start
        atomic_json(root/'status.json',state)
    try:
        runtime=BiasRuntime(root)
        state.update(stage='fixed_evaluation');save()
        rows=runtime.rollout(cases,np.zeros((108,6)),root/'physical/batch_00',allowed_real_batch_sizes=(108,))
        if len(rows)!=108:raise ValueError('complete108 required')
        atomic_json(root/'candidate_results.json',rows)
        state.update(phase='completed',stage='numeric_complete_analysis_pending',completed_cases=108);save()
        if max(state['active_lower_ticks'],state['allocated_lower_slots'])>216000:raise ValueError('fixed episode allocation exceeded')
    except BaseException as exc:
        (root/'traceback.txt').write_text(traceback.format_exc())
        state.update(phase='error',error=repr(exc));save();raise
